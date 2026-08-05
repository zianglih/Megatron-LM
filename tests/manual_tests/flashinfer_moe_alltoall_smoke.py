# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed B200 smoke test for FlashInfer MoE with Megatron all-to-all.

Run from the Megatron-LM root, for example:

    torchrun --standalone --nproc-per-node=8 \
        tests/manual_tests/flashinfer_moe_alltoall_smoke.py

The final EP rank intentionally receives no expert assignments while still
originating tokens. This exercises the M=0 kernel bypass and the inverse A2A.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["MILES_USE_FLASHINFER_MOE"] = "1"
os.environ["MILES_FLASHINFER_MOE_QUANTIZATION"] = "mxfp8"
os.environ["MILES_FLASHINFER_MOE_DEQUANTIZED"] = "0"

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from miles_megatron_plugins import flashinfer_moe


def _routing_ids(rank: int, world_size: int, tokens: int, experts: int) -> torch.Tensor:
    """Route away from the last EP rank while keeping two distinct owners."""

    active_owners = world_size - 1
    if active_owners < 2:
        raise RuntimeError("The all-to-all smoke test requires at least three ranks")
    if experts % world_size:
        raise RuntimeError(f"experts={experts} must be divisible by world_size={world_size}")
    local_experts = experts // world_size
    ids = []
    for token in range(tokens):
        base = (rank + token) % active_owners
        first_owner = (base + 1) % active_owners
        second_owner = (base + 2) % active_owners
        ids.append(
            [
                first_owner * local_experts + token % local_experts,
                second_owner * local_experts + (2 * token + 1) % local_experts,
            ]
        )
    return torch.tensor(ids, device="cuda", dtype=torch.long)


def _install_route(layer: MoELayer, route_logits: torch.Tensor, route_ids: torch.Tensor) -> None:
    """Install deterministic differentiable routing for one forward."""

    def route(_self, hidden_states, padding_mask=None, input_ids=None):
        del padding_mask, input_ids
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        if tokens != route_ids.shape[0]:
            raise RuntimeError(f"route token mismatch: {tokens} != {route_ids.shape[0]}")
        selected_probs = torch.softmax(route_logits, dim=-1)
        probs = torch.zeros(
            (tokens, _self.config.num_moe_experts),
            device=hidden_states.device,
            dtype=selected_probs.dtype,
        ).scatter(1, route_ids, selected_probs)
        routing_map = torch.zeros_like(probs, dtype=torch.bool)
        routing_map.scatter_(1, route_ids, True)
        return probs, routing_map

    layer.route = types.MethodType(route, layer)


def _bf16_reference(layer: MoELayer, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the same A2A lifecycle with the plugin's BF16 surrogate experts."""

    probs, routing_map = layer.route(hidden_states)
    hidden_states, probs = layer.preprocess(hidden_states, probs, routing_map)
    dispatched_input, probs = layer.dispatch(hidden_states, probs)
    dispatched_input, tokens_per_expert, permuted_probs = (
        layer.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
    )
    w13, w2 = flashinfer_moe._grouped_mlp_weights(layer.experts)
    weights, ids = flashinfer_moe._dispatched_topk_inputs(
        permuted_probs,
        tokens_per_expert,
        local_expert_offset=layer.local_expert_indices[0],
        num_local_experts=layer.num_local_experts,
    )
    output = flashinfer_moe._bf16_local_routed_experts(
        dispatched_input, weights, ids, w13, w2, layer.local_expert_indices[0]
    )
    output = layer.token_dispatcher.combine_preprocess(output)
    output = layer.combine(output)
    return layer.postprocess(output, shared_expert_output=None)


def _global_max(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"FlashInfer all-to-all smoke requires 8 ranks, got {world_size}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=world_size
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)

    experts = 128
    hidden_size = 2048
    intermediate_size = 768
    tokens = 8
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=16,
        num_moe_experts=experts,
        moe_ffn_hidden_size=intermediate_size,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_permute_fusion=False,
        moe_router_dtype="fp32",
        tensor_model_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )
    layer = MoELayer(
        config, MoESubmodules(experts=ModuleSpec(module=GroupedMLP)), layer_number=1
    ).cuda()
    layer.train()

    route_ids = _routing_ids(rank, world_size, tokens, experts)
    hidden_seed = torch.randn((tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
    logits_seed = torch.randn((tokens, 2), device="cuda", dtype=torch.float32)
    grad_seed = torch.randn_like(hidden_seed)

    hidden_ref = hidden_seed.detach().clone().requires_grad_()
    logits_ref = logits_seed.detach().clone().requires_grad_()
    _install_route(layer, logits_ref, route_ids)
    reference = _bf16_reference(layer, hidden_ref)
    torch.autograd.backward(reference, grad_seed)
    ref_param_grads = [parameter.grad.detach().clone() for parameter in layer.experts.parameters()]

    layer.zero_grad(set_to_none=True)
    hidden_actual = hidden_seed.detach().clone().requires_grad_()
    logits_actual = logits_seed.detach().clone().requires_grad_()
    _install_route(layer, logits_actual, route_ids)
    received_rows = []
    hook = layer.experts.register_forward_pre_hook(
        lambda _module, inputs: received_rows.append(inputs[0].shape[0])
    )
    with (
        mock.patch.object(
            flashinfer_moe._PaddedEPAllGather,
            "apply",
            side_effect=AssertionError("plugin padded all-gather used in all-to-all mode"),
        ),
        mock.patch.object(
            flashinfer_moe._EPAllReduceSum,
            "apply",
            side_effect=AssertionError("plugin EP all-reduce used in all-to-all mode"),
        ),
    ):
        actual, bias = layer(hidden_actual)
    hook.remove()
    torch.autograd.backward(actual, grad_seed)
    actual_param_grads = [
        parameter.grad.detach().clone() for parameter in layer.experts.parameters()
    ]

    forward_error_sq = (actual.float() - reference.float()).square().sum()
    reference_sq = reference.float().square().sum()
    torch.distributed.all_reduce(forward_error_sq)
    torch.distributed.all_reduce(reference_sq)
    forward_rel_l2 = torch.sqrt(forward_error_sq / reference_sq.clamp_min(1e-20))

    hidden_grad_error = _global_max(
        (hidden_actual.grad.float() - hidden_ref.grad.float()).abs().max()
    )
    route_grad_error = _global_max(
        (logits_actual.grad.float() - logits_ref.grad.float()).abs().max()
    )
    parameter_grad_error = torch.zeros((), device="cuda")
    for actual_grad, ref_grad in zip(actual_param_grads, ref_param_grads):
        parameter_grad_error = torch.maximum(
            parameter_grad_error, (actual_grad.float() - ref_grad.float()).abs().max()
        )
    parameter_grad_error = _global_max(parameter_grad_error)

    received = torch.tensor(
        received_rows if len(received_rows) == 1 else [-1], device="cuda", dtype=torch.int64
    )
    received_tensors = [torch.empty_like(received) for _ in range(world_size)]
    torch.distributed.all_gather(received_tensors, received)
    received_by_rank = [value.item() for value in received_tensors]
    output_nonzero = torch.tensor(
        [int(torch.count_nonzero(actual).item() > 0)], device="cuda", dtype=torch.int32
    )
    output_nonzero_tensors = [torch.empty_like(output_nonzero) for _ in range(world_size)]
    torch.distributed.all_gather(output_nonzero_tensors, output_nonzero)
    output_nonzero_by_rank = [value.item() for value in output_nonzero_tensors]

    if bias is not None:
        raise AssertionError("unexpected expert bias")
    if any(received <= 0 for received in received_by_rank[:-1]):
        raise AssertionError(f"active expert rank received no assignments: {received_by_rank}")
    if received_by_rank[-1] != 0:
        raise AssertionError(f"zero-receive rank got assignments: {received_by_rank}")
    if sum(received_by_rank) != world_size * tokens * 2:
        raise AssertionError(f"assignment count mismatch: {received_by_rank}")
    if output_nonzero_by_rank[-1] != 1:
        raise AssertionError("zero-receive rank lost its inverse-A2A token outputs")
    if forward_rel_l2.item() >= 0.10:
        raise AssertionError(f"MXFP8 forward relative L2 too large: {forward_rel_l2.item():.6f}")
    if hidden_grad_error.item() >= 0.02:
        raise AssertionError(f"hidden surrogate gradient mismatch: {hidden_grad_error.item():.6f}")
    if route_grad_error.item() >= 0.02:
        raise AssertionError(f"route surrogate gradient mismatch: {route_grad_error.item():.6f}")
    if parameter_grad_error.item() >= 0.02:
        raise AssertionError(
            f"parameter surrogate gradient mismatch: {parameter_grad_error.item():.6f}"
        )

    if rank == 0:
        print(
            "FlashInfer all-to-all smoke passed: "
            f"world={world_size}, forward_rel_l2={forward_rel_l2.item():.6f}, "
            f"hidden_grad_max={hidden_grad_error.item():.6f}, "
            f"routing_logit_grad_max={route_grad_error.item():.6f}, "
            f"parameter_grad_max={parameter_grad_error.item():.6f}"
        )

    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
