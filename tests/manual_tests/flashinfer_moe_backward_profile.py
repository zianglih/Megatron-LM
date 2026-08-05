# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Non-gating 8-GPU profile for FlashInfer MoE backward operand modes."""

from __future__ import annotations

import gc
import os
import time
import types

import torch
import torch.nn.functional as F

os.environ["MILES_USE_FLASHINFER_MOE"] = "1"

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

NUM_EXPERTS = 32
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
TOP_K = 8
NUM_TOKENS = 4096
ITERATIONS = 3
OUTSTANDING_FORWARDS = (1, 4)
BACKWARD_MODES = ("high_precision", "dequantized")


def _routing_ids(rank: int, world_size: int) -> torch.Tensor:
    local_experts = NUM_EXPERTS // world_size
    active_experts = NUM_EXPERTS - local_experts
    token_ids = torch.arange(NUM_TOKENS, device="cuda", dtype=torch.long)
    slot_ids = torch.arange(TOP_K, device="cuda", dtype=torch.long)
    bases = (rank * NUM_TOKENS + token_ids) * TOP_K
    return ((bases.unsqueeze(1) + slot_ids) % active_experts).contiguous()


def _install_route(layer: MoELayer, route_logits: torch.Tensor, route_ids: torch.Tensor) -> None:
    def route(_self, hidden_states, padding_mask=None, input_ids=None):
        del padding_mask, input_ids
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        background = torch.zeros((tokens, 1), device=route_logits.device, dtype=route_logits.dtype)
        selected_probs = torch.softmax(torch.cat((route_logits, background), dim=-1), dim=-1)[
            :, :-1
        ]
        probs = torch.zeros(
            (tokens, NUM_EXPERTS), device=hidden_states.device, dtype=selected_probs.dtype
        ).scatter(1, route_ids, selected_probs)
        routing_map = torch.zeros_like(probs, dtype=torch.bool)
        routing_map.scatter_(1, route_ids, True)
        return probs, routing_map

    layer.route = types.MethodType(route, layer)


def _build_layer(
    dispatcher: str, quantization: str, backward_mode: str, world_size: int
) -> MoELayer:
    if backward_mode not in BACKWARD_MODES:
        raise ValueError(f"unsupported backward mode {backward_mode!r}")
    os.environ["NVTE_BACKWARD_OVERRIDE"] = backward_mode
    if quantization == "mxfp8":
        precision_config = {"fp8": "e4m3", "fp8_recipe": "mxfp8"}
    elif quantization == "nvfp4":
        precision_config = {"fp4": "e2m1", "fp4_recipe": "nvfp4"}
        os.environ.update(
            {
                "NVTE_NVFP4_4OVER6": "all",
                "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
                "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
                "FLASHINFER_NVFP4_4OVER6": "1",
                "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
                "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
            }
        )
    else:
        raise ValueError(f"unsupported quantization {quantization!r}")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=16,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=INTERMEDIATE_SIZE,
        moe_router_topk=TOP_K,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type=dispatcher,
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
        **precision_config,
    )
    return MoELayer(
        config,
        MoESubmodules(
            experts=ModuleSpec(
                module=GroupedMLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
                ),
            )
        ),
        layer_number=1,
    ).cuda()


def _forward(
    layer: MoELayer, hidden_seed: torch.Tensor, logits_seed: torch.Tensor, route_ids: torch.Tensor
) -> torch.Tensor:
    hidden = hidden_seed.detach().clone().requires_grad_()
    route_logits = logits_seed.detach().clone().requires_grad_()
    _install_route(layer, route_logits, route_ids)
    layer_no = layer.layer_number - 1
    if layer.config.fp8:
        quantization_context = get_fp8_context(layer.config, layer_no)
    elif layer.config.fp4:
        quantization_context = get_fp4_context(layer.config, layer_no)
    else:
        raise ValueError("FlashInfer MoE profile requires an FP8 or FP4 configuration")
    with quantization_context:
        output, bias = layer(hidden)
    if bias is not None:
        raise RuntimeError("unexpected expert bias")
    return output


def _step(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    route_ids: torch.Tensor,
) -> None:
    layer.zero_grad(set_to_none=True)
    output = _forward(layer, hidden_seed, logits_seed, route_ids)
    torch.autograd.backward(output, grad_seed)


def _measure(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    route_ids: torch.Tensor,
) -> tuple[float, float]:
    _step(layer, hidden_seed, logits_seed, grad_seed, route_ids)
    layer.zero_grad(set_to_none=True)
    torch.distributed.barrier()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    torch.distributed.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _step(layer, hidden_seed, logits_seed, grad_seed, route_ids)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / ITERATIONS
    peak_mib = (torch.cuda.max_memory_allocated() - baseline) / (1024**2)

    values = torch.tensor([elapsed_ms, peak_mib], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return values[0].item(), values[1].item()


def _measure_outstanding_memory(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    route_ids: torch.Tensor,
    *,
    outstanding: int,
) -> tuple[float, float]:
    """Measure live post-forward memory and total forward/backward peak."""

    _step(layer, hidden_seed, logits_seed, grad_seed, route_ids)
    layer.zero_grad(set_to_none=True)
    torch.distributed.barrier()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    outputs = [_forward(layer, hidden_seed, logits_seed, route_ids) for _ in range(outstanding)]
    torch.cuda.synchronize()
    forward_live_mib = (torch.cuda.memory_allocated() - baseline) / (1024**2)
    torch.autograd.backward(outputs, [grad_seed for _ in outputs])
    torch.cuda.synchronize()
    peak_mib = (torch.cuda.max_memory_allocated() - baseline) / (1024**2)

    values = torch.tensor([forward_live_mib, peak_mib], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    layer.zero_grad(set_to_none=True)
    del outputs
    return values[0].item(), values[1].item()


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError(f"expected 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    route_ids = _routing_ids(rank, world_size)

    # Preserve each case's seed and backward-mode order while grouping MXFP8 before NVFP4.
    for dispatcher, quantization, profile_seed_offset in (
        ("alltoall", "mxfp8", 1),
        ("allgather", "mxfp8", 3),
        ("alltoall", "nvfp4", 0),
        ("allgather", "nvfp4", 2),
    ):
        torch.manual_seed(5678 + rank)
        hidden_seed = torch.randn((NUM_TOKENS, 1, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
        logits_seed = torch.randn((NUM_TOKENS, TOP_K), device="cuda", dtype=torch.float32)
        grad_seed = torch.randn_like(hidden_seed)
        results = {}
        memory_results = {}
        mode_order = (
            BACKWARD_MODES if profile_seed_offset % 2 == 0 else BACKWARD_MODES[::-1]
        )
        for backward_mode in mode_order:
            # Construct each mode independently with identical parameters. The
            # backward policy is model-static and must be selected before the
            # FlashInfer expert module is initialized.
            torch.manual_seed(1234 + profile_seed_offset)
            model_parallel_cuda_manual_seed(1234 + profile_seed_offset)
            layer = _build_layer(dispatcher, quantization, backward_mode, world_size)
            layer.train()
            results[backward_mode] = _measure(layer, hidden_seed, logits_seed, grad_seed, route_ids)
            memory_results[backward_mode] = {
                outstanding: _measure_outstanding_memory(
                    layer, hidden_seed, logits_seed, grad_seed, route_ids, outstanding=outstanding
                )
                for outstanding in OUTSTANDING_FORWARDS
            }
            del layer
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            high_ms, high_mib = results["high_precision"]
            deq_ms, deq_mib = results["dequantized"]
            print(
                "FlashInfer backward profile: "
                f"quantization={quantization}, dispatcher={dispatcher}, "
                f"tokens={NUM_TOKENS}, iterations={ITERATIONS}, "
                f"high_precision_ms={high_ms:.3f}, dequantized_ms={deq_ms:.3f}, "
                f"latency_delta_pct={(deq_ms / high_ms - 1) * 100:.2f}, "
                f"high_precision_peak_mib={high_mib:.2f}, "
                f"dequantized_peak_mib={deq_mib:.2f}, "
                f"peak_delta_mib={deq_mib - high_mib:.2f}",
                flush=True,
            )
            for outstanding in OUTSTANDING_FORWARDS:
                high_live, high_total_peak = memory_results["high_precision"][outstanding]
                deq_live, deq_total_peak = memory_results["dequantized"][outstanding]
                print(
                    "FlashInfer outstanding-forward memory: "
                    f"quantization={quantization}, dispatcher={dispatcher}, "
                    f"tokens={NUM_TOKENS}, outstanding={outstanding}, "
                    f"high_precision_forward_live_mib={high_live:.2f}, "
                    f"dequantized_forward_live_mib={deq_live:.2f}, "
                    f"forward_live_delta_mib={deq_live - high_live:.2f}, "
                    f"high_precision_total_peak_mib={high_total_peak:.2f}, "
                    f"dequantized_total_peak_mib={deq_total_peak:.2f}, "
                    f"total_peak_delta_mib={deq_total_peak - high_total_peak:.2f}",
                    flush=True,
                )
        del hidden_seed, logits_seed, grad_seed
        gc.collect()
        torch.cuda.empty_cache()

    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
