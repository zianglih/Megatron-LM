# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import statistics
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core import utils as core_utils
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.fp4_utils import get_fp4_context, get_fp4_recipe
from megatron.core.fp8_utils import get_fp8_context, get_fp8_recipe
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from miles_megatron_plugins.flashinfer_moe import (
    DEQUANTIZED_BACKWARD,
    HIGH_PRECISION_BACKWARD,
    FlashInferGroupedMLP,
    _BF16GroupedMLPSurrogate,
    _FlashInferMXFP8Runner,
    _FlashInferNVFP4Runner,
    maybe_replace_flashinfer_moe_expert_spec,
)
from tests.unit_tests.test_utilities import Utils


@dataclass(frozen=True)
class _PrecisionCase:
    configured: str
    execution: str
    num_layers: int = 1
    layer_no: int = 0
    first_last_layers_bf16: bool = False


@dataclass(frozen=True)
class _ModelHyperparameters:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int


@dataclass(frozen=True)
class _RoutingReplay:
    topk_ids: torch.Tensor
    routing_map: torch.Tensor


@dataclass(frozen=True)
class _LayerRun:
    output: torch.Tensor
    hidden_grad: torch.Tensor
    route_grad: torch.Tensor
    parameter_grads: tuple[torch.Tensor, ...]
    missing_grads: int
    received_rows: tuple[int, ...]


@dataclass(frozen=True)
class _NumericalMetrics:
    forward_rel_l2: float
    per_token_forward_rel_l2: float
    hidden_grad_rel_l2: float
    route_grad_rel_l2: float
    parameter_grad_rel_l2: float
    missing_grads: int


@dataclass(frozen=True)
class _MatchedPair:
    output: torch.Tensor
    metrics: _NumericalMetrics
    received_rows: tuple[int, ...]


def _te_grouped_mlp_spec(module=TEGroupedMLP):
    return ModuleSpec(
        module=module,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
    )


def _sequential_bf16_routed_experts(
    hidden_states,
    topk_weights,
    w13_gate_up,
    w2,
    tokens_per_expert,
    *,
    activation_in_fp32=False,
    fc2_input_qdq=None,
):
    """Independent per-expert oracle for the grouped production surrogate."""

    outputs = []
    start = 0
    for local_expert, count in enumerate(tokens_per_expert):
        if count == 0:
            continue
        end = start + count
        fc1 = F.linear(hidden_states[start:end], w13_gate_up[local_expert])
        if fc2_input_qdq is not None:
            activation_dtype = (
                torch.float32 if activation_in_fp32 else fc2_input_qdq.fc2_input_qdq_source_dtype
            )
            gate, up = fc1.to(activation_dtype).chunk(2, dim=-1)
            activated = fc2_input_qdq.qdq_fc2_input(
                (F.silu(gate) * up).to(fc2_input_qdq.fc2_input_qdq_source_dtype)
            )
        elif activation_in_fp32:
            gate, up = fc1.float().chunk(2, dim=-1)
            activated = (F.silu(gate) * up).to(hidden_states.dtype)
        else:
            gate, up = fc1.chunk(2, dim=-1)
            activated = (F.silu(gate) * up).to(hidden_states.dtype)
        output = F.linear(activated, w2[local_expert])
        output = output * topk_weights[start:end].to(output.dtype)
        outputs.append(output)
        start = end
    if not outputs:
        return torch.empty_like(hidden_states)
    return torch.cat(outputs, dim=0)


def _decode_mxfp8_payload(data, scales):
    rows, columns = data.shape
    scale_bytes = scales.view(torch.uint8).reshape(rows, columns // 32)
    expanded_scales = scale_bytes.repeat_interleave(32, dim=-1)
    return torch.where(
        expanded_scales == 0,
        torch.zeros_like(data, dtype=torch.float32),
        torch.ldexp(data.float(), expanded_scales.to(torch.int32) - 127),
    ).to(torch.bfloat16)


def _decode_nvfp4_payload(data, scales, per_token_scale):
    rows, packed_columns = data.shape
    columns = packed_columns * 2
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data.device)
    packed = data.view(torch.uint8).flatten()
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=1).flatten()
    values = e2m1[(nibbles & 0x7).long()] * torch.where(nibbles & 0x8 != 0, -1.0, 1.0)
    values = values.reshape(rows, columns // 16, 16)
    block_scales = scales.view(torch.float8_e4m3fn).float().reshape(rows, columns // 16, 1)
    return (
        (values * block_scales * per_token_scale.reshape(rows, 1, 1))
        .reshape(rows, columns)
        .to(torch.bfloat16)
    )


def test_flashinfer_moe_selects_extension_experts_without_mutating_source(monkeypatch):
    class OtherExperts:
        pass

    shared_experts = ModuleSpec(module=SharedExpertMLP)
    original = SimpleNamespace(
        experts=_te_grouped_mlp_spec(OtherExperts), shared_experts=shared_experts
    )
    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")

    replacement = maybe_replace_flashinfer_moe_expert_spec(original)

    assert replacement is not original
    assert replacement.experts.module is FlashInferGroupedMLP
    assert replacement.experts.submodules is original.experts.submodules
    assert replacement.shared_experts is shared_experts
    assert original.experts.module is OtherExperts


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped BF16 requires CUDA")
@pytest.mark.parametrize(
    "activation_in_fp32,fused_activation,fc2_input_qdq_source_dtype",
    [
        pytest.param(False, False, None, id="unfused-bf16"),
        pytest.param(False, True, None, id="fused-bf16"),
        pytest.param(True, False, None, id="activation-fp32"),
        pytest.param(True, False, torch.bfloat16, id="activation-fp32-fc2-input-qdq"),
        pytest.param(False, True, torch.bfloat16, id="fused-config-fc2-input-qdq"),
    ],
)
def test_te_grouped_bf16_surrogate_matches_independent_loop(
    activation_in_fp32, fused_activation, fc2_input_qdq_source_dtype
):
    tokens_per_expert = (0, 2, 0, 3, 0)
    num_experts = len(tokens_per_expert)
    hidden_size = 128
    intermediate_size = 128
    num_tokens = sum(tokens_per_expert)
    torch.manual_seed(1234)

    hidden = torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    probs = torch.rand((num_tokens, 1), device="cuda", dtype=torch.float32, requires_grad=True)
    w13 = tuple(
        torch.randn((2 * intermediate_size, hidden_size), device="cuda", dtype=torch.bfloat16)
        .mul_(0.02)
        .requires_grad_()
        for _ in range(num_experts)
    )
    w2 = tuple(
        torch.randn((hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16)
        .mul_(0.02)
        .requires_grad_()
        for _ in range(num_experts)
    )
    grad_output = torch.randn_like(hidden)

    def make_fc2_input_qdq(seen):
        if fc2_input_qdq_source_dtype is None:
            return None

        def qdq(source):
            seen.append(source.detach())
            source_bf16 = source.to(torch.bfloat16)
            decoded = (source.float() * 8).round().div(8).to(torch.bfloat16)
            return source_bf16 + (decoded - source_bf16).detach()

        return SimpleNamespace(
            fc2_input_qdq_source_dtype=fc2_input_qdq_source_dtype, qdq_fc2_input=qdq
        )

    actual_qdq_inputs = []
    expected_qdq_inputs = []
    actual_fc2_input_qdq = make_fc2_input_qdq(actual_qdq_inputs)
    expected_fc2_input_qdq = make_fc2_input_qdq(expected_qdq_inputs)

    surrogate = _BF16GroupedMLPSurrogate(
        num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    actual = surrogate(
        hidden,
        probs,
        w13,
        w2,
        tokens_per_expert,
        activation_in_fp32=activation_in_fp32,
        fused_activation=fused_activation,
        fc2_input_qdq=actual_fc2_input_qdq,
    )
    actual_inputs = (hidden, probs, *w13, *w2)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output, allow_unused=True)
    actual_grads = tuple(
        torch.zeros_like(tensor) if grad is None else grad
        for tensor, grad in zip(actual_inputs, actual_grads)
    )

    hidden_ref = hidden.detach().requires_grad_()
    probs_ref = probs.detach().requires_grad_()
    w13_ref = tuple(weight.detach().requires_grad_() for weight in w13)
    w2_ref = tuple(weight.detach().requires_grad_() for weight in w2)
    expected = _sequential_bf16_routed_experts(
        hidden_ref,
        probs_ref,
        w13_ref,
        w2_ref,
        tokens_per_expert,
        activation_in_fp32=activation_in_fp32,
        fc2_input_qdq=expected_fc2_input_qdq,
    )
    expected_inputs = (hidden_ref, probs_ref, *w13_ref, *w2_ref)
    expected_grads = torch.autograd.grad(expected, expected_inputs, grad_output, allow_unused=True)
    expected_grads = tuple(
        torch.zeros_like(tensor) if grad is None else grad
        for tensor, grad in zip(expected_inputs, expected_grads)
    )

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-3, atol=5e-3)
    if fc2_input_qdq_source_dtype is not None:
        assert len(actual_qdq_inputs) == 1
        assert len(expected_qdq_inputs) == 2
        assert actual_qdq_inputs[0].shape == (num_tokens, intermediate_size)
        assert actual_qdq_inputs[0].dtype == fc2_input_qdq_source_dtype


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer forward-operand dequantization requires Blackwell",
)
@pytest.mark.parametrize(
    "quantization,use_4over6,use_256,hidden_size",
    [
        pytest.param("mxfp8", False, False, 128, id="mxfp8"),
        pytest.param("nvfp4", False, False, 16, id="nvfp4-k16"),
        pytest.param("nvfp4", True, False, 128, id="nvfp4-4over6-e4m3-448"),
        pytest.param("nvfp4", True, True, 128, id="nvfp4-4over6-e4m3-256"),
    ],
)
def test_flashinfer_activation_payload_qdq(
    monkeypatch, quantization, use_4over6, use_256, hidden_size
):
    torch.manual_seed(123)
    hidden = torch.randn((33, hidden_size), device="cuda", dtype=torch.bfloat16)

    if quantization == "mxfp8":
        from flashinfer import mxfp8_quantize

        data, scales = mxfp8_quantize(hidden, False, backend="cute-dsl")
        actual = _FlashInferMXFP8Runner.dequantize_activation(data, scales, dtype=torch.bfloat16)
        expected = _decode_mxfp8_payload(data, scales)
    else:
        from flashinfer import SfLayout, nvfp4_quantize

        if use_4over6:
            _set_nvfp4_4over6_env(monkeypatch, flashinfer=True)
            if not use_256:
                monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "0")
        else:
            monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
            monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "0")
        e4m3_max = 256 if use_4over6 and use_256 else 448
        input_global_scale = torch.tensor(
            [1.0 / (e4m3_max * 6.0)], device="cuda", dtype=torch.float32
        )
        data, scales, per_token_scale = nvfp4_quantize(
            hidden,
            input_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cuda",
        )
        actual = _FlashInferNVFP4Runner.dequantize_activation(
            data,
            scales,
            per_token_scale,
            dtype=torch.bfloat16,
            e4m3_max=e4m3_max,
            use_4over6=use_4over6,
        )
        expected = _decode_nvfp4_payload(data, scales, per_token_scale)

    # TE's 4-over-6 decoder may choose a different BF16 multiply order than
    # this explicit payload formula; the observed discrepancy is at most one
    # BF16 rounding step. Standard MXFP8 and NVFP4 remain bitwise checks.
    rtol = 0.005 if quantization == "nvfp4" and use_4over6 else 0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)

    runner = _FlashInferMXFP8Runner if quantization == "mxfp8" else _FlashInferNVFP4Runner
    source_dtype = runner.fc2_input_qdq_source_dtype
    assert source_dtype is torch.bfloat16
    source = hidden.detach().requires_grad_(True)
    actual_qdq = runner.qdq_fc2_input(source)

    assert actual_qdq.dtype == torch.bfloat16
    torch.testing.assert_close(actual_qdq, expected, rtol=rtol, atol=0)
    grad_output = torch.randn_like(actual_qdq)
    torch.autograd.backward(actual_qdq, grad_output)
    torch.testing.assert_close(source.grad, grad_output.to(source_dtype), rtol=0, atol=0)


def _set_nvfp4_4over6_env(monkeypatch, *, flashinfer=False):
    settings = {
        "NVTE_NVFP4_4OVER6": "all",
        "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
        "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    }
    if flashinfer:
        settings.update(
            {
                "FLASHINFER_NVFP4_4OVER6": "1",
                "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
                "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
                "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
                "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
            }
        )
    for name, value in settings.items():
        monkeypatch.setenv(name, value)


def _distributed_routing_ids(
    rank: int, world_size: int, num_tokens: int, num_experts: int, top_k: int
) -> torch.Tensor:
    """Route to unique experts on every EP rank except the final rank."""

    if num_experts % world_size:
        raise ValueError(f"num_experts={num_experts} must be divisible by world_size={world_size}")
    local_experts = num_experts // world_size
    active_experts = num_experts - local_experts
    if top_k > active_experts:
        raise ValueError(f"top_k={top_k} exceeds active experts={active_experts}")
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    slot_ids = torch.arange(top_k, device="cuda", dtype=torch.long)
    bases = (rank * num_tokens + token_ids) * top_k
    return ((bases.unsqueeze(1) + slot_ids) % active_experts).contiguous()


def _balanced_distributed_routing_ids(
    rank: int, world_size: int, num_tokens: int, num_experts: int, top_k: int
) -> torch.Tensor:
    """Route benchmark assignments evenly across every global expert."""

    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    total_assignments = world_size * num_tokens * top_k
    if total_assignments % num_experts:
        raise ValueError(
            f"{total_assignments} benchmark assignments cannot balance over {num_experts} experts"
        )
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    slot_ids = torch.arange(top_k, device="cuda", dtype=torch.long)
    bases = (rank * num_tokens + token_ids) * top_k
    return ((bases.unsqueeze(1) + slot_ids) % num_experts).contiguous()


def _make_routing_replay(route_ids: torch.Tensor, num_experts: int) -> _RoutingReplay:
    """Materialize expert selections once for replay across matched layers."""

    routing_map = torch.zeros(
        (route_ids.shape[0], num_experts), device=route_ids.device, dtype=torch.bool
    )
    routing_map.scatter_(1, route_ids, True)
    return _RoutingReplay(topk_ids=route_ids, routing_map=routing_map)


def _install_replayed_route(
    layer: MoELayer, route_logits: torch.Tensor, replay: _RoutingReplay
) -> None:
    """Replay fixed selections while preserving meaningful router gradients."""

    def route(_self, hidden_states, padding_mask=None, input_ids=None):
        del padding_mask, input_ids
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if num_tokens != replay.topk_ids.shape[0]:
            raise RuntimeError(f"route token mismatch: {num_tokens} != {replay.topk_ids.shape[0]}")
        background = torch.zeros(
            (num_tokens, 1), device=route_logits.device, dtype=route_logits.dtype
        )
        selected_probs = torch.softmax(torch.cat((route_logits, background), dim=-1), dim=-1)[
            :, :-1
        ]
        probs = torch.zeros(
            (num_tokens, _self.config.num_moe_experts),
            device=hidden_states.device,
            dtype=selected_probs.dtype,
        ).scatter(1, replay.topk_ids, selected_probs)
        return probs, replay.routing_map.clone()

    layer.route = types.MethodType(route, layer)


def _distributed_layer_precision_context(layer: MoELayer, layer_no: int, backward_mode: str):
    config = layer.config
    is_bf16_boundary = config.first_last_layers_bf16 and (
        layer_no < config.num_layers_at_start_in_bf16
        or layer_no >= config.num_layers - config.num_layers_at_end_in_bf16
    )
    if is_bf16_boundary:
        return nullcontext()
    if config.fp8 is not None:
        recipe = replace(get_fp8_recipe(config), backward_override=backward_mode)
        with mock.patch("megatron.core.fp8_utils.get_fp8_recipe", return_value=recipe):
            return get_fp8_context(config, layer_no)
    if config.fp4 is not None:
        recipe = replace(
            get_fp4_recipe(config),
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
            row_scaled_activation=True,
            nvfp4_4over6="all",
            nvfp4_4over6_e4m3_use_256="all",
            nvfp4_4over6_err_mode="MSE",
            backward_override=backward_mode,
        )
        with mock.patch("megatron.core.fp4_utils.get_fp4_recipe", return_value=recipe):
            return get_fp4_context(config, layer_no)
    return nullcontext()


def _build_distributed_moe_layer(
    monkeypatch,
    config: TransformerConfig,
    *,
    layer_no: int,
    backward_mode: str,
    use_flashinfer: bool,
) -> MoELayer:
    """Build either the native Megatron/TE reference or FlashInfer replacement."""

    monkeypatch.setenv("NVTE_BACKWARD_OVERRIDE", backward_mode)
    if use_flashinfer:
        monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")
    else:
        monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "0")
    layer = MoELayer(
        config, MoESubmodules(experts=_te_grouped_mlp_spec()), layer_number=layer_no + 1
    ).cuda()
    layer.train()
    expected_experts_type = FlashInferGroupedMLP if use_flashinfer else TEGroupedMLP
    assert type(layer.experts) is expected_experts_type
    if use_flashinfer:
        assert layer.experts._flashinfer_moe_backward_mode == backward_mode
    return layer


def _run_distributed_layer_once(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    routing_replay: _RoutingReplay,
    *,
    layer_no: int,
    backward_mode: str,
    track_received_rows: bool = False,
) -> _LayerRun:
    """Run one full layer forward/backward and snapshot its local results."""

    layer.zero_grad(set_to_none=True)
    hidden = hidden_seed.detach().clone().requires_grad_()
    route_logits = logits_seed.detach().clone().requires_grad_()
    _install_replayed_route(layer, route_logits, routing_replay)
    os.environ["NVTE_BACKWARD_OVERRIDE"] = backward_mode

    received_rows = []
    hook = None
    if track_received_rows:
        hook = layer.experts.register_forward_pre_hook(
            lambda _module, inputs: received_rows.append(inputs[0].shape[0])
        )

    try:
        with _distributed_layer_precision_context(layer, layer_no, backward_mode):
            output, _ = layer(hidden)
    finally:
        if hook is not None:
            hook.remove()

    torch.autograd.backward(output, grad_seed)
    runner = getattr(layer.experts, "_flashinfer_moe_runner", None)
    if runner is not None:
        assert runner._prepared is None
        assert runner._weight_key is None
    parameter_grads = []
    missing_grads = int(hidden.grad is None) + int(route_logits.grad is None)
    hidden_grad = torch.zeros_like(hidden) if hidden.grad is None else hidden.grad
    route_grad = torch.zeros_like(route_logits) if route_logits.grad is None else route_logits.grad
    for parameter in layer.experts.parameters():
        if parameter.grad is None:
            missing_grads += 1
            parameter_grads.append(torch.zeros_like(parameter))
        else:
            parameter_grads.append(parameter.grad.detach())

    return _LayerRun(
        output=output.detach(),
        hidden_grad=hidden_grad.detach(),
        route_grad=route_grad.detach(),
        parameter_grads=tuple(parameter_grads),
        missing_grads=missing_grads,
        received_rows=tuple(received_rows),
    )


def _run_distributed_performance_step(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    routing_replay: _RoutingReplay,
    *,
    layer_no: int,
    group,
    nvtx_profile_label: str | None = None,
) -> float:
    """Run one synchronized step; input setup and the barrier are not timed."""

    layer.zero_grad(set_to_none=True)
    hidden = hidden_seed.detach().clone().requires_grad_()
    route_logits = logits_seed.detach().clone().requires_grad_()
    _install_replayed_route(layer, route_logits, routing_replay)
    os.environ["NVTE_BACKWARD_OVERRIDE"] = HIGH_PRECISION_BACKWARD

    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    if nvtx_profile_label is not None:
        core_utils.nvtx_range_push(msg="flashinfer_moe_perf", suffix=nvtx_profile_label)
    try:
        with _distributed_layer_precision_context(layer, layer_no, HIGH_PRECISION_BACKWARD):
            output, _ = layer(hidden)
        torch.autograd.backward(output, grad_seed)
        torch.cuda.synchronize()
    finally:
        if nvtx_profile_label is not None:
            core_utils.nvtx_range_pop(msg="flashinfer_moe_perf", suffix=nvtx_profile_label)
    step_wall_ms = (time.perf_counter() - wall_start) * 1e3

    del hidden, route_logits, output
    layer.zero_grad(set_to_none=True)
    return step_wall_ms


def _distributed_critical_time(local_step_ms: float, group) -> tuple[float, float]:
    local = torch.tensor([local_step_ms], device="cuda", dtype=torch.float64)
    maximum = local.clone()
    minimum = local.clone()
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=group)
    return maximum.item(), maximum.item() - minimum.item()


def _profile_distributed_dispatchers(
    layers,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    routing_replay: _RoutingReplay,
    *,
    layer_no: int,
    configured_precision: str,
    execution_precision: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    iterations: int = 6,
) -> None:
    """Print paired timings, then replay production NVTX ranges once per dispatcher."""

    dispatchers = ("allgather", "alltoall")
    group = layers["allgather"].token_dispatcher.ep_group
    nvtx_stack_depth = len(core_utils._nvtx_range_messages)
    critical_samples = {dispatcher: [] for dispatcher in dispatchers}
    rank_skew_samples = {dispatcher: [] for dispatcher in dispatchers}
    with mock.patch.object(core_utils, "_nvtx_enabled", False):
        for dispatcher in dispatchers:
            _run_distributed_performance_step(
                layers[dispatcher],
                hidden_seed,
                logits_seed,
                grad_seed,
                routing_replay,
                layer_no=layer_no,
                group=group,
            )
            runner = layers[dispatcher].experts._flashinfer_moe_runner
            assert runner is not None and runner.quantization == execution_precision
        for iteration in range(iterations):
            execution_order = dispatchers if iteration % 2 == 0 else tuple(reversed(dispatchers))
            for dispatcher in execution_order:
                local_step_ms = _run_distributed_performance_step(
                    layers[dispatcher],
                    hidden_seed,
                    logits_seed,
                    grad_seed,
                    routing_replay,
                    layer_no=layer_no,
                    group=group,
                )
                critical_ms, rank_skew_ms = _distributed_critical_time(local_step_ms, group)
                critical_samples[dispatcher].append(critical_ms)
                rank_skew_samples[dispatcher].append(rank_skew_ms)
    assert len(core_utils._nvtx_range_messages) == nvtx_stack_depth

    with mock.patch.object(core_utils, "_nvtx_enabled", True):
        for dispatcher in dispatchers:
            _run_distributed_performance_step(
                layers[dispatcher],
                hidden_seed,
                logits_seed,
                grad_seed,
                routing_replay,
                layer_no=layer_no,
                group=group,
                nvtx_profile_label=f"{dispatcher}_profile_step",
            )
    assert len(core_utils._nvtx_range_messages) == nvtx_stack_depth

    global_tokens = group.size() * hidden_seed.shape[0]
    results = {}
    for dispatcher in dispatchers:
        step_median_ms = statistics.median(critical_samples[dispatcher])
        results[dispatcher] = (step_median_ms, statistics.median(rank_skew_samples[dispatcher]))
        if torch.distributed.get_rank() != 0:
            continue
        samples = "/".join(f"{sample:.3f}" for sample in critical_samples[dispatcher])
        print(
            "FlashInfer distributed perf (non-gating): "
            f"configured={configured_precision}, execution={execution_precision}, "
            f"dispatcher={dispatcher}, backward_mode={HIGH_PRECISION_BACKWARD}, "
            "routing=balanced, measurement=uninstrumented, order=alternating, "
            f"tokens_per_rank={hidden_seed.shape[0]}, global_tokens={global_tokens}, "
            f"num_experts={num_experts}, hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}, top_k={top_k}, "
            f"iterations={iterations}, step_samples_ms={samples}, "
            f"step_median_ms={step_median_ms:.3f}, "
            f"rank_skew_median_ms={results[dispatcher][1]:.3f}, "
            f"global_tokens_per_s={global_tokens / (step_median_ms * 1e-3):.1f}, "
            "routed_assignments_per_s="
            f"{global_tokens * top_k / (step_median_ms * 1e-3):.1f}, "
            "nvtx_profile_replays=1",
            flush=True,
        )

    if torch.distributed.get_rank() != 0:
        return
    allgather_ms = results["allgather"][0]
    alltoall_ms = results["alltoall"][0]
    print(
        "FlashInfer distributed perf comparison (non-gating): "
        f"configured={configured_precision}, execution={execution_precision}, "
        "routing=balanced, measurement=uninstrumented, order=alternating, "
        f"tokens_per_rank={hidden_seed.shape[0]}, global_tokens={global_tokens}, "
        f"top_k={top_k}, allgather_step_ms={allgather_ms:.3f}, "
        f"alltoall_step_ms={alltoall_ms:.3f}, "
        f"alltoall_speedup={allgather_ms / alltoall_ms:.3f}, "
        "speedup_definition=allgather_step_ms/alltoall_step_ms",
        flush=True,
    )


def _global_max(value: torch.Tensor) -> float:
    value = value.detach().float()
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value.item()


def _global_relative_l2(actual_tensors, reference_tensors) -> float:
    actual_tensors = tuple(actual_tensors)
    reference_tensors = tuple(reference_tensors)
    if len(actual_tensors) != len(reference_tensors):
        raise ValueError(
            "distributed numerical comparison requires matching tensor lists, "
            f"got {len(actual_tensors)} and {len(reference_tensors)}"
        )
    error_sq = torch.zeros((), device="cuda")
    reference_sq = torch.zeros((), device="cuda")
    for actual, reference in zip(actual_tensors, reference_tensors):
        error_sq += (actual.float() - reference.float()).square().sum()
        reference_sq += reference.float().square().sum()
    torch.distributed.all_reduce(error_sq)
    torch.distributed.all_reduce(reference_sq)
    return torch.sqrt(error_sq / reference_sq.clamp_min(1e-20)).item()


def _distributed_numerical_metrics(actual: _LayerRun, reference: _LayerRun) -> _NumericalMetrics:
    per_token_forward_rel_l2 = _global_max(
        torch.sqrt(
            (actual.output.float() - reference.output.float()).square().sum(dim=-1)
            / reference.output.float().square().sum(dim=-1).clamp_min(1e-20)
        ).max()
    )
    missing_grads = torch.tensor(
        actual.missing_grads + reference.missing_grads, device="cuda", dtype=torch.int32
    )
    torch.distributed.all_reduce(missing_grads)
    return _NumericalMetrics(
        forward_rel_l2=_global_relative_l2((actual.output,), (reference.output,)),
        per_token_forward_rel_l2=per_token_forward_rel_l2,
        hidden_grad_rel_l2=_global_relative_l2((actual.hidden_grad,), (reference.hidden_grad,)),
        route_grad_rel_l2=_global_relative_l2((actual.route_grad,), (reference.route_grad,)),
        parameter_grad_rel_l2=_global_relative_l2(
            actual.parameter_grads, reference.parameter_grads
        ),
        missing_grads=missing_grads.item(),
    )


def _run_distributed_matched_pair(
    monkeypatch,
    make_config,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    routing_replay: _RoutingReplay,
    *,
    layer_no: int,
    backward_mode: str,
    execution_precision: str,
    track_received_rows: bool,
) -> _MatchedPair:
    """Compare one native/FlashInfer pair, then release its layer-sized gradients."""

    layers = []
    try:
        reference_layer = _build_distributed_moe_layer(
            monkeypatch,
            make_config(),
            layer_no=layer_no,
            backward_mode=backward_mode,
            use_flashinfer=False,
        )
        layers.append(reference_layer)
        flashinfer_layer = _build_distributed_moe_layer(
            monkeypatch,
            make_config(),
            layer_no=layer_no,
            backward_mode=backward_mode,
            use_flashinfer=True,
        )
        layers.append(flashinfer_layer)
        flashinfer_layer.load_state_dict(reference_layer.state_dict())

        monkeypatch.setenv("MEGATRON_MOE_APPLY_PROBS_ON_OUTPUT", "1")
        reference = _run_distributed_layer_once(
            reference_layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            routing_replay,
            layer_no=layer_no,
            backward_mode=backward_mode,
        )
        monkeypatch.setenv("MEGATRON_MOE_APPLY_PROBS_ON_OUTPUT", "0")
        actual = _run_distributed_layer_once(
            flashinfer_layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            routing_replay,
            layer_no=layer_no,
            backward_mode=backward_mode,
            track_received_rows=track_received_rows,
        )
        runner = flashinfer_layer.experts._flashinfer_moe_runner
        assert runner is not None and runner.quantization == execution_precision
        return _MatchedPair(
            output=actual.output,
            metrics=_distributed_numerical_metrics(actual, reference),
            received_rows=actual.received_rows,
        )
    finally:
        for layer in layers:
            layer.zero_grad(set_to_none=True)
            runner = getattr(layer.experts, "_flashinfer_moe_runner", None)
            if runner is not None:
                runner.invalidate_weights()


_NUMERICAL_TOLERANCES = {
    # (global forward, per-token forward), then
    # (hidden, router, parameter) gradients for each backward mode.
    # High-precision replay is a coarse semantic guard because it intentionally
    # uses BF16 operands. Dequantized replay is the tighter algebraic comparison.
    # Measured 8-GPU B200 dequantized maxima (hidden, router, parameter):
    # BF16 (0.0039, 0.0019, 0.0003), MXFP8 (0.0038, 0.0002, 0.0001),
    # NVFP4 (0.0062, 0.0387, 0.0258).
    "bf16": ((0.010, 0.012), (0.005, 0.003, 0.001), (0.005, 0.003, 0.001)),
    "mxfp8": ((0.050, 0.075), (0.055, 0.070, 0.060), (0.005, 0.001, 0.001)),
    "nvfp4": ((0.050, 0.085), (0.170, 0.220, 0.190), (0.008, 0.050, 0.035)),
}


def _assert_numerical_metrics(
    metrics: _NumericalMetrics, execution_precision: str, backward_mode: str
) -> None:
    forward_tolerances, high_precision_tolerances, dequantized_tolerances = _NUMERICAL_TOLERANCES[
        execution_precision
    ]
    gradient_tolerances = (
        high_precision_tolerances
        if backward_mode == HIGH_PRECISION_BACKWARD
        else dequantized_tolerances
    )
    assert metrics.missing_grads == 0
    assert metrics.forward_rel_l2 < forward_tolerances[0]
    assert metrics.per_token_forward_rel_l2 < forward_tolerances[1]
    for value, tolerance in zip(
        (metrics.hidden_grad_rel_l2, metrics.route_grad_rel_l2, metrics.parameter_grad_rel_l2),
        gradient_tolerances,
    ):
        assert value < tolerance


def _report_numerical_metrics(
    metrics: _NumericalMetrics,
    *,
    precision_case: _PrecisionCase,
    dispatcher: str,
    backward_mode: str,
    num_tokens: int,
    top_k: int,
) -> None:
    if torch.distributed.get_rank() != 0:
        return
    print(
        "FlashInfer distributed numerical check: "
        f"configured={precision_case.configured}, execution={precision_case.execution}, "
        f"dispatcher={dispatcher}, backward_mode={backward_mode}, "
        f"num_tokens={num_tokens}, top_k={top_k}, "
        f"forward_rel_l2={metrics.forward_rel_l2:.6f}, "
        f"per_token_forward_rel_l2={metrics.per_token_forward_rel_l2:.6f}, "
        f"hidden_grad_rel_l2={metrics.hidden_grad_rel_l2:.6f}, "
        f"route_grad_rel_l2={metrics.route_grad_rel_l2:.6f}, "
        f"parameter_grad_rel_l2={metrics.parameter_grad_rel_l2:.6f}",
        flush=True,
    )


def _assert_alltoall_execution(
    result: _MatchedPair, *, world_size: int, num_tokens: int, top_k: int
) -> None:
    received = torch.tensor(
        result.received_rows if len(result.received_rows) == 1 else (-1,),
        device="cuda",
        dtype=torch.int64,
    )
    received_tensors = [torch.empty_like(received) for _ in range(world_size)]
    torch.distributed.all_gather(received_tensors, received)
    received_by_rank = [value.item() for value in received_tensors]

    output_nonzero = torch.tensor(
        [int(torch.count_nonzero(result.output).item() > 0)], device="cuda", dtype=torch.int32
    )
    output_nonzero_tensors = [torch.empty_like(output_nonzero) for _ in range(world_size)]
    torch.distributed.all_gather(output_nonzero_tensors, output_nonzero)
    output_nonzero_by_rank = [value.item() for value in output_nonzero_tensors]

    assert all(received > 0 for received in received_by_rank[:-1])
    assert received_by_rank[-1] == 0
    assert sum(received_by_rank) == world_size * num_tokens * top_k
    assert output_nonzero_by_rank[-1] == 1


@pytest.mark.internal
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires torchrun with exactly 8 ranks"
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer routed MoE requires eight Blackwell GPUs",
)
@pytest.mark.parametrize(
    "precision_case",
    [
        pytest.param(_PrecisionCase("bf16", "bf16"), id="bf16"),
        pytest.param(
            _PrecisionCase("mxfp8", "bf16", num_layers=3, layer_no=2, first_last_layers_bf16=True),
            id="mxfp8-last-layer-bf16",
        ),
        pytest.param(
            _PrecisionCase("nvfp4", "bf16", num_layers=3, layer_no=0, first_last_layers_bf16=True),
            id="nvfp4-first-layer-bf16",
        ),
        pytest.param(_PrecisionCase("mxfp8", "mxfp8"), id="mxfp8"),
        pytest.param(_PrecisionCase("nvfp4", "nvfp4"), id="nvfp4"),
    ],
)
@pytest.mark.parametrize(
    "moe_token_dispatcher_type",
    [pytest.param("allgather", id="allgather"), pytest.param("alltoall", id="alltoall")],
)
@pytest.mark.parametrize(
    "model_hyperparameters",
    [
        pytest.param(
            _ModelHyperparameters(
                num_experts=32, hidden_size=7168, intermediate_size=2048, top_k=8
            ),
            id="e32-h7168-i2048-topk8",
        )
    ],
)
@pytest.mark.parametrize("num_tokens", [8, 4096])
def test_flashinfer_routed_forward_and_surrogate_backward(
    monkeypatch, precision_case, moe_token_dispatcher_type, model_hyperparameters, num_tokens
):
    """Compare FlashInfer with the same-precision native Megatron/TE MoE."""

    num_experts = model_hyperparameters.num_experts
    hidden_size = model_hyperparameters.hidden_size
    intermediate_size = model_hyperparameters.intermediate_size
    top_k = model_hyperparameters.top_k
    world_size = int(os.environ["WORLD_SIZE"])
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
    )
    rank = torch.distributed.get_rank()
    run_completed = False

    try:
        if precision_case.configured == "nvfp4":
            _set_nvfp4_4over6_env(monkeypatch, flashinfer=True)
        if precision_case.configured == "bf16":
            precision_config = {}
        elif precision_case.configured == "mxfp8":
            precision_config = {"fp8": "e4m3", "fp8_recipe": "mxfp8"}
        elif precision_case.configured == "nvfp4":
            precision_config = {"fp4": "e2m1", "fp4_recipe": "nvfp4"}
        else:
            raise NotImplementedError(
                f"test has no precision-config branch for {precision_case.configured!r}"
            )

        def make_config(dispatcher_type=moe_token_dispatcher_type):
            torch.manual_seed(1234)
            model_parallel_cuda_manual_seed(1234)
            return TransformerConfig(
                num_layers=precision_case.num_layers,
                hidden_size=hidden_size,
                num_attention_heads=16,
                num_moe_experts=num_experts,
                moe_ffn_hidden_size=intermediate_size,
                moe_router_topk=top_k,
                moe_router_pre_softmax=True,
                moe_router_load_balancing_type="none",
                moe_token_dispatcher_type=dispatcher_type,
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
                gradient_accumulation_fusion=False,
                use_cpu_initialization=False,
                first_last_layers_bf16=precision_case.first_last_layers_bf16,
                num_layers_at_start_in_bf16=(1 if precision_case.first_last_layers_bf16 else 0),
                num_layers_at_end_in_bf16=(1 if precision_case.first_last_layers_bf16 else 0),
                **precision_config,
            )

        benchmark_case = (
            num_tokens == 4096
            and precision_case.execution in ("mxfp8", "nvfp4")
            and moe_token_dispatcher_type == "allgather"
        )
        torch.manual_seed(5678 + rank)
        hidden_seed = torch.randn((num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
        logits_seed = torch.randn((num_tokens, top_k), device="cuda", dtype=torch.float32)
        grad_seed = torch.randn_like(hidden_seed)
        routing_replay = _make_routing_replay(
            _distributed_routing_ids(rank, world_size, num_tokens, num_experts, top_k), num_experts
        )

        if precision_case.execution != "bf16":
            backward_modes = (HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD)
        elif num_tokens == 8:
            backward_modes = (HIGH_PRECISION_BACKWARD,)
        else:
            # BF16 has no quantized operands, so a requested dequantized mode
            # deliberately resolves to the same high-precision replay.
            backward_modes = (DEQUANTIZED_BACKWARD,)
        mode_outputs = {}
        for backward_mode in backward_modes:
            pair = _run_distributed_matched_pair(
                monkeypatch,
                make_config,
                hidden_seed,
                logits_seed,
                grad_seed,
                routing_replay,
                layer_no=precision_case.layer_no,
                backward_mode=backward_mode,
                execution_precision=precision_case.execution,
                track_received_rows=(
                    moe_token_dispatcher_type == "alltoall"
                    and backward_mode == DEQUANTIZED_BACKWARD
                ),
            )
            _assert_numerical_metrics(pair.metrics, precision_case.execution, backward_mode)
            _report_numerical_metrics(
                pair.metrics,
                precision_case=precision_case,
                dispatcher=moe_token_dispatcher_type,
                backward_mode=backward_mode,
                num_tokens=num_tokens,
                top_k=top_k,
            )
            mode_outputs[backward_mode] = pair.output
            if moe_token_dispatcher_type == "alltoall" and backward_mode == DEQUANTIZED_BACKWARD:
                _assert_alltoall_execution(
                    pair, world_size=world_size, num_tokens=num_tokens, top_k=top_k
                )

        if len(mode_outputs) == 2:
            backward_mode_forward_rel_l2 = _global_relative_l2(
                (mode_outputs[DEQUANTIZED_BACKWARD],), (mode_outputs[HIGH_PRECISION_BACKWARD],)
            )
            if rank == 0:
                print(
                    "FlashInfer distributed backward-mode forward check: "
                    f"configured={precision_case.configured}, "
                    f"execution={precision_case.execution}, "
                    f"dispatcher={moe_token_dispatcher_type}, num_tokens={num_tokens}, "
                    f"backward_mode_forward_rel_l2={backward_mode_forward_rel_l2:.6f}",
                    flush=True,
                )
            # Separate fused-finalize launches can choose a different top-k reduction
            # order (0.0047 max observed here), but backward policy cannot change forward.
            assert backward_mode_forward_rel_l2 < 0.006

        if benchmark_case:
            performance_layers = {}
            try:
                monkeypatch.setenv("MEGATRON_MOE_APPLY_PROBS_ON_OUTPUT", "0")
                for dispatcher in ("allgather", "alltoall"):
                    layer = _build_distributed_moe_layer(
                        monkeypatch,
                        make_config(dispatcher),
                        layer_no=precision_case.layer_no,
                        backward_mode=HIGH_PRECISION_BACKWARD,
                        use_flashinfer=True,
                    )
                    if performance_layers:
                        layer.load_state_dict(performance_layers["allgather"].state_dict())
                    performance_layers[dispatcher] = layer
                balanced_routing_replay = _make_routing_replay(
                    _balanced_distributed_routing_ids(
                        rank, world_size, num_tokens, num_experts, top_k
                    ),
                    num_experts,
                )
                _profile_distributed_dispatchers(
                    performance_layers,
                    hidden_seed,
                    logits_seed,
                    grad_seed,
                    balanced_routing_replay,
                    layer_no=precision_case.layer_no,
                    configured_precision=precision_case.configured,
                    execution_precision=precision_case.execution,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    top_k=top_k,
                )
            finally:
                for layer in performance_layers.values():
                    layer.zero_grad(set_to_none=True)
                    runner = layer.experts._flashinfer_moe_runner
                    if runner is not None:
                        runner.invalidate_weights()

        torch.distributed.barrier()
        torch.cuda.synchronize()
        run_completed = True
    finally:
        if run_completed:
            Utils.destroy_model_parallel()
        else:
            parallel_state.destroy_model_parallel()
            Utils.inited = False
            # Let torchrun terminate peer ranks after the original exception;
            # synchronously tearing down the world here can hide the failure
            # behind an NCCL shutdown wait while another rank is still on GPU.
