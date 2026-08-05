# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import types
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from miles_megatron_plugins import flashinfer_moe as flashinfer_moe_module
from miles_megatron_plugins.flashinfer_moe import (
    _bf16_local_routed_experts,
    _dispatched_topk_inputs,
    _flashinfer_moe_description,
    _flashinfer_moe_quantization,
    _flashinfer_moe_runner_type,
    _FlashInferForwardBF16Backward,
    _FlashInferMXFP8Runner,
    _FlashInferNVFP4Runner,
    FlashInferGroupedMLP,
    _grouped_mlp_weights,
    _pack_topk_ids,
    _rollout_replay_topk_ids,
    _te_mxfp8_quantize_gated_weight,
    _te_mxfp8_quantize_weight,
    _te_nvfp4_quantize_gated_weight,
    _te_nvfp4_quantize_weight,
    _topk_from_dense_routing,
    flashinfer_moe_dispatch_mode,
    maybe_replace_flashinfer_moe_expert_spec,
    use_flashinfer_moe,
)
from tests.unit_tests.test_utilities import Utils


def test_flashinfer_moe_is_opt_in(monkeypatch):
    monkeypatch.delenv("MILES_USE_FLASHINFER_MOE", raising=False)
    assert not use_flashinfer_moe()

    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")
    assert use_flashinfer_moe()


def test_flashinfer_moe_quantization_resolves_config_and_override(monkeypatch):
    config = SimpleNamespace(fp8="e4m3", fp8_recipe="mxfp8", fp4=None, fp4_recipe="nvfp4")
    monkeypatch.delenv("MILES_FLASHINFER_MOE_QUANTIZATION", raising=False)
    assert _flashinfer_moe_quantization(config) == "mxfp8"

    config.fp8 = None
    config.fp4 = "e2m1"
    assert _flashinfer_moe_quantization(config) == "nvfp4"

    config.fp4 = None
    with pytest.raises(ValueError, match="requires an explicit supported quantization"):
        _flashinfer_moe_quantization(config)

    config.fp8 = "e4m3"
    config.fp8_recipe = "delayed"
    with pytest.raises(ValueError, match="does not support active FP8 recipe"):
        _flashinfer_moe_quantization(config)

    config.fp8_recipe = "mxfp8"
    config.fp4 = "e2m1"
    with pytest.raises(ValueError, match="requires exactly one quantization"):
        _flashinfer_moe_quantization(config)

    config.fp4 = None
    monkeypatch.setenv("MILES_FLASHINFER_MOE_QUANTIZATION", "mxfp8")
    assert _flashinfer_moe_quantization(config) == "mxfp8"

    config.fp8 = None
    config.fp4 = "e2m1"
    with pytest.raises(ValueError, match="conflicts with active 'nvfp4'"):
        _flashinfer_moe_quantization(config)

    monkeypatch.setenv("MILES_FLASHINFER_MOE_QUANTIZATION", "fp6")
    with pytest.raises(ValueError, match="nvfp4.*mxfp8"):
        _flashinfer_moe_quantization(config)


def test_flashinfer_moe_has_explicit_unsupported_dispatch_branches():
    with pytest.raises(NotImplementedError, match="no runner branch"):
        _flashinfer_moe_runner_type("fp6")
    with pytest.raises(NotImplementedError, match="no log-description branch"):
        _flashinfer_moe_description("fp6")


def test_flashinfer_moe_dispatch_mode_selects_explicit_collective_branch():
    assert (
        flashinfer_moe_dispatch_mode(SimpleNamespace(moe_token_dispatcher_type="allgather"))
        == "allgather"
    )
    assert (
        flashinfer_moe_dispatch_mode(SimpleNamespace(moe_token_dispatcher_type="alltoall"))
        == "alltoall"
    )


@pytest.mark.parametrize("backend", ["deepep", "hybridep"])
def test_flashinfer_moe_dispatch_mode_rejects_flex_backends_explicitly(backend):
    config = SimpleNamespace(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend=backend)

    with pytest.raises(NotImplementedError, match=rf"flex.*{backend}"):
        flashinfer_moe_dispatch_mode(config)


def test_flashinfer_moe_dispatch_mode_rejects_unknown_dispatcher_explicitly():
    config = SimpleNamespace(moe_token_dispatcher_type="future_dispatcher")

    with pytest.raises(NotImplementedError, match="future_dispatcher.*no execution branch"):
        flashinfer_moe_dispatch_mode(config)


def test_flashinfer_moe_dispatch_mode_rejects_fp32_combine():
    config = SimpleNamespace(moe_token_dispatcher_type="alltoall", moe_combine_in_fp32=True)

    with pytest.raises(ValueError, match="FP32 combine.*router gradients"):
        flashinfer_moe_dispatch_mode(config)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer routed MoE capability checks require Blackwell",
)
def test_flashinfer_moe_dispatch_checks_supported_runner_capabilities():
    assert _flashinfer_moe_runner_type("nvfp4") is _FlashInferNVFP4Runner
    assert _flashinfer_moe_runner_type("mxfp8") is _FlashInferMXFP8Runner
    assert "NVFP4" in _flashinfer_moe_description("nvfp4")
    assert "MXFP8" in _flashinfer_moe_description("mxfp8")


def test_flashinfer_moe_selects_non_te_grouped_parameters(monkeypatch):
    class OtherExperts:
        pass

    original = SimpleNamespace(experts=ModuleSpec(module=OtherExperts))
    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")

    replacement = maybe_replace_flashinfer_moe_expert_spec(original)

    assert replacement is not original
    assert replacement.experts.module is FlashInferGroupedMLP
    assert original.experts.module is OtherExperts


def test_flashinfer_parameter_holder_bypasses_grouped_gemm_runtime():
    assert not issubclass(FlashInferGroupedMLP, GroupedMLP)
    assert not hasattr(FlashInferGroupedMLP, "finish_init")


def test_dense_routing_conversion_preserves_selected_weight_gradients():
    probs = torch.tensor(
        [[0.0, 0.7, 0.3, 0.0], [0.4, 0.0, 0.0, 0.6]], dtype=torch.float32, requires_grad=True
    )
    routing_map = probs.detach() != 0

    weights, ids = _topk_from_dense_routing(probs, routing_map, top_k=2)

    torch.testing.assert_close(ids, torch.tensor([[1, 2], [3, 0]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[0.7, 0.3], [0.6, 0.4]]))
    weights.sum().backward()
    torch.testing.assert_close(probs.grad, routing_map.to(torch.float32))


def test_dense_routing_conversion_preserves_rollout_replay_slot_order():
    probs = torch.tensor([[0.0, 0.2, 0.8, 0.0]], dtype=torch.float32)
    routing_map = probs != 0
    replay_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    weights, ids = _topk_from_dense_routing(
        probs, routing_map, top_k=2, ordered_topk_ids=replay_ids
    )

    torch.testing.assert_close(ids, replay_ids.to(torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[0.2, 0.8]]))


def test_dense_routing_conversion_matches_miles_padding_normalization():
    probs = torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.2, 0.8]], dtype=torch.float32)
    routing_map = probs != 0
    replay_ids = torch.tensor([[-1, -1], [2, 1]], dtype=torch.int64)

    weights, ids = _topk_from_dense_routing(
        probs, routing_map, top_k=2, ordered_topk_ids=replay_ids
    )

    torch.testing.assert_close(ids, torch.tensor([[0, 1], [2, 1]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[0.6, 0.4], [0.8, 0.2]]))


def test_rollout_replay_ids_follow_current_replay_stage(monkeypatch):
    from miles.utils.replay_base import routing_replay_manager

    replay_ids = torch.tensor([[7, 3]], dtype=torch.int64)
    replay = SimpleNamespace(forward_index=1, backward_index=0, top_indices_list=[replay_ids])
    layer = SimpleNamespace(router=SimpleNamespace(routing_replay=replay))
    monkeypatch.setattr(routing_replay_manager, "enabled", True)
    monkeypatch.setattr(routing_replay_manager, "stage", "replay_forward")

    assert _rollout_replay_topk_ids(layer) is replay_ids


def test_pack_topk_ids_matches_flashinfer_packed_contract():
    ids = torch.tensor([[3, 17]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    packed = _pack_topk_ids(ids, weights)

    torch.testing.assert_close(packed >> 16, ids)
    unpacked_weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    torch.testing.assert_close(unpacked_weights, weights.to(torch.bfloat16))


def test_dispatched_topk_inputs_use_global_expert_ids_and_preserve_probability_grads():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
    tokens_per_expert = torch.tensor([2, 0, 3], dtype=torch.long)

    weights, ids = _dispatched_topk_inputs(
        probs, tokens_per_expert, local_expert_offset=4, num_local_experts=3
    )

    torch.testing.assert_close(ids, torch.tensor([[4], [4], [6], [6], [6]], dtype=torch.int32))
    torch.testing.assert_close(weights, probs.detach().unsqueeze(1))
    weights.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_grouped_mlp_weight_view_preserves_megatron_gate_up_order():
    num_experts, hidden_size, intermediate_size = 2, 3, 2
    weight1 = torch.arange(
        num_experts * 2 * intermediate_size * hidden_size, dtype=torch.float32
    ).reshape(num_experts, 2 * intermediate_size, hidden_size)
    weight2 = torch.arange(
        num_experts * hidden_size * intermediate_size, dtype=torch.float32
    ).reshape(num_experts, hidden_size, intermediate_size)
    experts = SimpleNamespace(
        num_local_experts=num_experts,
        linear_fc1=SimpleNamespace(**{f"weight{i}": weight1[i] for i in range(num_experts)}),
        linear_fc2=SimpleNamespace(**{f"weight{i}": weight2[i] for i in range(num_experts)}),
    )

    w13, w2 = _grouped_mlp_weights(experts)

    torch.testing.assert_close(w13, weight1)
    torch.testing.assert_close(w2, weight2)


def test_bf16_surrogate_matches_megatron_expert_formula_and_gradients():
    hidden = torch.tensor([[0.5, -1.0], [1.5, 0.25]], requires_grad=True)
    # Deliberately asymmetric gate/up halves catch an accidental FlashInfer [up, gate] swap.
    w13 = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, -4.0]]], requires_grad=True)
    w2 = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]], requires_grad=True)
    topk_weights = torch.tensor([[0.25], [0.75]], requires_grad=True)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)

    actual = _bf16_local_routed_experts(
        hidden, topk_weights, topk_ids, w13, w2, local_expert_offset=0
    )
    gate, up = F.linear(hidden, w13[0]).chunk(2, dim=-1)
    expected = F.linear((F.silu(gate) * up * topk_weights).to(hidden.dtype), w2[0])
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    actual_grads = (
        hidden.grad.clone(),
        topk_weights.grad.clone(),
        w13.grad.clone(),
        w2.grad.clone(),
    )

    hidden_ref = hidden.detach().requires_grad_()
    weights_ref = topk_weights.detach().requires_grad_()
    w13_ref = w13.detach().requires_grad_()
    w2_ref = w2.detach().requires_grad_()
    gate_ref, up_ref = F.linear(hidden_ref, w13_ref[0]).chunk(2, dim=-1)
    expected_ref = F.linear(
        (F.silu(gate_ref) * up_ref * weights_ref).to(hidden_ref.dtype), w2_ref[0]
    )
    expected_ref.sum().backward()
    expected_grads = (hidden_ref.grad, weights_ref.grad, w13_ref.grad, w2_ref.grad)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_bf16_surrogate_returns_zero_grads_when_rank_has_no_routed_tokens():
    hidden = torch.randn(3, 4, requires_grad=True)
    topk_weights = torch.randn(3, 1, requires_grad=True)
    topk_ids = torch.ones((3, 1), dtype=torch.int32)
    w13 = torch.randn(1, 8, 4, requires_grad=True)
    w2 = torch.randn(1, 4, 4, requires_grad=True)

    output = _bf16_local_routed_experts(
        hidden, topk_weights, topk_ids, w13, w2, local_expert_offset=0
    )
    output.sum().backward()

    for tensor in (hidden, topk_weights, w13, w2):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


def test_flashinfer_autograd_skips_empty_forward_and_invalidates_on_backward():
    class FakeRunner:
        local_expert_offset = 4

        def __init__(self):
            self.forward_calls = 0
            self.invalidate_calls = 0

        def forward(self, *_args, **_kwargs):
            self.forward_calls += 1
            raise AssertionError("FlashInfer must not be launched with zero tokens")

        def invalidate_weights(self):
            self.invalidate_calls += 1

    runner = FakeRunner()
    hidden = torch.empty((0, 4), requires_grad=True)
    topk_weights = torch.empty((0, 1), requires_grad=True)
    topk_ids = torch.empty((0, 1), dtype=torch.int32)
    w13 = torch.randn((2, 8, 4), requires_grad=True)
    w2 = torch.randn((2, 4, 4), requires_grad=True)

    output = _FlashInferForwardBF16Backward.apply(
        hidden, topk_weights, topk_ids, w13, w2, runner, (0, 0, 0, 0)
    )

    assert output.shape == hidden.shape
    assert runner.forward_calls == 0
    output.sum().backward()
    assert runner.invalidate_calls == 1
    for tensor in (hidden, topk_weights, w13, w2):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="TE MXFP8 weight quantization requires Blackwell",
)
@pytest.mark.parametrize("seed", [29, 44])
def test_te_mxfp8_weight_quantization_matches_miles_rollout_sync(seed):
    from miles.utils.mxfp8 import mxfp8_quantize

    torch.manual_seed(seed)
    weight = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16) * 0.02

    actual_qweight, actual_scale = _te_mxfp8_quantize_weight(weight)
    expected_qweight, expected_scale = mxfp8_quantize(weight)

    torch.testing.assert_close(
        actual_qweight.view(torch.uint8), expected_qweight.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="TE MXFP8 weight quantization requires Blackwell",
)
def test_te_mxfp8_gated_weight_adapts_megatron_to_trtllm_order():
    from miles.utils.mxfp8 import mxfp8_quantize

    torch.manual_seed(123)
    gate = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) + 4
    gate_q, gate_scale = mxfp8_quantize(gate)
    up_q, up_scale = mxfp8_quantize(up)

    actual_qweight, actual_scale = _te_mxfp8_quantize_gated_weight(torch.cat((gate, up), dim=0))

    torch.testing.assert_close(
        actual_qweight.view(torch.uint8),
        torch.cat((up_q, gate_q), dim=0).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_scale, torch.cat((up_scale, gate_scale), dim=0), rtol=0, atol=0
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer MXFP8 weight preparation requires Blackwell",
)
def test_flashinfer_mxfp8_prepared_weights_match_sglang_layout():
    from miles.utils.mxfp8 import mxfp8_quantize

    # Match SGLang's application import order; importing the runner first
    # enters its quantization/runner cycle before the FP8 types are registered.
    __import__("sglang.srt.layers.quantization.fp8")
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        align_mxfp8_moe_weights_for_flashinfer_trtllm,
    )

    torch.manual_seed(123)
    num_experts = 2
    hidden_size = 2048
    intermediate_size = 768
    gate = (
        torch.randn(
            num_experts, intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    up = torch.randn_like(gate) * 0.02 + 1.0
    w13 = torch.cat((gate, up), dim=1)
    w2 = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )

    w13_q, w13_sf = mxfp8_quantize(w13)
    gate_q, up_q = w13_q.chunk(2, dim=1)
    gate_sf, up_sf = w13_sf.chunk(2, dim=1)
    w2_q, w2_sf = mxfp8_quantize(w2)

    reference = torch.nn.Module()
    reference.w13_weight = torch.nn.Parameter(torch.cat((up_q, gate_q), dim=1), requires_grad=False)
    reference.w13_weight_scale_inv = torch.nn.Parameter(
        torch.cat((up_sf, gate_sf), dim=1), requires_grad=False
    )
    reference.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
    reference.w2_weight_scale_inv = torch.nn.Parameter(w2_sf, requires_grad=False)
    align_mxfp8_moe_weights_for_flashinfer_trtllm(reference)

    runner = _FlashInferMXFP8Runner(
        num_experts=num_experts,
        local_expert_offset=0,
        local_num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    actual = runner._prepare_weights(w13, w2, (1,))

    pairs = (
        (actual.gemm1_weights.view(torch.uint8), reference.w13_weight.view(torch.uint8)),
        (actual.gemm1_scales, reference.w13_weight_scale_inv),
        (actual.gemm2_weights.view(torch.uint8), reference.w2_weight.view(torch.uint8)),
        (actual.gemm2_scales, reference.w2_weight_scale_inv),
    )
    for actual_tensor, expected_tensor in pairs:
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


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


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="TE NVFP4 weight quantization requires Blackwell",
)
@pytest.mark.parametrize("seed", [29, 44])
def test_te_weight_quantization_matches_miles_rollout_sync(monkeypatch, seed):
    from miles.utils.nvfp4 import nvfp4_quantize_1d

    _set_nvfp4_4over6_env(monkeypatch)

    torch.manual_seed(seed)
    weight = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16) * 0.02

    actual_qweight, actual_block_scale, actual_global_scale = _te_nvfp4_quantize_weight(weight)
    expected_qweight, expected_block_scale, expected_global_scale = nvfp4_quantize_1d(weight)

    torch.testing.assert_close(actual_qweight, expected_qweight, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_block_scale.view(torch.uint8), expected_block_scale.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_global_scale, expected_global_scale, rtol=0, atol=0)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="TE NVFP4 weight quantization requires Blackwell",
)
@pytest.mark.parametrize("seed", [29, 44])
def test_te_gated_weight_quantization_matches_miles_pair_sync(monkeypatch, seed):
    from miles.utils.nvfp4 import nvfp4_quantize_1d_pair

    _set_nvfp4_4over6_env(monkeypatch)

    torch.manual_seed(seed)
    gate_up_weight = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16) * 0.02
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

    actual_qweight, actual_block_scale, actual_global_scale = _te_nvfp4_quantize_gated_weight(
        gate_up_weight
    )
    (
        (expected_gate_qweight, expected_gate_block_scale, expected_gate_scale),
        (expected_up_qweight, expected_up_block_scale, expected_up_scale),
    ) = nvfp4_quantize_1d_pair(gate_weight, up_weight)

    torch.testing.assert_close(
        actual_qweight,
        torch.cat((expected_up_qweight, expected_gate_qweight), dim=0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_block_scale.view(torch.uint8),
        torch.cat((expected_up_block_scale, expected_gate_block_scale), dim=0).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual_global_scale, expected_gate_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_global_scale, expected_up_scale, rtol=0, atol=0)


def _distributed_routing_ids(
    rank: int, world_size: int, num_tokens: int, num_experts: int, top_k: int
) -> torch.Tensor:
    """Route to unique experts on every EP rank except the final rank."""

    local_experts = num_experts // world_size
    active_experts = num_experts - local_experts
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    slot_ids = torch.arange(top_k, device="cuda", dtype=torch.long)
    bases = (rank * num_tokens + token_ids) * top_k
    return ((bases.unsqueeze(1) + slot_ids) % active_experts).contiguous()


def _install_distributed_route(
    layer: MoELayer, route_logits: torch.Tensor, route_ids: torch.Tensor
) -> None:
    """Install deterministic routing while preserving meaningful router gradients."""

    def route(_self, hidden_states, padding_mask=None, input_ids=None):
        del padding_mask, input_ids
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if num_tokens != route_ids.shape[0]:
            raise RuntimeError(f"route token mismatch: {num_tokens} != {route_ids.shape[0]}")
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
        ).scatter(1, route_ids, selected_probs)
        routing_map = torch.zeros_like(probs, dtype=torch.bool)
        routing_map.scatter_(1, route_ids, True)
        return probs, routing_map

    layer.route = types.MethodType(route, layer)


def _bf16_flashinfer_apply(
    hidden_states, topk_weights, topk_ids, w13_gate_up, w2, runner, _weight_key
):
    """Replace only the fused kernel boundary for a topology-identical reference."""

    return _bf16_local_routed_experts(
        hidden_states, topk_weights, topk_ids, w13_gate_up, w2, runner.local_expert_offset
    )


def _run_distributed_layer_once(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    route_ids: torch.Tensor,
    *,
    dispatch_mode: str,
    bf16_reference: bool,
):
    """Run one full layer forward/backward and snapshot its local results."""

    layer.zero_grad(set_to_none=True)
    hidden = hidden_seed.detach().clone().requires_grad_()
    route_logits = logits_seed.detach().clone().requires_grad_()
    _install_distributed_route(layer, route_logits, route_ids)

    received_rows = []
    hook = None
    if dispatch_mode == "alltoall" and not bf16_reference:
        hook = layer.experts.register_forward_pre_hook(
            lambda _module, inputs: received_rows.append(inputs[0].shape[0])
        )

    try:
        with ExitStack() as stack:
            if bf16_reference:
                stack.enter_context(
                    mock.patch.object(
                        _FlashInferForwardBF16Backward, "apply", side_effect=_bf16_flashinfer_apply
                    )
                )
            if dispatch_mode == "alltoall":
                stack.enter_context(
                    mock.patch.object(
                        flashinfer_moe_module._PaddedEPAllGather,
                        "apply",
                        side_effect=AssertionError(
                            "plugin padded all-gather used in all-to-all mode"
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        flashinfer_moe_module._EPAllReduceSum,
                        "apply",
                        side_effect=AssertionError("plugin EP all-reduce used in all-to-all mode"),
                    )
                )
            output, bias = layer(hidden)
    finally:
        if hook is not None:
            hook.remove()

    output.float().sum().backward()
    parameter_grads = []
    missing_grads = int(hidden.grad is None) + int(route_logits.grad is None)
    hidden_grad = torch.zeros_like(hidden) if hidden.grad is None else hidden.grad
    route_grad = torch.zeros_like(route_logits) if route_logits.grad is None else route_logits.grad
    for parameter in layer.experts.parameters():
        if parameter.grad is None:
            missing_grads += 1
            parameter_grads.append(torch.zeros_like(parameter))
        else:
            parameter_grads.append(parameter.grad.detach().clone())

    return SimpleNamespace(
        output=output.detach().clone(),
        bias=bias,
        hidden_grad=hidden_grad.detach().clone(),
        route_grad=route_grad.detach().clone(),
        parameter_grads=parameter_grads,
        missing_grads=missing_grads,
        received_rows=received_rows,
    )


def _global_max(value: torch.Tensor) -> float:
    value = value.detach().float()
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value.item()


def _global_relative_l2(actual_tensors, reference_tensors) -> float:
    error_sq = torch.zeros((), device="cuda")
    reference_sq = torch.zeros((), device="cuda")
    for actual, reference in zip(actual_tensors, reference_tensors):
        error_sq += (actual.float() - reference.float()).square().sum()
        reference_sq += reference.float().square().sum()
    torch.distributed.all_reduce(error_sq)
    torch.distributed.all_reduce(reference_sq)
    return torch.sqrt(error_sq / reference_sq.clamp_min(1e-20)).item()


@pytest.mark.internal
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires torchrun with exactly 8 ranks"
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer routed MoE requires eight Blackwell GPUs",
)
@pytest.mark.parametrize(
    "runner_type",
    [
        pytest.param(_FlashInferNVFP4Runner, id="nvfp4"),
        pytest.param(_FlashInferMXFP8Runner, id="mxfp8"),
    ],
)
@pytest.mark.parametrize(
    "moe_token_dispatcher_type",
    [pytest.param("alltoall", id="alltoall"), pytest.param("allgather", id="allgather")],
)
@pytest.mark.parametrize(
    "model_hyperparameters",
    [
        pytest.param(
            SimpleNamespace(num_experts=32, hidden_size=7168, intermediate_size=2048, top_k=8),
            id="e32-h7168-i2048-topk8",
        )
    ],
)
@pytest.mark.parametrize("num_tokens", [8, 4096])
def test_flashinfer_routed_forward_and_surrogate_backward(
    monkeypatch, runner_type, moe_token_dispatcher_type, model_hyperparameters, num_tokens
):
    """Compare both EP dispatchers against a distributed BF16 full-layer reference."""

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
    received_by_rank = None
    output_nonzero_by_rank = None
    metrics = None
    reference = None
    actual = None

    try:
        monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")
        monkeypatch.setenv("MILES_FLASHINFER_MOE_QUANTIZATION", runner_type.quantization)
        if runner_type.quantization == "nvfp4":
            _set_nvfp4_4over6_env(monkeypatch, flashinfer=True)

        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=16,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=intermediate_size,
            moe_router_topk=top_k,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="none",
            moe_token_dispatcher_type=moe_token_dispatcher_type,
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

        torch.manual_seed(5678 + rank)
        hidden_seed = torch.randn((num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
        logits_seed = torch.randn((num_tokens, top_k), device="cuda", dtype=torch.float32)
        route_ids = _distributed_routing_ids(rank, world_size, num_tokens, num_experts, top_k)

        reference = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
            bf16_reference=True,
        )
        actual = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
            bf16_reference=False,
        )

        forward_error_sq = (actual.output.float() - reference.output.float()).square().sum()
        reference_sq = reference.output.float().square().sum()
        torch.distributed.all_reduce(forward_error_sq)
        torch.distributed.all_reduce(reference_sq)
        forward_rel_l2 = torch.sqrt(forward_error_sq / reference_sq.clamp_min(1e-20)).item()
        hidden_grad_max = _global_max(
            (actual.hidden_grad.float() - reference.hidden_grad.float()).abs().max()
        )
        hidden_grad_rel_l2 = _global_relative_l2((actual.hidden_grad,), (reference.hidden_grad,))
        route_grad_max = _global_max(
            (actual.route_grad.float() - reference.route_grad.float()).abs().max()
        )
        route_grad_rel_l2 = _global_relative_l2((actual.route_grad,), (reference.route_grad,))
        parameter_grad_error = torch.zeros((), device="cuda")
        for actual_grad, reference_grad in zip(actual.parameter_grads, reference.parameter_grads):
            parameter_grad_error = torch.maximum(
                parameter_grad_error, (actual_grad.float() - reference_grad.float()).abs().max()
            )
        parameter_grad_max = _global_max(parameter_grad_error)
        parameter_grad_rel_l2 = _global_relative_l2(
            actual.parameter_grads, reference.parameter_grads
        )
        missing_grads = torch.tensor(
            reference.missing_grads + actual.missing_grads, device="cuda", dtype=torch.int32
        )
        torch.distributed.all_reduce(missing_grads)
        metrics = (
            forward_rel_l2,
            hidden_grad_max,
            hidden_grad_rel_l2,
            route_grad_max,
            route_grad_rel_l2,
            parameter_grad_max,
            parameter_grad_rel_l2,
            missing_grads.item(),
        )

        if moe_token_dispatcher_type == "alltoall":
            received = torch.tensor(
                actual.received_rows if len(actual.received_rows) == 1 else [-1],
                device="cuda",
                dtype=torch.int64,
            )
            received_tensors = [torch.empty_like(received) for _ in range(world_size)]
            torch.distributed.all_gather(received_tensors, received)
            received_by_rank = [value.item() for value in received_tensors]

            output_nonzero = torch.tensor(
                [int(torch.count_nonzero(actual.output).item() > 0)],
                device="cuda",
                dtype=torch.int32,
            )
            output_nonzero_tensors = [torch.empty_like(output_nonzero) for _ in range(world_size)]
            torch.distributed.all_gather(output_nonzero_tensors, output_nonzero)
            output_nonzero_by_rank = [value.item() for value in output_nonzero_tensors]

        owner = layer.experts if moe_token_dispatcher_type == "alltoall" else layer
        actual_runner = getattr(owner, "_flashinfer_moe_runner", None)
        runner_cache_cleared = (
            isinstance(actual_runner, runner_type)
            and actual_runner._prepared is None
            and actual_runner._weight_key is None
        )
        result_metadata = (
            reference.bias,
            actual.bias,
            tuple(actual.output.shape),
            actual.output.dtype,
            bool(torch.isfinite(reference.output).all().item()),
            bool(torch.isfinite(actual.output).all().item()),
            runner_cache_cleared,
        )
        torch.cuda.synchronize()
    finally:
        Utils.destroy_model_parallel()

    assert reference is not None and actual is not None and metrics is not None
    assert result_metadata == (
        None,
        None,
        (num_tokens, 1, hidden_size),
        torch.bfloat16,
        True,
        True,
        True,
    )
    (
        forward_rel_l2,
        hidden_grad_max,
        hidden_grad_rel_l2,
        route_grad_max,
        route_grad_rel_l2,
        parameter_grad_max,
        parameter_grad_rel_l2,
        missing_grads,
    ) = metrics
    if rank == 0:
        print(
            "FlashInfer distributed numerical check: "
            f"quantization={runner_type.quantization}, "
            f"dispatcher={moe_token_dispatcher_type}, top_k={top_k}, "
            f"forward_rel_l2={forward_rel_l2:.6f}, "
            f"hidden_grad_rel_l2={hidden_grad_rel_l2:.6f}, "
            f"hidden_grad_max={hidden_grad_max:.6f}, "
            f"route_grad_rel_l2={route_grad_rel_l2:.6f}, "
            f"route_grad_max={route_grad_max:.6f}, "
            f"parameter_grad_rel_l2={parameter_grad_rel_l2:.6f}, "
            f"parameter_grad_max={parameter_grad_max:.6f}"
        )
    assert missing_grads == 0
    forward_tolerance = 0.25 if runner_type.quantization == "nvfp4" else 0.10
    assert forward_rel_l2 < forward_tolerance
    assert hidden_grad_rel_l2 < 0.01
    assert route_grad_rel_l2 < 0.01
    assert parameter_grad_rel_l2 < 0.01

    if moe_token_dispatcher_type == "alltoall":
        assert received_by_rank is not None
        assert all(received > 0 for received in received_by_rank[:-1])
        assert received_by_rank[-1] == 0
        assert output_nonzero_by_rank is not None
        assert output_nonzero_by_rank[-1] == 1
