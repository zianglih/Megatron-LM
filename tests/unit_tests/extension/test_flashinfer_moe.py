# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from miles_megatron_plugins.flashinfer_moe import (
    _bf16_local_routed_experts,
    _FlashInferForwardBF16Backward,
    _FlashInferNVFP4Runner,
    FlashInferGroupedMLP,
    _grouped_mlp_weights,
    _pack_topk_ids,
    _rollout_replay_topk_ids,
    _te_nvfp4_quantize_gated_weight,
    _te_nvfp4_quantize_weight,
    _topk_from_dense_routing,
    maybe_replace_flashinfer_moe_expert_spec,
    use_flashinfer_moe,
)


def test_flashinfer_moe_is_opt_in(monkeypatch):
    monkeypatch.delenv("MILES_USE_FLASHINFER_MOE", raising=False)
    assert not use_flashinfer_moe()

    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")
    assert use_flashinfer_moe()


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
        [[0.0, 0.7, 0.3, 0.0], [0.4, 0.0, 0.0, 0.6]],
        dtype=torch.float32,
        requires_grad=True,
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
    replay = SimpleNamespace(
        forward_index=1,
        backward_index=0,
        top_indices_list=[replay_ids],
    )
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
        linear_fc1=SimpleNamespace(
            **{f"weight{i}": weight1[i] for i in range(num_experts)}
        ),
        linear_fc2=SimpleNamespace(
            **{f"weight{i}": weight2[i] for i in range(num_experts)}
        ),
    )

    w13, w2 = _grouped_mlp_weights(experts)

    torch.testing.assert_close(w13, weight1)
    torch.testing.assert_close(w2, weight2)


def test_bf16_surrogate_matches_megatron_expert_formula_and_gradients():
    hidden = torch.tensor([[0.5, -1.0], [1.5, 0.25]], requires_grad=True)
    # Deliberately asymmetric gate/up halves catch an accidental FlashInfer [up, gate] swap.
    w13 = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, -4.0]]],
        requires_grad=True,
    )
    w2 = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]], requires_grad=True)
    topk_weights = torch.tensor([[0.25], [0.75]], requires_grad=True)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)

    actual = _bf16_local_routed_experts(
        hidden, topk_weights, topk_ids, w13, w2, local_expert_offset=0
    )
    gate, up = F.linear(hidden, w13[0]).chunk(2, dim=-1)
    expected = F.linear(
        (F.silu(gate) * up * topk_weights).to(hidden.dtype),
        w2[0],
    )
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
        (F.silu(gate_ref) * up_ref * weights_ref).to(hidden_ref.dtype),
        w2_ref[0],
    )
    expected_ref.sum().backward()
    expected_grads = (
        hidden_ref.grad,
        weights_ref.grad,
        w13_ref.grad,
        w2_ref.grad,
    )

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


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="TE NVFP4 weight quantization requires Blackwell",
)
@pytest.mark.parametrize("seed", [29, 44])
def test_te_weight_quantization_matches_miles_rollout_sync(monkeypatch, seed):
    from miles.utils.nvfp4 import nvfp4_quantize_1d

    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")

    torch.manual_seed(seed)
    weight = (
        torch.randn((3, 128), device="cuda", dtype=torch.bfloat16) * 0.02
    )

    actual_qweight, actual_block_scale, actual_global_scale = (
        _te_nvfp4_quantize_weight(weight)
    )
    expected_qweight, expected_block_scale, expected_global_scale = (
        nvfp4_quantize_1d(weight)
    )

    torch.testing.assert_close(actual_qweight, expected_qweight, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_block_scale.view(torch.uint8),
        expected_block_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_global_scale, expected_global_scale, rtol=0, atol=0
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="TE NVFP4 weight quantization requires Blackwell",
)
@pytest.mark.parametrize("seed", [29, 44])
def test_te_gated_weight_quantization_matches_miles_pair_sync(monkeypatch, seed):
    from miles.utils.nvfp4 import nvfp4_quantize_1d_pair

    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")

    torch.manual_seed(seed)
    gate_up_weight = (
        torch.randn((32, 128), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

    actual_qweight, actual_block_scale, actual_global_scale = (
        _te_nvfp4_quantize_gated_weight(gate_up_weight)
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
        torch.cat(
            (expected_up_block_scale, expected_gate_block_scale), dim=0
        ).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_global_scale, expected_gate_scale, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_global_scale, expected_up_scale, rtol=0, atol=0
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer routed NVFP4 MoE requires Blackwell",
)
def test_flashinfer_per_token_4over6_forward_and_surrogate_backward(monkeypatch):
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")
    monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    num_tokens = 8
    num_experts = 2
    hidden_size = 128
    intermediate_size = 128
    top_k = 2
    hidden = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).requires_grad_()
    topk_ids = torch.tensor([[0, 1]] * num_tokens, dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    ).requires_grad_()
    runner = _FlashInferNVFP4Runner(
        num_experts=num_experts,
        local_expert_offset=0,
        local_num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    output = _FlashInferForwardBF16Backward.apply(
        hidden,
        topk_weights,
        topk_ids,
        w13,
        w2,
        runner,
        (0, 0, 0, 0),
    )

    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for tensor in (hidden, topk_weights, w13, w2):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert runner._prepared is None
    assert runner._weight_key is None
