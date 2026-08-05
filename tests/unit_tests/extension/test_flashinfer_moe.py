# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import types
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from miles_megatron_plugins.flashinfer_moe import (
    DEQUANTIZED_BACKWARD,
    HIGH_PRECISION_BACKWARD,
    FlashInferGroupedMLP,
    _dispatched_topk_inputs,
    _flashinfer_moe_description,
    _flashinfer_moe_quantization,
    _flashinfer_moe_runner_type,
    _FlashInferForwardBF16Backward,
    _FlashInferMXFP8Runner,
    _FlashInferNVFP4Runner,
    _grouped_mlp_weight_parameters,
    _pack_topk_ids,
    _run_flashinfer_forward_with_surrogate,
    _te_mxfp8_quantize_gated_weight,
    _te_mxfp8_quantize_weight,
    _te_nvfp4_quantize_gated_weight,
    _te_nvfp4_quantize_weight,
    _validate_flashinfer_moe_config,
    dequantize_mxfp8_activation,
    dequantize_nvfp4_activation,
    flashinfer_moe_backward_mode,
    flashinfer_moe_dispatch_mode,
    maybe_replace_flashinfer_moe_expert_spec,
    use_flashinfer_moe,
)
from miles_megatron_plugins.flashinfer_moe_surrogate import _BF16GroupedMLPSurrogate
from tests.unit_tests.test_utilities import Utils


def _sequential_bf16_routed_experts(
    hidden_states, topk_weights, w13_gate_up, w2, tokens_per_expert, *, activation_in_fp32=False
):
    """Independent per-expert oracle for the grouped production surrogate."""

    outputs = []
    start = 0
    for local_expert, count in enumerate(tokens_per_expert):
        if count == 0:
            continue
        end = start + count
        fc1 = F.linear(hidden_states[start:end], w13_gate_up[local_expert])
        if activation_in_fp32:
            gate, up = fc1.float().chunk(2, dim=-1)
            activated = (F.silu(gate) * up * topk_weights[start:end].float()).to(
                hidden_states.dtype
            )
        else:
            gate, up = fc1.chunk(2, dim=-1)
            activated = (F.silu(gate) * up * topk_weights[start:end]).to(hidden_states.dtype)
        outputs.append(F.linear(activated, w2[local_expert]))
        start = end
    if not outputs:
        return torch.empty_like(hidden_states)
    return torch.cat(outputs, dim=0)


def test_flashinfer_moe_is_opt_in(monkeypatch):
    monkeypatch.delenv("MILES_USE_FLASHINFER_MOE", raising=False)
    assert not use_flashinfer_moe()

    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")
    assert use_flashinfer_moe()


def test_flashinfer_moe_backward_mode_is_explicit(monkeypatch):
    monkeypatch.delenv("MILES_FLASHINFER_MOE_DEQUANTIZED", raising=False)
    assert flashinfer_moe_backward_mode() == HIGH_PRECISION_BACKWARD

    monkeypatch.setenv("MILES_FLASHINFER_MOE_DEQUANTIZED", "0")
    assert flashinfer_moe_backward_mode() == HIGH_PRECISION_BACKWARD

    monkeypatch.setenv("MILES_FLASHINFER_MOE_DEQUANTIZED", "1")
    assert flashinfer_moe_backward_mode() == DEQUANTIZED_BACKWARD

    monkeypatch.setenv("MILES_FLASHINFER_MOE_DEQUANTIZED", "future")
    with pytest.raises(ValueError, match="expected '0' or '1'"):
        flashinfer_moe_backward_mode()


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


def test_flashinfer_moe_selects_extension_experts_without_mutating_source(monkeypatch):
    class OtherExperts:
        pass

    original = SimpleNamespace(experts=ModuleSpec(module=OtherExperts))
    monkeypatch.setenv("MILES_USE_FLASHINFER_MOE", "1")

    replacement = maybe_replace_flashinfer_moe_expert_spec(original)

    assert replacement is not original
    assert replacement.experts.module is FlashInferGroupedMLP
    assert replacement.experts.submodules is not None
    assert original.experts.module is OtherExperts
    assert original.experts.submodules is None


def test_flashinfer_experts_reuse_megatron_te_parameter_contract():
    assert issubclass(FlashInferGroupedMLP, TEGroupedMLP)


@pytest.mark.parametrize(
    "attribute,message",
    [
        pytest.param("delay_wgrad_compute", "delayed expert weight gradients", id="delayed-wgrad"),
        pytest.param(
            "use_te_activation_func", "Transformer Engine activation modules", id="te-activation"
        ),
    ],
)
def test_flashinfer_moe_rejects_unsupported_te_execution_paths(monkeypatch, attribute, message):
    monkeypatch.setenv("MILES_FLASHINFER_MOE_QUANTIZATION", "mxfp8")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    setattr(config, attribute, True)

    with pytest.raises(ValueError, match=message):
        _validate_flashinfer_moe_config(config)


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


def test_grouped_mlp_weight_parameters_do_not_materialize_stacked_copies():
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

    w13, w2 = _grouped_mlp_weight_parameters(experts)

    assert all(
        weight is getattr(experts.linear_fc1, f"weight{expert}")
        for expert, weight in enumerate(w13)
    )
    assert all(
        weight is getattr(experts.linear_fc2, f"weight{expert}") for expert, weight in enumerate(w2)
    )


def test_bf16_surrogate_matches_megatron_expert_formula_and_gradients():
    hidden = torch.tensor([[0.5, -1.0], [1.5, 0.25]], requires_grad=True)
    # Deliberately asymmetric gate/up halves catch an accidental FlashInfer [up, gate] swap.
    w13 = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, -4.0]]], requires_grad=True)
    w2 = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]], requires_grad=True)
    topk_weights = torch.tensor([[0.25], [0.75]], requires_grad=True)

    actual = _sequential_bf16_routed_experts(
        hidden, topk_weights, tuple(w13.unbind()), tuple(w2.unbind()), (2,)
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


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped BF16 requires CUDA")
@pytest.mark.parametrize("tokens_per_expert", [(2, 0, 3, 0), (0, 0, 5, 0)])
@pytest.mark.parametrize(
    "activation_in_fp32,fused_activation", [(False, False), (False, True), (True, False)]
)
def test_te_grouped_bf16_surrogate_matches_independent_loop(
    tokens_per_expert, activation_in_fp32, fused_activation
):
    num_experts = 4
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
    assert surrogate._fc1 is not None and surrogate._fc2 is not None
    assert all(parameter.is_meta for parameter in surrogate._fc1.op.parameters())
    assert all(parameter.is_meta for parameter in surrogate._fc2.op.parameters())
    assert not surrogate._fc1.op.fuse_wgrad_accumulation
    assert not surrogate._fc2.op.fuse_wgrad_accumulation


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped BF16 requires CUDA")
@pytest.mark.parametrize("outer_quantization", ["mxfp8", "nvfp4"])
def test_te_grouped_bf16_surrogate_disables_outer_quantized_autocast(outer_quantization):
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling, NVFP4BlockScaling

    recipe = (
        MXFP8BlockScaling(fp8_format=Format.E4M3)
        if outer_quantization == "mxfp8"
        else NVFP4BlockScaling()
    )
    torch.manual_seed(2345)
    hidden = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand((5, 1), device="cuda", dtype=torch.float32, requires_grad=True)
    w13 = tuple(
        (torch.randn((256, 128), device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
        for _ in range(2)
    )
    w2 = tuple(
        (torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
        for _ in range(2)
    )
    grad_output = torch.randn_like(hidden)
    surrogate = _BF16GroupedMLPSurrogate(num_experts=2, hidden_size=128, intermediate_size=128)

    with te.autocast(enabled=True, recipe=recipe):
        actual = surrogate(
            hidden, probs, w13, w2, (2, 3), activation_in_fp32=False, fused_activation=False
        )
    actual_inputs = (hidden, probs, *w13, *w2)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    reference_inputs = tuple(tensor.detach().requires_grad_() for tensor in actual_inputs)
    hidden_ref, probs_ref, *expert_weights_ref = reference_inputs
    expected = _sequential_bf16_routed_experts(
        hidden_ref, probs_ref, expert_weights_ref[:2], expert_weights_ref[2:], (2, 3)
    )
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-3, atol=5e-3)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped BF16 requires CUDA")
@pytest.mark.parametrize(
    "w13_trainable,w2_trainable",
    [
        pytest.param((True, False), (False, True), id="partial-experts"),
        pytest.param((True, True), (False, False), id="fc1-only"),
        pytest.param((False, False), (True, True), id="fc2-only"),
        pytest.param((False, False), (False, False), id="weights-frozen"),
    ],
)
def test_flashinfer_grouped_surrogate_respects_frozen_expert_weights(w13_trainable, w2_trainable):
    class BF16Runner:
        local_num_experts = 2

        def __init__(self):
            self.surrogate = _BF16GroupedMLPSurrogate(
                num_experts=2, hidden_size=128, intermediate_size=128
            )
            self.seen_weight_grad_states = None

        def forward(self, hidden_states, *_args):
            return SimpleNamespace(output=hidden_states.detach().clone())

        def bf16_surrogate(self, hidden_states, topk_weights, w13, w2, counts, **kwargs):
            self.seen_weight_grad_states = tuple(weight.requires_grad for weight in (*w13, *w2))
            return self.surrogate(hidden_states, topk_weights, w13, w2, counts, **kwargs)

        def invalidate_weights(self):
            pass

    torch.manual_seed(3456)
    hidden = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand((5, 1), device="cuda", dtype=torch.float32, requires_grad=True)
    w13 = tuple(
        (torch.randn((256, 128), device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(need)
        for need in w13_trainable
    )
    w2 = tuple(
        (torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(need)
        for need in w2_trainable
    )
    runner = BF16Runner()
    grad_output = torch.randn_like(hidden)

    output = _FlashInferForwardBF16Backward.apply(
        hidden,
        probs,
        torch.zeros((5, 1), device="cuda", dtype=torch.int32),
        (2, 3),
        runner,
        (1,),
        HIGH_PRECISION_BACKWARD,
        False,
        False,
        *w13,
        *w2,
    )
    torch.autograd.backward(output, grad_output)

    hidden_ref = hidden.detach().requires_grad_()
    probs_ref = probs.detach().requires_grad_()
    w13_ref = tuple(
        weight.detach().requires_grad_(need) for weight, need in zip(w13, w13_trainable)
    )
    w2_ref = tuple(weight.detach().requires_grad_(need) for weight, need in zip(w2, w2_trainable))
    expected = _sequential_bf16_routed_experts(hidden_ref, probs_ref, w13_ref, w2_ref, (2, 3))
    torch.autograd.backward(expected, grad_output)

    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(probs.grad, probs_ref.grad, rtol=5e-3, atol=5e-3)
    for actual, reference, trainable in zip(
        (*w13, *w2), (*w13_ref, *w2_ref), (*w13_trainable, *w2_trainable)
    ):
        if trainable:
            torch.testing.assert_close(actual.grad, reference.grad, rtol=5e-3, atol=5e-3)
        else:
            assert actual.grad is None
    assert runner.seen_weight_grad_states == (
        *(any(w13_trainable) for _ in w13_trainable),
        *(any(w2_trainable) for _ in w2_trainable),
    )


def test_flashinfer_autograd_skips_empty_forward_and_invalidates_on_backward():
    class FakeRunner:
        local_expert_offset = 4
        local_num_experts = 2

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
        hidden,
        topk_weights,
        topk_ids,
        (0, 0),
        runner,
        (0, 0, 0, 0),
        DEQUANTIZED_BACKWARD,
        False,
        False,
        *w13.unbind(),
        *w2.unbind(),
    )

    assert output.shape == hidden.shape
    assert runner.forward_calls == 0
    output.sum().backward()
    assert runner.invalidate_calls == 1
    for tensor in (hidden, topk_weights, w13, w2):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.parametrize("backward_mode", [HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD])
def test_flashinfer_autograd_uses_selected_forward_operands(backward_mode):
    dequantized_hidden = torch.tensor([[0.25, -0.5], [1.0, 0.75]])
    dequantized_w13 = torch.tensor([[[0.5, -0.25], [0.75, 0.125], [1.25, 0.5], [-0.5, 1.0]]])
    dequantized_w2 = torch.tensor([[[0.5, -1.0], [1.5, 0.25]]])

    class FakeStorage:
        def __init__(self, tensor):
            self._rowwise_data = tensor

        def dequantize(self, *, dtype):
            return self._rowwise_data.to(dtype)

        def prepare_for_saving(self):
            tensors = [self._rowwise_data]
            self._rowwise_data = None
            return tensors, self

        def restore_from_saved(self, tensors):
            self._rowwise_data = tensors[0]
            return tensors[1:]

    class FakeRunner:
        local_expert_offset = 0
        local_num_experts = 1

        def __init__(self):
            self.invalidate_calls = 0
            self.seen_mode = None

        def forward(self, hidden_states, _topk_weights, _topk_ids, _w13, _w2, _weight_key, mode):
            self.seen_mode = mode
            return SimpleNamespace(
                output=hidden_states.detach().clone(),
                backward_hidden_states=FakeStorage(dequantized_hidden),
                backward_w13=(dequantized_w13[0],),
                backward_w2=(dequantized_w2[0],),
            )

        def invalidate_weights(self):
            self.invalidate_calls += 1

        def bf16_surrogate(
            self, hidden_states, topk_weights, w13_gate_up, w2, tokens_per_expert, **_kwargs
        ):
            return _sequential_bf16_routed_experts(
                hidden_states, topk_weights, w13_gate_up, w2, tokens_per_expert
            )

    runner = FakeRunner()
    hidden = torch.tensor([[0.5, -1.0], [1.5, 0.25]], requires_grad=True)
    topk_weights = torch.tensor([[0.25], [0.75]], requires_grad=True)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    w13 = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, -4.0]]], requires_grad=True)
    w2 = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]], requires_grad=True)
    source_snapshots = tuple(tensor.detach().clone() for tensor in (hidden, w13, w2))
    grad_output = torch.tensor([[0.5, -0.25], [1.25, 0.75]])

    output = _FlashInferForwardBF16Backward.apply(
        hidden,
        topk_weights,
        topk_ids,
        (2,),
        runner,
        (0, 0, 0, 0),
        backward_mode,
        False,
        False,
        *w13.unbind(),
        *w2.unbind(),
    )
    torch.autograd.backward(output, grad_output, retain_graph=True)
    torch.autograd.backward(output, grad_output)

    reference_hidden = (
        hidden.detach() if backward_mode == HIGH_PRECISION_BACKWARD else dequantized_hidden
    ).requires_grad_()
    reference_weights = topk_weights.detach().requires_grad_()
    reference_w13 = (
        w13.detach() if backward_mode == HIGH_PRECISION_BACKWARD else dequantized_w13
    ).requires_grad_()
    reference_w2 = (
        w2.detach() if backward_mode == HIGH_PRECISION_BACKWARD else dequantized_w2
    ).requires_grad_()
    reference_output = _sequential_bf16_routed_experts(
        reference_hidden,
        reference_weights,
        tuple(reference_w13.unbind()),
        tuple(reference_w2.unbind()),
        (2,),
    )
    torch.autograd.backward(reference_output, grad_output, retain_graph=True)
    torch.autograd.backward(reference_output, grad_output)

    for actual, expected in (
        (hidden.grad, reference_hidden.grad),
        (topk_weights.grad, reference_weights.grad),
        (w13.grad, reference_w13.grad),
        (w2.grad, reference_w2.grad),
    ):
        torch.testing.assert_close(actual, expected)
    for source, snapshot in zip((hidden, w13, w2), source_snapshots):
        torch.testing.assert_close(source, snapshot)
    assert runner.seen_mode == backward_mode
    assert runner.invalidate_calls == 2


@pytest.mark.parametrize("num_tokens", [0, 2])
def test_flashinfer_dequantized_mode_avoids_backward_payloads_under_no_grad(num_tokens):
    class FakeRunner:
        def __init__(self):
            self.seen_mode = None

        def forward(
            self, hidden_states, _topk_weights, _topk_ids, _w13, _w2, _weight_key, backward_mode
        ):
            self.seen_mode = backward_mode
            return SimpleNamespace(output=hidden_states + 1)

    runner = FakeRunner()
    hidden = torch.ones((num_tokens, 4), requires_grad=True)
    topk_weights = torch.ones((num_tokens, 1), requires_grad=True)
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32)
    w13 = torch.ones((1, 8, 4), requires_grad=True)
    w2 = torch.ones((1, 4, 4), requires_grad=True)

    with torch.no_grad():
        output = _run_flashinfer_forward_with_surrogate(
            hidden,
            topk_weights,
            topk_ids,
            (num_tokens,),
            tuple(w13.unbind()),
            tuple(w2.unbind()),
            runner,
            (1,),
            DEQUANTIZED_BACKWARD,
        )

    expected = hidden if num_tokens == 0 else hidden + 1
    torch.testing.assert_close(output, expected)
    assert not output.requires_grad
    assert runner.seen_mode == (None if num_tokens == 0 else HIGH_PRECISION_BACKWARD)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer forward-operand dequantization requires Blackwell",
)
@pytest.mark.parametrize(
    "quantization,use_4over6,use_256,hidden_size",
    [
        pytest.param("nvfp4", False, False, 16, id="nvfp4-k16"),
        pytest.param("nvfp4", True, False, 128, id="nvfp4-4over6-e4m3-448"),
        pytest.param("nvfp4", True, True, 128, id="nvfp4-4over6-e4m3-256"),
        pytest.param("mxfp8", False, False, 128, id="mxfp8"),
    ],
)
def test_flashinfer_dequantizes_forward_activation_payload(
    monkeypatch, quantization, use_4over6, use_256, hidden_size
):
    torch.manual_seed(123)
    hidden = torch.randn((33, hidden_size), device="cuda", dtype=torch.bfloat16)

    if quantization == "mxfp8":
        from flashinfer import mxfp8_quantize

        data, scales = mxfp8_quantize(hidden, False, backend="cute-dsl")
        actual = dequantize_mxfp8_activation(data, scales, dtype=torch.bfloat16)
        scale_bytes = scales.view(torch.uint8).reshape(33, hidden_size // 32)
        expanded_scales = scale_bytes.repeat_interleave(32, dim=-1)
        expected = torch.where(
            expanded_scales == 0,
            torch.zeros_like(data, dtype=torch.float32),
            torch.ldexp(data.float(), expanded_scales.to(torch.int32) - 127),
        ).to(torch.bfloat16)
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
        actual = dequantize_nvfp4_activation(
            data,
            scales,
            per_token_scale,
            dtype=torch.bfloat16,
            e4m3_max=e4m3_max,
            use_4over6=use_4over6,
        )
        e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
        packed = data.view(torch.uint8).flatten()
        nibbles = torch.stack((packed & 0xF, packed >> 4), dim=1).flatten()
        values = e2m1[(nibbles & 0x7).long()] * torch.where(nibbles & 0x8 != 0, -1.0, 1.0)
        values = values.reshape(33, hidden_size // 16, 16)
        block_scales = scales.view(torch.float8_e4m3fn).float().reshape(33, hidden_size // 16, 1)
        expected = (
            (values * block_scales * per_token_scale.reshape(33, 1, 1))
            .reshape_as(hidden)
            .to(torch.bfloat16)
        )

    # TE's 4-over-6 decoder may choose a different BF16 multiply order than
    # this explicit payload formula; the observed discrepancy is at most one
    # BF16 rounding step. Standard NVFP4 and MXFP8 remain bitwise checks.
    rtol = 0.005 if quantization == "nvfp4" and use_4over6 else 0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer MXFP8 activation dequantization requires Blackwell",
)
def test_flashinfer_mxfp8_zero_scale_dequantizes_to_zero():
    from flashinfer import mxfp8_quantize

    hidden = torch.zeros((3, 128), device="cuda", dtype=torch.bfloat16)
    data, scales = mxfp8_quantize(hidden, False, backend="cute-dsl")

    assert torch.count_nonzero(scales.view(torch.uint8)).item() == 0
    actual = dequantize_mxfp8_activation(data, scales, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, hidden, rtol=0, atol=0)


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
    reason="FlashInfer dequantized expert weights require Blackwell",
)
@pytest.mark.parametrize(
    "runner_type",
    [
        pytest.param(_FlashInferNVFP4Runner, id="nvfp4"),
        pytest.param(_FlashInferMXFP8Runner, id="mxfp8"),
    ],
)
def test_flashinfer_prepares_exact_dequantized_forward_weights(monkeypatch, runner_type):
    if runner_type is _FlashInferNVFP4Runner:
        _set_nvfp4_4over6_env(monkeypatch, flashinfer=True)

    torch.manual_seed(321)
    w13 = torch.randn((1, 256, 128), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((1, 128, 128), device="cuda", dtype=torch.bfloat16)
    source_snapshots = (w13.clone(), w2.clone())
    runner = runner_type(
        num_experts=1,
        local_expert_offset=0,
        local_num_experts=1,
        hidden_size=128,
        intermediate_size=128,
    )

    prepared = runner._prepare_weights(w13, w2, (1,), DEQUANTIZED_BACKWARD)
    assert prepared.backward_w13 is not None
    assert prepared.backward_w2 is not None

    if runner_type is _FlashInferNVFP4Runner:
        *_, w13_quantized = _te_nvfp4_quantize_gated_weight(w13[0], return_quantized=True)
        *_, w2_quantized = _te_nvfp4_quantize_weight(w2[0], return_quantized=True)
    else:
        *_, w13_quantized = _te_mxfp8_quantize_gated_weight(w13[0], return_quantized=True)
        *_, w2_quantized = _te_mxfp8_quantize_weight(w2[0], return_quantized=True)
    expected_w13 = w13_quantized.dequantize(dtype=torch.bfloat16)[:256, :128]
    expected_w2 = w2_quantized.dequantize(dtype=torch.bfloat16)[:128, :128]

    actual_w13 = prepared.backward_w13[0]
    actual_w2 = prepared.backward_w2[0]
    torch.testing.assert_close(actual_w13, expected_w13, rtol=0, atol=0)
    torch.testing.assert_close(actual_w2, expected_w2, rtol=0, atol=0)
    assert torch.count_nonzero(actual_w13 != w13[0]).item() > 0
    assert torch.count_nonzero(actual_w2 != w2[0]).item() > 0
    torch.testing.assert_close(w13, source_snapshots[0], rtol=0, atol=0)
    torch.testing.assert_close(w2, source_snapshots[1], rtol=0, atol=0)

    assert runner._prepare_weights(w13, w2, (1,), DEQUANTIZED_BACKWARD) is prepared
    high_precision = runner._prepare_weights(w13, w2, (1,), HIGH_PRECISION_BACKWARD)
    assert high_precision is not prepared
    assert high_precision.backward_w13 is None
    assert high_precision.backward_w2 is None


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FlashInfer dequantized autograd requires Blackwell",
)
@pytest.mark.parametrize(
    "runner_type",
    [
        pytest.param(_FlashInferNVFP4Runner, id="nvfp4"),
        pytest.param(_FlashInferMXFP8Runner, id="mxfp8"),
    ],
)
@pytest.mark.parametrize("backward_mode", [HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD])
def test_flashinfer_autograd_supports_outstanding_forwards(monkeypatch, runner_type, backward_mode):
    if runner_type is _FlashInferNVFP4Runner:
        _set_nvfp4_4over6_env(monkeypatch, flashinfer=True)
        hidden_size = 128
        intermediate_size = 128
    else:
        hidden_size = 2048
        intermediate_size = 768

    torch.manual_seed(4321)
    runner = runner_type(
        num_experts=4,
        local_expert_offset=0,
        local_num_experts=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    hidden = tuple(
        torch.randn((8, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    )
    topk_weights = tuple(
        torch.rand((8, 1), device="cuda", dtype=torch.float32, requires_grad=True) for _ in range(2)
    )
    topk_ids = torch.zeros((8, 1), device="cuda", dtype=torch.int32)
    w13 = torch.randn(
        (1, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    w2 = torch.randn(
        (1, hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    snapshots = tuple(tensor.detach().clone() for tensor in (*hidden, w13, w2))
    grad_outputs = tuple(torch.randn_like(tensor) for tensor in hidden)

    outputs = tuple(
        _FlashInferForwardBF16Backward.apply(
            hidden[index],
            topk_weights[index],
            topk_ids,
            (8,),
            runner,
            (1,),
            backward_mode,
            False,
            False,
            *w13.unbind(),
            *w2.unbind(),
        )
        for index in range(2)
    )
    torch.autograd.backward(outputs[1], grad_outputs[1], retain_graph=True)
    torch.autograd.backward(outputs[1], grad_outputs[1])
    torch.autograd.backward(outputs[0], grad_outputs[0])

    reference_hidden_values = tuple(tensor.detach() for tensor in hidden)
    reference_w13_values = tuple(w13.detach().unbind())
    reference_w2_values = tuple(w2.detach().unbind())
    if backward_mode == DEQUANTIZED_BACKWARD:
        reference_runner = runner_type(
            num_experts=4,
            local_expert_offset=0,
            local_num_experts=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        with torch.no_grad():
            reference_results = tuple(
                reference_runner.forward(
                    tensor,
                    probs,
                    topk_ids,
                    tuple(w13.unbind()),
                    tuple(w2.unbind()),
                    (1,),
                    DEQUANTIZED_BACKWARD,
                )
                for tensor, probs in zip(hidden, topk_weights)
            )
            reference_hidden_values = tuple(
                result.backward_hidden_states.dequantize(dtype=torch.bfloat16)
                for result in reference_results
            )
            reference_w13_values = reference_results[0].backward_w13
            reference_w2_values = reference_results[0].backward_w2

    hidden_ref = tuple(value.detach().requires_grad_() for value in reference_hidden_values)
    probs_ref = tuple(value.detach().requires_grad_() for value in topk_weights)
    w13_ref = tuple(value.detach().requires_grad_() for value in reference_w13_values)
    w2_ref = tuple(value.detach().requires_grad_() for value in reference_w2_values)
    reference_outputs = tuple(
        _sequential_bf16_routed_experts(hidden_ref[index], probs_ref[index], w13_ref, w2_ref, (8,))
        for index in range(2)
    )
    torch.autograd.backward(reference_outputs[1], grad_outputs[1], retain_graph=True)
    torch.autograd.backward(reference_outputs[1], grad_outputs[1])
    torch.autograd.backward(reference_outputs[0], grad_outputs[0])

    for actual, expected in zip(hidden, hidden_ref):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-3, atol=5e-3)
    for actual, expected in zip(topk_weights, probs_ref):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(w13.grad, torch.stack([weight.grad for weight in w13_ref]))
    torch.testing.assert_close(w2.grad, torch.stack([weight.grad for weight in w2_ref]))
    for source, snapshot in zip((*hidden, w13, w2), snapshots):
        torch.testing.assert_close(source, snapshot, rtol=0, atol=0)
    assert runner._prepared is None
    assert runner._weight_key is None


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
    hidden_states,
    topk_weights,
    _topk_ids,
    tokens_per_expert,
    runner,
    _weight_key,
    _backward_mode,
    activation_in_fp32,
    _fused_activation,
    *expert_weights,
):
    """Replace only the fused kernel boundary for a topology-identical reference."""

    w13_gate_up = expert_weights[: runner.local_num_experts]
    w2 = expert_weights[runner.local_num_experts :]
    if hidden_states.shape[0] == 0:
        zero = hidden_states.sum() + topk_weights.sum()
        for weight in expert_weights:
            zero = zero + weight.reshape(-1)[0] * 0
        return torch.empty_like(hidden_states) + zero
    return _sequential_bf16_routed_experts(
        hidden_states,
        topk_weights,
        w13_gate_up,
        w2,
        tokens_per_expert,
        activation_in_fp32=activation_in_fp32,
    )


def _dequantized_flashinfer_apply(
    hidden_states,
    topk_weights,
    topk_ids,
    tokens_per_expert,
    runner,
    weight_key,
    backward_mode,
    activation_in_fp32,
    _fused_activation,
    *expert_weights,
):
    """Build an ordinary-autograd reference from forward-derived QDQ operands."""

    assert backward_mode == DEQUANTIZED_BACKWARD
    w13_gate_up = expert_weights[: runner.local_num_experts]
    w2 = expert_weights[runner.local_num_experts :]
    if hidden_states.shape[0] == 0:
        return _bf16_flashinfer_apply(
            hidden_states,
            topk_weights,
            topk_ids,
            tokens_per_expert,
            runner,
            weight_key,
            backward_mode,
            activation_in_fp32,
            _fused_activation,
            *expert_weights,
        )
    with torch.no_grad():
        result = runner.forward(
            hidden_states, topk_weights, topk_ids, w13_gate_up, w2, weight_key, backward_mode
        )
        decoded_hidden = result.backward_hidden_states.dequantize(dtype=hidden_states.dtype)
        decoded_w13 = result.backward_w13
        decoded_w2 = result.backward_w2
    hidden_ref = decoded_hidden.detach() + (hidden_states - hidden_states.detach())
    w13_ref = tuple(
        decoded.detach() + (source - source.detach())
        for decoded, source in zip(decoded_w13, w13_gate_up)
    )
    w2_ref = tuple(
        decoded.detach() + (source - source.detach()) for decoded, source in zip(decoded_w2, w2)
    )
    surrogate = _sequential_bf16_routed_experts(
        hidden_ref,
        topk_weights,
        w13_ref,
        w2_ref,
        tokens_per_expert,
        activation_in_fp32=activation_in_fp32,
    )
    return result.output.detach() + (surrogate - surrogate.detach())


def _run_distributed_layer_once(
    layer: MoELayer,
    hidden_seed: torch.Tensor,
    logits_seed: torch.Tensor,
    grad_seed: torch.Tensor,
    route_ids: torch.Tensor,
    *,
    dispatch_mode: str,
    surrogate_reference: str | None = None,
):
    """Run one full layer forward/backward and snapshot its local results."""

    layer.zero_grad(set_to_none=True)
    hidden = hidden_seed.detach().clone().requires_grad_()
    route_logits = logits_seed.detach().clone().requires_grad_()
    _install_distributed_route(layer, route_logits, route_ids)

    received_rows = []
    hook = None
    if dispatch_mode == "alltoall" and surrogate_reference is None:
        hook = layer.experts.register_forward_pre_hook(
            lambda _module, inputs: received_rows.append(inputs[0].shape[0])
        )

    try:
        with ExitStack() as stack:
            if surrogate_reference == "bf16":
                stack.enter_context(
                    mock.patch.object(
                        _FlashInferForwardBF16Backward, "apply", side_effect=_bf16_flashinfer_apply
                    )
                )
            elif surrogate_reference == DEQUANTIZED_BACKWARD:
                stack.enter_context(
                    mock.patch.object(
                        _FlashInferForwardBF16Backward,
                        "apply",
                        side_effect=_dequantized_flashinfer_apply,
                    )
                )
            elif surrogate_reference is not None:
                raise ValueError(f"unknown distributed surrogate reference {surrogate_reference!r}")
            output, bias = layer(hidden)
    finally:
        if hook is not None:
            hook.remove()

    torch.autograd.backward(output, grad_seed)
    if surrogate_reference == DEQUANTIZED_BACKWARD:
        layer.experts._flashinfer_moe_runner.invalidate_weights()
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


def _global_abs_max(tensors) -> float:
    value = torch.zeros((), device="cuda")
    for tensor in tensors:
        value = torch.maximum(value, tensor.detach().float().abs().max())
    return _global_max(value)


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
    high_precision = None
    dequantized_reference = None
    dequantized = None
    run_completed = False

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
            gradient_accumulation_fusion=True,
            use_cpu_initialization=False,
        )
        layer = MoELayer(
            config, MoESubmodules(experts=ModuleSpec(module=GroupedMLP)), layer_number=1
        ).cuda()
        layer.train()
        parameter_ownership_before = tuple(
            (name, id(parameter)) for name, parameter in layer.named_parameters()
        )
        state_keys_before = tuple(layer.state_dict())

        torch.manual_seed(5678 + rank)
        hidden_seed = torch.randn((num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
        logits_seed = torch.randn((num_tokens, top_k), device="cuda", dtype=torch.float32)
        grad_seed = torch.randn_like(hidden_seed)
        route_ids = _distributed_routing_ids(rank, world_size, num_tokens, num_experts, top_k)

        monkeypatch.setenv("MILES_FLASHINFER_MOE_DEQUANTIZED", "0")
        reference = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
            surrogate_reference="bf16",
        )
        high_precision = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
        )

        monkeypatch.setenv("MILES_FLASHINFER_MOE_DEQUANTIZED", "1")
        dequantized_reference = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
            surrogate_reference=DEQUANTIZED_BACKWARD,
        )
        dequantized = _run_distributed_layer_once(
            layer,
            hidden_seed,
            logits_seed,
            grad_seed,
            route_ids,
            dispatch_mode=moe_token_dispatcher_type,
        )

        forward_error_sq = (high_precision.output.float() - reference.output.float()).square().sum()
        reference_sq = reference.output.float().square().sum()
        torch.distributed.all_reduce(forward_error_sq)
        torch.distributed.all_reduce(reference_sq)
        forward_rel_l2 = torch.sqrt(forward_error_sq / reference_sq.clamp_min(1e-20)).item()
        per_token_forward_rel_l2 = _global_max(
            torch.sqrt(
                (high_precision.output.float() - reference.output.float()).square().sum(dim=-1)
                / reference.output.float().square().sum(dim=-1).clamp_min(1e-20)
            ).max()
        )
        hidden_grad_max = _global_max(
            (high_precision.hidden_grad.float() - reference.hidden_grad.float()).abs().max()
        )
        hidden_grad_reference_max = _global_abs_max((reference.hidden_grad,))
        hidden_grad_rel_l2 = _global_relative_l2(
            (high_precision.hidden_grad,), (reference.hidden_grad,)
        )
        route_grad_max = _global_max(
            (high_precision.route_grad.float() - reference.route_grad.float()).abs().max()
        )
        route_grad_reference_max = _global_abs_max((reference.route_grad,))
        route_grad_rel_l2 = _global_relative_l2(
            (high_precision.route_grad,), (reference.route_grad,)
        )
        parameter_grad_error = torch.zeros((), device="cuda")
        for actual_grad, reference_grad in zip(
            high_precision.parameter_grads, reference.parameter_grads
        ):
            parameter_grad_error = torch.maximum(
                parameter_grad_error, (actual_grad.float() - reference_grad.float()).abs().max()
            )
        parameter_grad_max = _global_max(parameter_grad_error)
        parameter_grad_reference_max = _global_abs_max(reference.parameter_grads)
        parameter_grad_rel_l2 = _global_relative_l2(
            high_precision.parameter_grads, reference.parameter_grads
        )
        dequantized_hidden_grad_rel_l2 = _global_relative_l2(
            (dequantized.hidden_grad,), (dequantized_reference.hidden_grad,)
        )
        dequantized_hidden_grad_reference_max = _global_abs_max(
            (dequantized_reference.hidden_grad,)
        )
        dequantized_route_grad_rel_l2 = _global_relative_l2(
            (dequantized.route_grad,), (dequantized_reference.route_grad,)
        )
        dequantized_parameter_grad_rel_l2 = _global_relative_l2(
            dequantized.parameter_grads, dequantized_reference.parameter_grads
        )
        dequantized_hidden_grad_max = _global_max(
            (dequantized.hidden_grad.float() - dequantized_reference.hidden_grad.float())
            .abs()
            .max()
        )
        dequantized_route_grad_max = _global_max(
            (dequantized.route_grad.float() - dequantized_reference.route_grad.float()).abs().max()
        )
        dequantized_parameter_grad_error = torch.zeros((), device="cuda")
        for actual_grad, reference_grad in zip(
            dequantized.parameter_grads, dequantized_reference.parameter_grads
        ):
            dequantized_parameter_grad_error = torch.maximum(
                dequantized_parameter_grad_error,
                (actual_grad.float() - reference_grad.float()).abs().max(),
            )
        dequantized_parameter_grad_max = _global_max(dequantized_parameter_grad_error)
        dequantized_vs_high_hidden_grad_rel_l2 = _global_relative_l2(
            (dequantized.hidden_grad,), (high_precision.hidden_grad,)
        )
        dequantized_vs_high_route_grad_rel_l2 = _global_relative_l2(
            (dequantized.route_grad,), (high_precision.route_grad,)
        )
        dequantized_vs_high_parameter_grad_rel_l2 = _global_relative_l2(
            dequantized.parameter_grads, high_precision.parameter_grads
        )
        forward_mode_rel_l2 = max(
            _global_relative_l2((dequantized.output,), (high_precision.output,)),
            _global_relative_l2((dequantized_reference.output,), (high_precision.output,)),
        )
        forward_mode_max = max(
            _global_max((dequantized.output.float() - high_precision.output.float()).abs().max()),
            _global_max(
                (dequantized_reference.output.float() - high_precision.output.float()).abs().max()
            ),
        )
        missing_grads = torch.tensor(
            reference.missing_grads
            + high_precision.missing_grads
            + dequantized_reference.missing_grads
            + dequantized.missing_grads,
            device="cuda",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(missing_grads)
        metrics = (
            forward_rel_l2,
            per_token_forward_rel_l2,
            hidden_grad_max,
            hidden_grad_reference_max,
            hidden_grad_rel_l2,
            route_grad_max,
            route_grad_reference_max,
            route_grad_rel_l2,
            parameter_grad_max,
            parameter_grad_reference_max,
            parameter_grad_rel_l2,
            dequantized_hidden_grad_rel_l2,
            dequantized_hidden_grad_reference_max,
            dequantized_route_grad_rel_l2,
            dequantized_parameter_grad_rel_l2,
            dequantized_hidden_grad_max,
            dequantized_route_grad_max,
            dequantized_parameter_grad_max,
            dequantized_vs_high_hidden_grad_rel_l2,
            dequantized_vs_high_route_grad_rel_l2,
            dequantized_vs_high_parameter_grad_rel_l2,
            forward_mode_rel_l2,
            forward_mode_max,
            missing_grads.item(),
        )

        if moe_token_dispatcher_type == "alltoall":
            received = torch.tensor(
                dequantized.received_rows if len(dequantized.received_rows) == 1 else [-1],
                device="cuda",
                dtype=torch.int64,
            )
            received_tensors = [torch.empty_like(received) for _ in range(world_size)]
            torch.distributed.all_gather(received_tensors, received)
            received_by_rank = [value.item() for value in received_tensors]

            output_nonzero = torch.tensor(
                [int(torch.count_nonzero(dequantized.output).item() > 0)],
                device="cuda",
                dtype=torch.int32,
            )
            output_nonzero_tensors = [torch.empty_like(output_nonzero) for _ in range(world_size)]
            torch.distributed.all_gather(output_nonzero_tensors, output_nonzero)
            output_nonzero_by_rank = [value.item() for value in output_nonzero_tensors]

        actual_runner = getattr(layer.experts, "_flashinfer_moe_runner", None)
        runner_cache_cleared = (
            isinstance(actual_runner, runner_type)
            and actual_runner._prepared is None
            and actual_runner._weight_key is None
        )
        parameter_ownership_unchanged = parameter_ownership_before == tuple(
            (name, id(parameter)) for name, parameter in layer.named_parameters()
        ) and state_keys_before == tuple(layer.state_dict())
        result_metadata = (
            reference.bias,
            high_precision.bias,
            dequantized_reference.bias,
            dequantized.bias,
            tuple(dequantized.output.shape),
            dequantized.output.dtype,
            bool(torch.isfinite(reference.output).all().item()),
            bool(torch.isfinite(high_precision.output).all().item()),
            bool(torch.isfinite(dequantized_reference.output).all().item()),
            bool(torch.isfinite(dequantized.output).all().item()),
            runner_cache_cleared,
            parameter_ownership_unchanged,
        )
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

    assert all(
        result is not None
        for result in (reference, high_precision, dequantized_reference, dequantized)
    )
    assert metrics is not None
    assert result_metadata == (
        None,
        None,
        None,
        None,
        (num_tokens, 1, hidden_size),
        torch.bfloat16,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    (
        forward_rel_l2,
        per_token_forward_rel_l2,
        hidden_grad_max,
        hidden_grad_reference_max,
        hidden_grad_rel_l2,
        route_grad_max,
        route_grad_reference_max,
        route_grad_rel_l2,
        parameter_grad_max,
        parameter_grad_reference_max,
        parameter_grad_rel_l2,
        dequantized_hidden_grad_rel_l2,
        dequantized_hidden_grad_reference_max,
        dequantized_route_grad_rel_l2,
        dequantized_parameter_grad_rel_l2,
        dequantized_hidden_grad_max,
        dequantized_route_grad_max,
        dequantized_parameter_grad_max,
        dequantized_vs_high_hidden_grad_rel_l2,
        dequantized_vs_high_route_grad_rel_l2,
        dequantized_vs_high_parameter_grad_rel_l2,
        forward_mode_rel_l2,
        forward_mode_max,
        missing_grads,
    ) = metrics
    if rank == 0:
        print(
            "FlashInfer distributed numerical check: "
            f"quantization={runner_type.quantization}, "
            f"dispatcher={moe_token_dispatcher_type}, top_k={top_k}, "
            f"forward_rel_l2={forward_rel_l2:.6f}, "
            f"per_token_forward_rel_l2={per_token_forward_rel_l2:.6f}, "
            f"hidden_grad_rel_l2={hidden_grad_rel_l2:.6f}, "
            f"hidden_grad_max={hidden_grad_max:.6f}, "
            f"route_grad_rel_l2={route_grad_rel_l2:.6f}, "
            f"route_grad_max={route_grad_max:.6f}, "
            f"parameter_grad_rel_l2={parameter_grad_rel_l2:.6f}, "
            f"parameter_grad_max={parameter_grad_max:.6f}, "
            f"dequantized_hidden_grad_rel_l2={dequantized_hidden_grad_rel_l2:.6f}, "
            f"forward_mode_rel_l2={forward_mode_rel_l2:.6f}, "
            f"forward_mode_max={forward_mode_max:.6f}, "
            f"dequantized_route_grad_rel_l2={dequantized_route_grad_rel_l2:.6f}, "
            "dequantized_parameter_grad_rel_l2="
            f"{dequantized_parameter_grad_rel_l2:.6f}, "
            "dequantized_vs_high_hidden_grad_rel_l2="
            f"{dequantized_vs_high_hidden_grad_rel_l2:.6f}, "
            "dequantized_vs_high_route_grad_rel_l2="
            f"{dequantized_vs_high_route_grad_rel_l2:.6f}, "
            "dequantized_vs_high_parameter_grad_rel_l2="
            f"{dequantized_vs_high_parameter_grad_rel_l2:.6f}"
        )
    assert missing_grads == 0
    assert forward_mode_rel_l2 < 0.01
    assert forward_mode_max < 0.125
    forward_tolerance = 0.25 if runner_type.quantization == "nvfp4" else 0.10
    per_token_forward_tolerance = 0.30 if runner_type.quantization == "nvfp4" else 0.15
    assert forward_rel_l2 < forward_tolerance
    assert per_token_forward_rel_l2 < per_token_forward_tolerance
    assert hidden_grad_rel_l2 < 0.01
    assert route_grad_rel_l2 < 0.01
    assert parameter_grad_rel_l2 < 0.01
    gradient_atol = 0.125
    gradient_rtol = 0.01
    assert hidden_grad_max < gradient_atol + gradient_rtol * hidden_grad_reference_max
    assert route_grad_max < gradient_atol + gradient_rtol * route_grad_reference_max
    assert parameter_grad_max < gradient_atol + gradient_rtol * parameter_grad_reference_max
    assert dequantized_hidden_grad_rel_l2 < 0.01
    assert dequantized_route_grad_rel_l2 < 1e-6
    assert dequantized_parameter_grad_rel_l2 < 1e-6
    assert dequantized_hidden_grad_max < (
        gradient_atol + gradient_rtol * dequantized_hidden_grad_reference_max
    )
    assert dequantized_route_grad_max < 1e-5
    assert dequantized_parameter_grad_max < 1e-5
    assert (
        max(
            dequantized_vs_high_hidden_grad_rel_l2,
            dequantized_vs_high_route_grad_rel_l2,
            dequantized_vs_high_parameter_grad_rel_l2,
        )
        > 0
    )

    if moe_token_dispatcher_type == "alltoall":
        assert received_by_rank is not None
        assert all(received > 0 for received in received_by_rank[:-1])
        assert received_by_rank[-1] == 0
        assert sum(received_by_rank) == world_size * num_tokens * top_k
        assert output_nonzero_by_rank is not None
        assert output_nonzero_by_rank[-1] == 1
