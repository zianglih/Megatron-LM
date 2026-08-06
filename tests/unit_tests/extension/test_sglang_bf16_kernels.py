# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import get_module
from miles_megatron_plugins.sglang_bf16_kernels import maybe_replace_sglang_bf16_kernel_specs
from miles_megatron_plugins.sglang_bf16_kernels.selection import (
    resolve_sglang_bf16_kernel_selection,
)


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=2048,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        ffn_hidden_size=6144,
        moe_ffn_hidden_size=768,
        num_moe_experts=128,
        moe_router_topk=8,
        moe_grouped_gemm=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=True,
        perform_initialization=False,
    )


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        ("rmsnorm", ("rmsnorm",)),
        ("qk_rmsnorm", ("qk_rmsnorm",)),
        ("final_rmsnorm", ("final_rmsnorm",)),
        ("all", ("rmsnorm", "qk_rmsnorm", "final_rmsnorm")),
    ],
)
def test_independent_selection(enabled, expected):
    assert resolve_sglang_bf16_kernel_selection(enabled).enabled_names() == expected


def test_unset_selection_is_an_identity_without_importing_sglang(monkeypatch):
    monkeypatch.delenv("MILES_SGLANG_BF16_KERNELS", raising=False)
    config = _config()
    original = object()

    assert maybe_replace_sglang_bf16_kernel_specs(config, original) is original
    assert not config._miles_sglang_bf16_qk_after_rope


@pytest.mark.parametrize(
    "enabled", ["rmsnorm", "qk_rmsnorm", "final_rmsnorm", "rmsnorm,qk_rmsnorm,final_rmsnorm"]
)
def test_selected_specs_change_only_non_moe_norm_boundaries(enabled, monkeypatch):
    pytest.importorskip("sglang.srt.batch_invariant_ops")
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from miles_megatron_plugins.sglang_bf16_kernels.rms_norm import (
        SGLangBF16FinalRMSNorm,
        SGLangBF16QKRMSNorm,
        SGLangBF16RMSNorm,
    )
    from tests.unit_tests.test_utilities import Utils

    config = _config()
    parallel_state.destroy_model_parallel()
    Utils.fake_initialize_model_parallel()
    try:
        original = get_gpt_decoder_block_spec(
            config, use_transformer_engine=True, normalization="RMSNorm", pp_rank=0
        )
        selection = resolve_sglang_bf16_kernel_selection(enabled)
        monkeypatch.setenv("MILES_SGLANG_BF16_KERNELS", enabled)
        replaced = maybe_replace_sglang_bf16_kernel_specs(config, original)
    finally:
        parallel_state.destroy_model_parallel()

    assert replaced is not original
    assert len(replaced.layer_specs) == len(original.layer_specs) == 2
    for native_layer, drop_in_layer in zip(original.layer_specs, replaced.layer_specs):
        native = native_layer.submodules
        drop_in = drop_in_layer.submodules

        # Routed experts, shared experts, router, dispatch, and combine stay opaque.
        assert drop_in.mlp is native.mlp

        native_attention = native.self_attention.submodules
        drop_in_attention = drop_in.self_attention.submodules
        assert drop_in_attention.core_attention is native_attention.core_attention
        assert drop_in_attention.linear_proj is native_attention.linear_proj
        if selection.rmsnorm:
            assert get_module(native_attention.linear_qkv) is TELayerNormColumnParallelLinear
            assert get_module(drop_in_attention.linear_qkv) is TEColumnParallelLinear
            assert get_module(drop_in.input_layernorm) is SGLangBF16RMSNorm
            assert get_module(drop_in.pre_mlp_layernorm) is SGLangBF16RMSNorm
            assert (
                drop_in.sharded_state_dict_keys_map["input_layernorm."]
                == "self_attention.linear_qkv.layer_norm_"
            )
        else:
            assert drop_in.input_layernorm is native.input_layernorm
            assert drop_in.pre_mlp_layernorm is native.pre_mlp_layernorm

        if selection.qk_rmsnorm:
            assert get_module(drop_in_attention.q_layernorm) is SGLangBF16QKRMSNorm
            assert get_module(drop_in_attention.k_layernorm) is SGLangBF16QKRMSNorm
        else:
            assert drop_in_attention.q_layernorm is native_attention.q_layernorm
            assert drop_in_attention.k_layernorm is native_attention.k_layernorm

    if selection.final_rmsnorm:
        assert get_module(replaced.layer_norm) is SGLangBF16FinalRMSNorm
    else:
        assert replaced.layer_norm is original.layer_norm

    assert config._miles_sglang_bf16_qk_after_rope is selection.qk_rmsnorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_input_norm_preserves_sharded_checkpoint_keys(monkeypatch):
    pytest.importorskip("sglang.srt.batch_invariant_ops")
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from tests.unit_tests.test_utilities import Utils

    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=64,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_grouped_gemm=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="transformer_engine",
        use_cpu_initialization=True,
        perform_initialization=False,
    )
    Utils.initialize_model_parallel(1, 1)
    try:
        native_block = get_gpt_decoder_block_spec(
            config, use_transformer_engine=True, normalization="RMSNorm", pp_rank=0
        )
        monkeypatch.setenv("MILES_SGLANG_BF16_KERNELS", "rmsnorm")
        drop_in_block = maybe_replace_sglang_bf16_kernel_specs(config, native_block)
        native_layer = TransformerLayer(config, native_block.layer_specs[0].submodules)
        drop_in_layer = TransformerLayer(config, drop_in_block.layer_specs[0].submodules)
        native_state = native_layer.sharded_state_dict()
        drop_in_state = drop_in_layer.sharded_state_dict()
    finally:
        Utils.destroy_model_parallel()

    from megatron.core.dist_checkpointing import ShardedTensor

    native_keys = {
        value.key for value in native_state.values() if isinstance(value, ShardedTensor)
    }
    drop_in_keys = {
        value.key for value in drop_in_state.values() if isinstance(value, ShardedTensor)
    }
    assert native_keys == drop_in_keys
    assert "self_attention.linear_qkv.layer_norm_weight" in drop_in_keys
    assert "input_layernorm.weight" not in drop_in_keys
    assert (
        drop_in_state["input_layernorm.weight"].key
        == "self_attention.linear_qkv.layer_norm_weight"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden_size", [128, 2048, 7168])
@pytest.mark.parametrize(
    ("module_name", "norm_cast_dtype", "weight_cast_dtype", "output_dtype"),
    [
        ("block", torch.float32, torch.float32, torch.bfloat16),
        ("qk", torch.bfloat16, torch.float32, torch.float32),
        ("final", torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_sglang_rmsnorm_forward_and_surrogate_backward(
    hidden_size, module_name, norm_cast_dtype, weight_cast_dtype, output_dtype
):
    pytest.importorskip("sglang.srt.batch_invariant_ops")
    from miles_megatron_plugins.sglang_bf16_kernels.rms_norm import (
        SGLangBF16FinalRMSNorm,
        SGLangBF16QKRMSNorm,
        SGLangBF16RMSNorm,
        _native_sglang_rms_norm,
    )

    torch.manual_seed(1234)
    config = _config()
    modules = {
        "block": SGLangBF16RMSNorm,
        "qk": SGLangBF16QKRMSNorm,
        "final": SGLangBF16FinalRMSNorm,
    }
    module = modules[module_name](config, hidden_size, eps=1e-6).cuda()
    x = (
        torch.randn(hidden_size, 17, device="cuda", dtype=torch.bfloat16)
        .T.detach()
        .requires_grad_(True)
    )
    assert not x.is_contiguous()
    grad = torch.randn_like(x)
    if output_dtype == torch.float32:
        grad = grad.float()

    actual = module(x)
    reference = _native_sglang_rms_norm(
        x,
        module.weight,
        module.eps,
        norm_cast_dtype=norm_cast_dtype,
        weight_cast_dtype=weight_cast_dtype,
        output_dtype=output_dtype,
    )
    assert actual.dtype == output_dtype
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)

    actual.backward(grad)
    actual_x_grad = x.grad.detach().clone()
    actual_weight_grad = module.weight.grad.detach().clone()

    reference_x = x.detach().contiguous().requires_grad_(True)
    reference_weight = module.weight.detach().requires_grad_(True)
    reference = _native_sglang_rms_norm(
        reference_x,
        reference_weight,
        module.eps,
        norm_cast_dtype=norm_cast_dtype,
        weight_cast_dtype=weight_cast_dtype,
        output_dtype=output_dtype,
    )
    reference.backward(grad)

    torch.testing.assert_close(actual_x_grad, reference_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad, rtol=0, atol=0)
