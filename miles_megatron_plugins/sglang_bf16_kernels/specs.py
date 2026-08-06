"""Shallow spec rewrites for independently selected SGLang BF16 kernels."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TENorm,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec, get_module

from .rms_norm import SGLangBF16FinalRMSNorm, SGLangBF16QKRMSNorm, SGLangBF16RMSNorm
from .selection import SGLangBF16KernelSelection, validate_sglang_bf16_kernel_config

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules


def _clone_spec(spec: ModuleSpec, *, module=None, submodules=None) -> ModuleSpec:
    cloned = copy.copy(spec)
    cloned.params = dict(spec.params)
    cloned.metainfo = dict(spec.metainfo)
    if module is not None:
        cloned.module = module
    if submodules is not None:
        cloned.submodules = submodules
    return cloned


def _replace_module(spec_or_type, replacement):
    if isinstance(spec_or_type, ModuleSpec):
        return _clone_spec(spec_or_type, module=replacement)
    return replacement


def _is_module(spec_or_type, expected: type) -> bool:
    if spec_or_type is None:
        return False
    return get_module(spec_or_type) is expected


def _norm_spec(module: type, site: str) -> ModuleSpec:
    return ModuleSpec(module=module, params={"site": site})


def _decompose_fused_qkv_norm(layer_submodules) -> None:
    attention_spec = layer_submodules.self_attention
    if not isinstance(attention_spec, ModuleSpec):
        raise ValueError("Expected self_attention to be a ModuleSpec")
    attention_submodules = copy.copy(attention_spec.submodules)
    if _is_module(attention_submodules.linear_qkv, TELayerNormColumnParallelLinear):
        attention_submodules.linear_qkv = _replace_module(
            attention_submodules.linear_qkv, TEColumnParallelLinear
        )
        layer_submodules.self_attention = _clone_spec(
            attention_spec, submodules=attention_submodules
        )
        layer_submodules.input_layernorm = _norm_spec(SGLangBF16RMSNorm, "input_rmsnorm")
        layer_submodules.sharded_state_dict_keys_map["input_layernorm."] = (
            "self_attention.linear_qkv.layer_norm_"
        )
    elif _is_module(layer_submodules.input_layernorm, IdentityOp):
        raise ValueError(
            "Cannot install rmsnorm: input norm is fused into an unsupported QKV module"
        )
    else:
        if not _is_module(layer_submodules.input_layernorm, TENorm):
            raise ValueError("Cannot install rmsnorm: unsupported standalone input norm")
        layer_submodules.input_layernorm = _norm_spec(SGLangBF16RMSNorm, "input_rmsnorm")


def _replace_pre_mlp_norm(layer_submodules) -> None:
    if not _is_module(layer_submodules.pre_mlp_layernorm, IdentityOp):
        if not _is_module(layer_submodules.pre_mlp_layernorm, TENorm):
            raise ValueError("Cannot install rmsnorm: unsupported standalone pre-MLP norm")
        layer_submodules.pre_mlp_layernorm = _norm_spec(SGLangBF16RMSNorm, "pre_mlp_rmsnorm")
        return

    mlp_spec = layer_submodules.mlp
    if not isinstance(mlp_spec, ModuleSpec) or not _is_module(mlp_spec, MLP):
        raise ValueError("Cannot install rmsnorm: pre-MLP norm is fused into an unsupported module")
    mlp_submodules = copy.copy(mlp_spec.submodules)
    if not _is_module(mlp_submodules.linear_fc1, TELayerNormColumnParallelLinear):
        raise ValueError("Cannot install rmsnorm: dense pre-MLP norm is not a TE LayerNormLinear")
    mlp_submodules.linear_fc1 = _replace_module(mlp_submodules.linear_fc1, TEColumnParallelLinear)
    layer_submodules.mlp = _clone_spec(mlp_spec, submodules=mlp_submodules)
    layer_submodules.pre_mlp_layernorm = _norm_spec(SGLangBF16RMSNorm, "pre_mlp_rmsnorm")
    layer_submodules.sharded_state_dict_keys_map["pre_mlp_layernorm."] = (
        "mlp.linear_fc1.layer_norm_"
    )


def _replace_qk_norms(layer_submodules) -> None:
    attention_spec = layer_submodules.self_attention
    if not isinstance(attention_spec, ModuleSpec):
        raise ValueError("Expected self_attention to be a ModuleSpec")
    attention_submodules = copy.copy(attention_spec.submodules)
    if not _is_module(attention_submodules.q_layernorm, TENorm) or not _is_module(
        attention_submodules.k_layernorm, TENorm
    ):
        raise ValueError("qk_rmsnorm requires native TE Q/K RMSNorm modules")
    attention_submodules.q_layernorm = _norm_spec(SGLangBF16QKRMSNorm, "q_rmsnorm")
    attention_submodules.k_layernorm = _norm_spec(SGLangBF16QKRMSNorm, "k_rmsnorm")
    layer_submodules.self_attention = _clone_spec(attention_spec, submodules=attention_submodules)


def _replace_layer_spec(layer_spec: ModuleSpec, selection: SGLangBF16KernelSelection) -> ModuleSpec:
    if not isinstance(layer_spec, ModuleSpec):
        raise ValueError("SGLang BF16 kernel replacement requires ModuleSpec layers")

    layer_submodules = copy.copy(layer_spec.submodules)
    layer_submodules.sharded_state_dict_keys_map = dict(
        layer_submodules.sharded_state_dict_keys_map
    )
    original_mlp = layer_submodules.mlp

    if selection.rmsnorm:
        _decompose_fused_qkv_norm(layer_submodules)
        _replace_pre_mlp_norm(layer_submodules)
    if selection.qk_rmsnorm:
        _replace_qk_norms(layer_submodules)

    # MoE specs are opaque to this drop-in. A standalone pre-MoE norm may be
    # replaced, but the MoE object itself must retain identity.
    if not _is_module(original_mlp, MLP) and layer_submodules.mlp is not original_mlp:
        raise AssertionError("SGLang BF16 kernels must not replace MoE modules")

    return _clone_spec(layer_spec, submodules=layer_submodules)


def replace_sglang_bf16_kernel_specs(
    config, submodules: "TransformerBlockSubmodules", selection: SGLangBF16KernelSelection
) -> "TransformerBlockSubmodules":
    """Clone and rewrite only the selected non-MoE module specs."""

    validate_sglang_bf16_kernel_config(config)
    cloned = copy.copy(submodules)
    if selection.rmsnorm or selection.qk_rmsnorm:
        cloned.layer_specs = [
            _replace_layer_spec(layer_spec, selection) for layer_spec in submodules.layer_specs
        ]
    else:
        cloned.layer_specs = list(submodules.layer_specs)

    if selection.final_rmsnorm:
        if not _is_module(submodules.layer_norm, TENorm):
            raise ValueError("final_rmsnorm requires the native TE final RMSNorm")
        cloned.layer_norm = _norm_spec(SGLangBF16FinalRMSNorm, "final_rmsnorm")
    return cloned


__all__ = ["replace_sglang_bf16_kernel_specs"]
