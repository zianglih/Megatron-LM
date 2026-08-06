"""SGLang Triton RMSNorm forward with a Megatron-compatible BF16 surface."""

from __future__ import annotations

from typing import Optional

import torch
from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm

from megatron.core.transformer.transformer_config import TransformerConfig


def _native_sglang_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    norm_cast_dtype: torch.dtype,
    weight_cast_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """PyTorch expression for one explicit SGLang RMSNorm dtype contract."""

    normalized = x.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized.to(norm_cast_dtype) * weight.to(weight_cast_dtype)).to(output_dtype)


class _SGLangBF16RMSNormFunction(torch.autograd.Function):
    """Use the SGLang forward and recompute the plain BF16 backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_cast_dtype: torch.dtype,
        weight_cast_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.norm_cast_dtype = norm_cast_dtype
        ctx.weight_cast_dtype = weight_cast_dtype
        ctx.output_dtype = output_dtype
        ctx.save_for_backward(x, weight)
        return true_on_policy_rms_norm(
            x,
            weight,
            eps,
            cast_x_before_out_mul=True,
            norm_cast_dtype=norm_cast_dtype,
            weight_cast_dtype=weight_cast_dtype,
            output_dtype=output_dtype,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            native_x = x.detach().requires_grad_(True)
            native_weight = weight.detach().requires_grad_(True)
            native_output = _native_sglang_rms_norm(
                native_x,
                native_weight,
                ctx.eps,
                norm_cast_dtype=ctx.norm_cast_dtype,
                weight_cast_dtype=ctx.weight_cast_dtype,
                output_dtype=ctx.output_dtype,
            )
            grad_x, grad_weight = torch.autograd.grad(
                native_output, (native_x, native_weight), grad_output, create_graph=create_graph
            )
        return (
            grad_x if ctx.needs_input_grad[0] else None,
            grad_weight if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
        )


class SGLangBF16RMSNorm(torch.nn.Module):
    """RMSNorm parameter/checkpoint surface shared by all selected sites."""

    backend_name = "sglang_bf16"
    norm_cast_dtype = torch.float32
    weight_cast_dtype = torch.float32
    output_dtype = torch.bfloat16

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
        site: str = "rmsnorm",
    ) -> None:
        super().__init__()
        del persist_layer_norm

        if normalization != "RMSNorm" or config.normalization != "RMSNorm":
            raise ValueError("SGLangBF16RMSNorm only supports RMSNorm")
        if zero_centered_gamma or config.layernorm_zero_centered_gamma:
            raise ValueError("SGLangBF16RMSNorm does not support zero-centered gamma")

        self.hidden_size = hidden_size
        self.eps = eps
        self.site = site
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.register_parameter("bias", None)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def _apply(self, fn):
        super()._apply(fn)
        self.weight.data = self.weight.data.float()
        if self.weight.grad is not None:
            self.weight.grad.data = self.weight.grad.data.float()
        return self

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residual is not None or post_residual_addition is not None:
            raise ValueError(
                "The isolated SGLang BF16 RMSNorm drop-in expects Megatron to materialize "
                "the residual before normalization"
            )
        if not x.is_cuda:
            raise ValueError("SGLangBF16RMSNorm requires a CUDA tensor")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"SGLangBF16RMSNorm requires BF16 input, got {x.dtype}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {x.shape[-1]}")
        return _SGLangBF16RMSNormFunction.apply(
            x.contiguous(),
            self.weight,
            self.eps,
            self.norm_cast_dtype,
            self.weight_cast_dtype,
            self.output_dtype,
        )


class SGLangBF16QKRMSNorm(SGLangBF16RMSNorm):
    """Q/K RMSNorm that retains FP32 through RoPE, as SGLang does."""

    norm_cast_dtype = torch.bfloat16
    output_dtype = torch.float32


class SGLangBF16FinalRMSNorm(SGLangBF16RMSNorm):
    """Final RMSNorm; the LM head itself remains native and out of scope."""

    norm_cast_dtype = torch.bfloat16
    weight_cast_dtype = torch.bfloat16


__all__ = ["SGLangBF16FinalRMSNorm", "SGLangBF16QKRMSNorm", "SGLangBF16RMSNorm"]
