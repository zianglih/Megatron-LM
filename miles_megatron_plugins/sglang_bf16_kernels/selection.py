"""Selection and validation for isolated SGLang BF16 kernel drop-ins."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

SGLANG_BF16_KERNELS_ENV = "MILES_SGLANG_BF16_KERNELS"
SUPPORTED_SGLANG_BF16_KERNELS = ("rmsnorm", "qk_rmsnorm", "final_rmsnorm")


@dataclass(frozen=True)
class SGLangBF16KernelSelection:
    """The independently selectable non-MoE BF16 kernel sites."""

    rmsnorm: bool = False
    qk_rmsnorm: bool = False
    final_rmsnorm: bool = False

    def __bool__(self) -> bool:
        return self.rmsnorm or self.qk_rmsnorm or self.final_rmsnorm

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(name for name in SUPPORTED_SGLANG_BF16_KERNELS if getattr(self, name))


def resolve_sglang_bf16_kernel_selection(raw_value: str | None = None) -> SGLangBF16KernelSelection:
    """Parse a strict comma-separated kernel set.

    The unset and empty values preserve native Megatron behavior. ``all`` is a
    convenience alias for the complete currently supported set.
    """

    raw_value = os.environ.get(SGLANG_BF16_KERNELS_ENV, "") if raw_value is None else raw_value
    values = {item.strip() for item in raw_value.split(",") if item.strip()}
    if not values:
        return SGLangBF16KernelSelection()
    if "all" in values:
        if values != {"all"}:
            raise ValueError(
                f"{SGLANG_BF16_KERNELS_ENV}=all cannot be combined with individual kernels"
            )
        values = set(SUPPORTED_SGLANG_BF16_KERNELS)

    unsupported = values.difference(SUPPORTED_SGLANG_BF16_KERNELS)
    if unsupported:
        supported = ", ".join(SUPPORTED_SGLANG_BF16_KERNELS)
        unknown = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Unsupported {SGLANG_BF16_KERNELS_ENV} value(s): {unknown}. "
            f"Supported values: {supported}"
        )

    return SGLangBF16KernelSelection(
        **{name: name in values for name in SUPPORTED_SGLANG_BF16_KERNELS}
    )


def validate_sglang_bf16_kernel_config(config) -> None:
    """Reject configurations outside the intentionally narrow BF16 contract."""

    if not config.bf16 or config.fp16:
        raise ValueError("SGLang BF16 kernel drop-ins require plain BF16 training")
    if config.params_dtype != torch.bfloat16:
        raise ValueError("SGLang BF16 kernel drop-ins require params_dtype=torch.bfloat16")
    if config.transformer_impl != "transformer_engine":
        raise ValueError("SGLang BF16 kernel drop-ins currently require Transformer Engine specs")
    if config.fp8 is not None:
        raise ValueError("SGLang BF16 kernel drop-ins do not support FP8 training")
    if config.fp4 is not None:
        raise ValueError("SGLang BF16 kernel drop-ins do not support FP4 training")
    if config.normalization != "RMSNorm":
        raise ValueError("SGLang BF16 RMSNorm drop-ins require normalization='RMSNorm'")
    if config.layernorm_zero_centered_gamma:
        raise ValueError("SGLang BF16 RMSNorm drop-ins do not support zero-centered gamma")
    if getattr(config, "true_on_policy_contract", None) is not None:
        raise ValueError(
            "MILES_SGLANG_BF16_KERNELS is independent of Megatron's broad "
            "true_on_policy_contract; do not enable both"
        )


__all__ = [
    "SGLANG_BF16_KERNELS_ENV",
    "SUPPORTED_SGLANG_BF16_KERNELS",
    "SGLangBF16KernelSelection",
    "resolve_sglang_bf16_kernel_selection",
    "validate_sglang_bf16_kernel_config",
]
