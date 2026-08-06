"""Opt-in SGLang BF16 kernel replacements for Megatron modules."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .selection import resolve_sglang_bf16_kernel_selection

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig


def maybe_replace_sglang_bf16_kernel_specs(
    config: "TransformerConfig", submodules: "TransformerBlockSubmodules"
) -> "TransformerBlockSubmodules":
    """Return cloned specs with only the explicitly selected kernels replaced.

    Keeping the implementation import behind the empty-selection check makes the
    default Megatron path independent of an installed SGLang package. Once a
    kernel is selected, import and validation errors are intentionally surfaced.
    """

    selection = resolve_sglang_bf16_kernel_selection()
    config._miles_sglang_bf16_qk_after_rope = selection.qk_rmsnorm
    if not selection:
        return submodules

    from .specs import replace_sglang_bf16_kernel_specs

    return replace_sglang_bf16_kernel_specs(config, submodules, selection)


def maybe_install_megatron_debug_hooks(block) -> None:
    """Install verbose hooks only when the experiment explicitly requests them."""

    if not os.environ.get("MILES_SGLANG_BF16_DEBUG_DIR"):
        return
    from .debug import maybe_install_megatron_debug_hooks as install

    install(block)


__all__ = ["maybe_install_megatron_debug_hooks", "maybe_replace_sglang_bf16_kernel_specs"]
