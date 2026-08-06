"""Verbose, opt-in tensor taps for the two-layer parity harness."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from megatron.core.transformer.identity_op import IdentityOp

DEBUG_DIR_ENV = "MILES_SGLANG_BF16_DEBUG_DIR"
DEBUG_MIN_TOKENS_ENV = "MILES_SGLANG_BF16_DEBUG_MIN_TOKENS"
DEBUG_MAX_CALLS_ENV = "MILES_SGLANG_BF16_DEBUG_MAX_CALLS"


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _reshape_attention_output(value: Any, query: Any) -> Any:
    output_tensor = _first_tensor(value)
    query_tensor = _first_tensor(query)
    if output_tensor is None or query_tensor is None or query_tensor.ndim < 3:
        return value
    num_heads, head_dim = query_tensor.shape[-2:]
    if output_tensor.shape[-2:] == (num_heads, head_dim):
        return output_tensor
    if output_tensor.shape[-1] == num_heads * head_dim:
        return output_tensor.reshape(*output_tensor.shape[:-1], num_heads, head_dim)
    raise ValueError(
        f"Unexpected attention output shape {tuple(output_tensor.shape)} for "
        f"{num_heads} heads of width {head_dim}"
    )


class _TensorDumper:
    def __init__(self, base_dir: str, role: str) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.enabled = rank == 0
        self.role = role
        self.output_dir = Path(base_dir) / role / f"rank_{rank:03d}"
        self.min_tokens = int(os.environ.get(DEBUG_MIN_TOKENS_ENV, "4096"))
        self.max_calls = int(os.environ.get(DEBUG_MAX_CALLS_ENV, "1"))
        self.calls: dict[tuple[str, str], int] = defaultdict(int)
        self.lock = threading.Lock()
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def dump(self, name: str, phase: str, value: Any) -> None:
        if not self.enabled:
            return
        tensor = _first_tensor(value)
        if tensor is None or tensor.ndim == 0 or tensor.shape[0] < self.min_tokens:
            return

        key = (name, phase)
        with self.lock:
            call = self.calls[key]
            self.calls[key] += 1
        if call >= self.max_calls:
            return

        cpu = tensor.detach().contiguous().cpu()
        finite = (
            torch.isfinite(cpu)
            if cpu.is_floating_point()
            else torch.ones_like(cpu, dtype=torch.bool)
        )
        numeric = cpu.float() if cpu.is_floating_point() else cpu.to(torch.float32)
        metadata = {
            "role": self.role,
            "name": name,
            "phase": phase,
            "call": call,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "finite": bool(finite.all()),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "mean": float(numeric.mean()),
            "rms": float(numeric.square().mean().sqrt()),
        }
        stem = f"{_safe_name(name)}__{phase}__call_{call:03d}"
        path = self.output_dir / f"{stem}.pt"
        torch.save({"value": cpu, "meta": metadata}, path)
        with (self.output_dir / "summary.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**metadata, "path": str(path)}, sort_keys=True) + "\n")
        print(
            "[sglang-bf16-parity] "
            f"role={self.role} name={name} phase={phase} shape={tuple(cpu.shape)} "
            f"dtype={cpu.dtype} min={metadata['min']:.7g} max={metadata['max']:.7g} "
            f"mean={metadata['mean']:.7g} rms={metadata['rms']:.7g} path={path}",
            flush=True,
        )


def _debug_site(module_name: str) -> str | None:
    layer = re.fullmatch(r"layers\.(\d+)", module_name)
    if layer:
        return f"layer_{layer.group(1)}.layer"

    patterns = (
        (r"layers\.(\d+)\.input_layernorm", "input_rmsnorm"),
        (r"layers\.(\d+)\.self_attention\.linear_qkv", "qkv"),
        (r"layers\.(\d+)\.self_attention\.q_layernorm", "q_rmsnorm"),
        (r"layers\.(\d+)\.self_attention\.k_layernorm", "k_rmsnorm"),
        (r"layers\.(\d+)\.self_attention\.core_attention", "attention_core"),
        (r"layers\.(\d+)\.self_attention\.linear_proj", "attention_output_projection"),
        (r"layers\.(\d+)\.pre_mlp_layernorm", "pre_mlp_rmsnorm"),
        (r"layers\.(\d+)\.mlp", "moe_boundary"),
    )
    for pattern, site in patterns:
        match = re.fullmatch(pattern, module_name)
        if match:
            return f"layer_{match.group(1)}.{site}"
    if module_name == "final_layernorm":
        return "final_rmsnorm"
    return None


def maybe_install_megatron_debug_hooks(block: torch.nn.Module) -> None:
    """Install full-tensor taps without changing any forward implementation."""

    base_dir = os.environ.get(DEBUG_DIR_ENV)
    if not base_dir:
        return
    dumper = _TensorDumper(base_dir, "megatron")
    if not dumper.enabled:
        return

    for module_name, module in block.named_modules():
        site = _debug_site(module_name)
        if site is None:
            continue
        if site.endswith(".input_rmsnorm") and isinstance(module, IdentityOp):
            # Native TE fuses this norm into QKV and does not expose the exact
            # intermediate. The QKV output remains the first comparable tap.
            continue

        def hook(_module, args, kwargs, output, *, name=site):
            dumper.dump(name, "input", (args, kwargs))
            if name.endswith(".attention_core") and len(args) >= 3:
                layer = name.removesuffix(".attention_core")
                dumper.dump(f"{layer}.q_after_rope", "output", args[0])
                dumper.dump(f"{layer}.k_after_rope", "output", args[1])
                dumper.dump(f"{layer}.value", "output", args[2])
                dumper.dump(name, "output", _reshape_attention_output(output, args[0]))
            else:
                dumper.dump(name, "output", output)

        module.register_forward_hook(hook, with_kwargs=True)


__all__ = [
    "DEBUG_DIR_ENV",
    "DEBUG_MAX_CALLS_ENV",
    "DEBUG_MIN_TOKENS_ENV",
    "maybe_install_megatron_debug_hooks",
]
