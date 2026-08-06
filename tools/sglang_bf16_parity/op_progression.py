"""Build an operation-ordered drift progression from matched tensor records."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

TensorKey = tuple[str, str]


@dataclass(frozen=True)
class OperationSpec:
    name: str
    input_keys: tuple[TensorKey, ...]
    output_keys: tuple[TensorKey, ...]
    note: str = ""


def _layer_specs(layer: int) -> tuple[OperationSpec, ...]:
    prefix = f"layer_{layer}"
    layer_input_keys = () if layer == 0 else ((f"layer_{layer - 1}.layer", "output"),)
    return (
        OperationSpec(
            f"layer {layer} input",
            layer_input_keys,
            ((f"{prefix}.layer", "input"),),
            "Deeper layers carry the preceding layer output rather than a new initial condition.",
        ),
        OperationSpec(
            f"layer {layer} input RMSNorm",
            ((f"{prefix}.input_rmsnorm", "input"),),
            ((f"{prefix}.input_rmsnorm", "output"),),
            "Native TE fuses this boundary into QKV in the baseline.",
        ),
        OperationSpec(
            f"layer {layer} QKV projection",
            ((f"{prefix}.qkv", "input"),),
            (
                (f"{prefix}.q_rmsnorm", "input"),
                (f"{prefix}.k_rmsnorm", "input"),
                (f"{prefix}.value", "output"),
            ),
            "Packed QKV output is layout-opaque; canonical Q, K, and V taps are aggregated.",
        ),
        OperationSpec(
            f"layer {layer} Q RMSNorm",
            ((f"{prefix}.q_rmsnorm", "input"),),
            ((f"{prefix}.q_rmsnorm", "output"),),
        ),
        OperationSpec(
            f"layer {layer} K RMSNorm",
            ((f"{prefix}.k_rmsnorm", "input"),),
            ((f"{prefix}.k_rmsnorm", "output"),),
        ),
        OperationSpec(
            f"layer {layer} Q RoPE + BF16 cast",
            ((f"{prefix}.q_rmsnorm", "output"),),
            ((f"{prefix}.q_after_rope", "output"),),
        ),
        OperationSpec(
            f"layer {layer} K RoPE + BF16 cast",
            ((f"{prefix}.k_rmsnorm", "output"),),
            ((f"{prefix}.k_after_rope", "output"),),
        ),
        OperationSpec(
            f"layer {layer} fused attention core",
            (
                (f"{prefix}.q_after_rope", "output"),
                (f"{prefix}.k_after_rope", "output"),
                (f"{prefix}.value", "output"),
            ),
            ((f"{prefix}.attention_core", "output"),),
            "QK, softmax, and PV are fused and cannot be separated by these dumps.",
        ),
        OperationSpec(
            f"layer {layer} attention output projection",
            ((f"{prefix}.attention_output_projection", "input"),),
            ((f"{prefix}.attention_output_projection", "output"),),
        ),
        OperationSpec(
            f"layer {layer} attention residual merge",
            ((f"{prefix}.layer", "input"), (f"{prefix}.attention_output_projection", "output")),
            ((f"{prefix}.pre_mlp_rmsnorm", "input"),),
            "The merge output is observed at the following RMSNorm input.",
        ),
        OperationSpec(
            f"layer {layer} pre-MoE RMSNorm",
            ((f"{prefix}.pre_mlp_rmsnorm", "input"),),
            ((f"{prefix}.pre_mlp_rmsnorm", "output"),),
        ),
        OperationSpec(
            f"layer {layer} MoE control boundary",
            ((f"{prefix}.moe_boundary", "input"),),
            ((f"{prefix}.moe_boundary", "output"),),
            "MoE internals are a declared non-goal and remain opaque.",
        ),
        OperationSpec(
            f"layer {layer} MoE residual merge",
            ((f"{prefix}.pre_mlp_rmsnorm", "input"), (f"{prefix}.moe_boundary", "output")),
            ((f"{prefix}.layer", "output"),),
            "The merge output is observed at the layer output.",
        ),
    )


def _operation_specs(records: list[dict]) -> tuple[OperationSpec, ...]:
    layers = sorted(
        {
            int(record["name"].split(".", 1)[0].removeprefix("layer_"))
            for record in records
            if record["name"].startswith("layer_")
        }
    )
    specs = tuple(spec for layer in layers for spec in _layer_specs(layer))
    if any(record["name"] == "final_rmsnorm" for record in records):
        specs += (
            OperationSpec(
                "final RMSNorm", (("final_rmsnorm", "input"),), (("final_rmsnorm", "output"),)
            ),
        )
    return specs


def _aggregate(index: dict[TensorKey, dict], keys: tuple[TensorKey, ...]) -> dict | None:
    if not keys:
        return None
    records = [index.get(key) for key in keys]
    if any(record is None or record["status"] != "compared" for record in records):
        return None
    compared = [record for record in records if record is not None]
    reference_l2 = math.sqrt(sum(record["reference_l2"] ** 2 for record in compared))
    difference_l2 = math.sqrt(sum(record["difference_l2"] ** 2 for record in compared))
    numel = sum(record["numel"] for record in compared)
    exact_count = sum(record["exact_count"] for record in compared)
    return {
        "rel_l2": difference_l2 / max(reference_l2, sys.float_info.min),
        "reference_l2": reference_l2,
        "difference_l2": difference_l2,
        "max_abs": max(record["max_abs"] for record in compared),
        "mean_abs": sum(record["mean_abs"] * record["numel"] for record in compared) / numel,
        "exact_fraction": exact_count / numel,
        "numel": numel,
    }


def _attribution(input_metrics: dict | None, output_metrics: dict | None, has_inputs: bool) -> str:
    if output_metrics is None:
        return "not exposed"
    if not has_inputs:
        return "initial condition"
    if input_metrics is None:
        return "compound / unavailable"
    if output_metrics["max_abs"] == 0:
        return "exact"
    if input_metrics["max_abs"] == 0:
        return "introduced at this boundary"
    return "cumulative; not attributable"


def build_operation_progression(records: list[dict]) -> list[dict]:
    """Aggregate canonical tensor comparisons into logical operation boundaries."""

    index = {(record["name"], record["phase"]): record for record in records}
    progression = []
    for spec in _operation_specs(records):
        input_metrics = _aggregate(index, spec.input_keys)
        output_metrics = _aggregate(index, spec.output_keys)
        progression.append(
            {
                "operation": spec.name,
                "inputs": [f"{name}:{phase}" for name, phase in spec.input_keys],
                "outputs": [f"{name}:{phase}" for name, phase in spec.output_keys],
                "input": input_metrics,
                "output": output_metrics,
                "output_minus_input_rel_l2": (
                    output_metrics["rel_l2"] - input_metrics["rel_l2"]
                    if input_metrics is not None and output_metrics is not None
                    else None
                ),
                "attribution": _attribution(input_metrics, output_metrics, bool(spec.input_keys)),
                "note": spec.note,
            }
        )
    return progression


def _metric(metrics: dict | None, key: str) -> str:
    return "-" if metrics is None else f"{metrics[key]:.7g}"


def write_operation_progression(report: dict, output_dir: Path) -> None:
    """Write reusable JSON and a human-readable operation progression."""

    progression = report["operation_progression"]
    (output_dir / "op_progression.json").write_text(
        json.dumps(progression, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Operation-by-operation drift progression",
        "",
        "Relative L2 is measured between the matched SGLang and Megatron tensors selected by "
        "the comparator. "
        "Multi-input or multi-output rows use the joint L2 norm across components. The difference "
        "column subtracts normalized drifts from different tensor spaces; it is descriptive only, "
        "not an operator gain or additive causal error budget.",
        "",
        "| Operation | Input rel L2 | Output rel L2 | Output - input rel L2 | Output max abs | Output exact | Attribution |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in progression:
        difference = (
            "-"
            if row["output_minus_input_rel_l2"] is None
            else f"{row['output_minus_input_rel_l2']:+.7g}"
        )
        exact = "-" if row["output"] is None else f"{row['output']['exact_fraction']:.3%}"
        lines.append(
            f"| {row['operation']} | {_metric(row['input'], 'rel_l2')} | "
            f"{_metric(row['output'], 'rel_l2')} | {difference} | "
            f"{_metric(row['output'], 'max_abs')} | {exact} | {row['attribution']} |"
        )
    lines.extend(
        [
            "",
            "Limitations: baseline TE fuses input RMSNorm into QKV; packed QKV output, fused "
            "attention internals, residual implementation details, and MoE internals are not "
            "separately attributable from these natural-forward dumps.",
            "",
        ]
    )
    (output_dir / "op_progression.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build_operation_progression", "write_operation_progression"]
