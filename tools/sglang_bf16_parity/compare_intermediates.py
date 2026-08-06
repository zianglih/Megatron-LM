"""Compare matched full-tensor dumps from SGLang and Megatron."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .op_progression import build_operation_progression, write_operation_progression
else:
    from op_progression import build_operation_progression, write_operation_progression


def _layer_taps(layer: int) -> tuple[str, ...]:
    prefix = f"layer_{layer}"
    return (
        f"{prefix}.layer",
        f"{prefix}.input_rmsnorm",
        f"{prefix}.qkv",
        f"{prefix}.q_rmsnorm",
        f"{prefix}.k_rmsnorm",
        f"{prefix}.q_after_rope",
        f"{prefix}.k_after_rope",
        f"{prefix}.value",
        f"{prefix}.attention_core",
        f"{prefix}.attention_output_projection",
        f"{prefix}.pre_mlp_rmsnorm",
        f"{prefix}.moe_boundary",
    )


TAP_SEQUENCE = _layer_taps(0) + _layer_taps(1) + ("final_rmsnorm",)


def _layer_tap_order(layer: int) -> tuple[tuple[str, str], ...]:
    prefix = f"layer_{layer}"
    return (
        (f"{prefix}.layer", "input"),
        (f"{prefix}.input_rmsnorm", "input"),
        (f"{prefix}.input_rmsnorm", "output"),
        (f"{prefix}.qkv", "input"),
        (f"{prefix}.qkv", "output"),
        (f"{prefix}.q_rmsnorm", "input"),
        (f"{prefix}.q_rmsnorm", "output"),
        (f"{prefix}.k_rmsnorm", "input"),
        (f"{prefix}.k_rmsnorm", "output"),
        (f"{prefix}.q_after_rope", "output"),
        (f"{prefix}.k_after_rope", "output"),
        (f"{prefix}.value", "output"),
        (f"{prefix}.attention_core", "input"),
        (f"{prefix}.attention_core", "output"),
        (f"{prefix}.attention_output_projection", "input"),
        (f"{prefix}.attention_output_projection", "output"),
        (f"{prefix}.pre_mlp_rmsnorm", "input"),
        (f"{prefix}.pre_mlp_rmsnorm", "output"),
        (f"{prefix}.moe_boundary", "input"),
        (f"{prefix}.moe_boundary", "output"),
        (f"{prefix}.layer", "output"),
    )


TAP_PHASE_SEQUENCE = (
    _layer_tap_order(0)
    + _layer_tap_order(1)
    + (("final_rmsnorm", "input"), ("final_rmsnorm", "output"))
)
TAP_ORDER = {key: index for index, key in enumerate(TAP_PHASE_SEQUENCE)}

# Input RMSNorm is fused into the baseline TE QKV module, so it participates in
# ordering whenever exposed by an ablation but is not required in every arm.
EXPECTED_OUTPUT_TAPS = tuple(
    name
    for name in TAP_SEQUENCE
    if not name.endswith(".input_rmsnorm") and not name.endswith(".qkv")
)


def _load_dumps(root: Path) -> dict[tuple[str, str, int], tuple[torch.Tensor, dict]]:
    dumps = {}
    for path in sorted(root.rglob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        meta = record["meta"]
        key = (meta["name"], meta["phase"], int(meta["call"]))
        if key in dumps:
            raise ValueError(f"Duplicate dump key {key} under {root}")
        dumps[key] = (record["value"], {**meta, "path": str(path)})
    return dumps


def _group_dumps(
    dumps: dict[tuple[str, str, int], tuple[torch.Tensor, dict]],
) -> dict[tuple[str, str], list[tuple[int, torch.Tensor, dict]]]:
    grouped: dict[tuple[str, str], list[tuple[int, torch.Tensor, dict]]] = {}
    for (name, phase, call), (tensor, metadata) in dumps.items():
        grouped.setdefault((name, phase), []).append((call, tensor, metadata))
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item[0])
    return grouped


def _align(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.squeeze()
    right = right.squeeze()
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return left.float(), right.float()


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    left, right = _align(left, right)
    # These tensors contain tens of millions of elements at 8K context. FP32
    # reductions can accumulate enough error to report an impossible cosine
    # above one, so use FP64 for the scalar diagnostics.
    left64 = left.double()
    right64 = right.double()
    diff64 = right64 - left64
    left_norm = float(torch.linalg.vector_norm(left64))
    right_norm = float(torch.linalg.vector_norm(right64))
    difference_l2 = float(torch.linalg.vector_norm(diff64))
    exact_count = int((left == right).sum())
    denominator = max(left_norm, torch.finfo(torch.float64).tiny)
    cosine_denominator = max(left_norm * right_norm, torch.finfo(torch.float64).tiny)
    cosine = float(torch.dot(left64.reshape(-1), right64.reshape(-1))) / cosine_denominator
    return {
        "max_abs": float(diff64.abs().max()),
        "mean_abs": float(diff64.abs().mean()),
        "rms_abs": float(diff64.square().mean().sqrt()),
        "rel_l2": difference_l2 / denominator,
        "cosine": min(1.0, max(-1.0, cosine)),
        "exact_fraction": exact_count / left.numel(),
        "reference_l2": left_norm,
        "difference_l2": difference_l2,
        "numel": left.numel(),
        "exact_count": exact_count,
    }


def _select_compatible_pair(
    left: list[tuple[int, torch.Tensor, dict]], right: list[tuple[int, torch.Tensor, dict]]
) -> tuple[tuple[int, torch.Tensor, dict], tuple[int, torch.Tensor, dict]] | None:
    """Select the largest compatible call, preferring later calls on ties.

    Rollout generation first runs an 8064-token prompt prefill. Log-probability
    recomputation then runs the longer scoring prefill that Megatron trains on.
    Calls are process-local, so their ordinal is not a cross-backend identity;
    canonical name, phase, and squeezed shape are.
    """

    compatible = []
    for left_item in left:
        for right_item in right:
            if left_item[1].squeeze().shape == right_item[1].squeeze().shape:
                compatible.append((left_item, right_item))
    if not compatible:
        return None
    return max(compatible, key=lambda pair: (pair[0][1].numel(), pair[0][0], pair[1][0]))


def compare(left_root: Path, right_root: Path) -> dict:
    left = _load_dumps(left_root)
    right = _load_dumps(right_root)
    if not left:
        raise RuntimeError(f"No SGLang tensor dumps found under {left_root}")
    if not right:
        raise RuntimeError(f"No Megatron tensor dumps found under {right_root}")
    left_grouped = _group_dumps(left)
    right_grouped = _group_dumps(right)
    common_keys = left_grouped.keys() & right_grouped.keys()
    common = sorted(common_keys)
    if not common:
        raise RuntimeError("SGLang and Megatron dumps have no common canonical tensor names")
    records = []
    used_left: set[tuple[str, str, int]] = set()
    used_right: set[tuple[str, str, int]] = set()
    for key in common:
        selected = _select_compatible_pair(left_grouped[key], right_grouped[key])
        if selected is None:
            left_call, left_tensor, left_meta = max(
                left_grouped[key], key=lambda item: (item[1].numel(), item[0])
            )
            right_call, right_tensor, right_meta = max(
                right_grouped[key], key=lambda item: (item[1].numel(), item[0])
            )
        else:
            (left_call, left_tensor, left_meta), (right_call, right_tensor, right_meta) = selected
        used_left.add((key[0], key[1], left_call))
        used_right.add((key[0], key[1], right_call))
        record = {
            "name": key[0],
            "phase": key[1],
            "left_call": left_call,
            "right_call": right_call,
            "left_shape": list(left_tensor.shape),
            "right_shape": list(right_tensor.shape),
            "left_path": left_meta["path"],
            "right_path": right_meta["path"],
        }
        if key[0].endswith(".qkv") and key[1] == "output":
            record["status"] = "layout_opaque"
            record["error"] = (
                "Packed QKV layouts are backend-specific; compare canonical Q, K, and V taps"
            )
            records.append(record)
            continue
        if key[0].endswith(".qkv") and (
            (key[0].removesuffix(".qkv") + ".input_rmsnorm", "output") not in common_keys
        ):
            record["status"] = "fused_opaque"
            record["error"] = (
                "Native TE QKV input is pre-RMSNorm while SGLang QKV input is post-RMSNorm"
            )
            records.append(record)
            continue
        if selected is None:
            record["status"] = "shape_mismatch"
            record["error"] = (
                "no compatible call shapes: "
                f"{[list(item[1].shape) for item in left_grouped[key]]} vs "
                f"{[list(item[1].shape) for item in right_grouped[key]]}"
            )
        else:
            record.update(_metrics(left_tensor, right_tensor))
            record["status"] = "compared"
        records.append(record)
    records.sort(
        key=lambda record: (
            TAP_ORDER.get((record["name"], record["phase"]), len(TAP_ORDER)),
            record["name"],
            record["phase"],
        )
    )
    compared = [record for record in records if record["status"] == "compared"]
    compared_output_names = {record["name"] for record in compared if record["phase"] == "output"}
    missing_expected = [name for name in EXPECTED_OUTPUT_TAPS if name not in compared_output_names]
    nonzero = [record for record in compared if record["max_abs"] > 0]
    worst = max(compared, key=lambda record: record["rel_l2"], default=None)
    summary = {
        "compared": len(compared),
        "shape_mismatches": sum(record["status"] == "shape_mismatch" for record in records),
        "missing_expected_output_taps": missing_expected,
        "first_nonzero": nonzero[0] if nonzero else None,
        "worst_relative_l2": worst,
    }
    report = {
        "left": str(left_root),
        "right": str(right_root),
        "left_only": [list(key) for key in sorted(left.keys() - used_left)],
        "right_only": [list(key) for key in sorted(right.keys() - used_right)],
        "records": records,
        "summary": summary,
    }
    report["operation_progression"] = build_operation_progression(records)
    return report


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# SGLang and Megatron intermediate comparison",
        "",
        f"- SGLang dumps: `{report['left']}`",
        f"- Megatron dumps: `{report['right']}`",
        "",
        "| Tensor | Phase | Calls | Shapes | Max abs | Mean abs | Relative L2 | Cosine | Exact |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        shapes = f"{record['left_shape']} / {record['right_shape']}"
        calls = f"{record['left_call']} / {record['right_call']}"
        if record["status"] == "compared":
            lines.append(
                f"| `{record['name']}` | {record['phase']} | `{calls}` | `{shapes}` | "
                f"{record['max_abs']:.7g} | {record['mean_abs']:.7g} | "
                f"{record['rel_l2']:.7g} | {record['cosine']:.7g} | "
                f"{record['exact_fraction']:.3%} |"
            )
        elif record["status"] == "shape_mismatch":
            lines.append(
                f"| `{record['name']}` | {record['phase']} | `{calls}` | `{shapes}` | "
                "shape mismatch | - | - | - | - |"
            )
        else:
            lines.append(
                f"| `{record['name']}` | {record['phase']} | `{calls}` | `{shapes}` | "
                f"{record['error']} | - | - | - | - |"
            )
    lines.extend(
        [
            "",
            f"Left-only tensors: {len(report['left_only'])}",
            "",
            f"Right-only tensors: {len(report['right_only'])}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare(args.sglang, args.megatron)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "intermediate_diff.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_markdown(report, args.output / "intermediate_diff.md")
    write_operation_progression(report, args.output)

    compared = [record for record in report["records"] if record["status"] == "compared"]
    worst = max(compared, key=lambda item: item["rel_l2"], default=None)
    if worst is not None:
        print(
            f"Compared {len(compared)} tensors; worst relative L2 "
            f"{worst['rel_l2']:.7g} at {worst['name']}:{worst['phase']}"
        )
    summary = report["summary"]
    if summary["missing_expected_output_taps"]:
        raise RuntimeError(
            "Missing expected output taps: " + ", ".join(summary["missing_expected_output_taps"])
        )
    if summary["shape_mismatches"]:
        raise RuntimeError(
            f"Found {summary['shape_mismatches']} canonical tensor shape mismatch(es)"
        )


if __name__ == "__main__":
    main()
