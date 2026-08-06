"""Compare matched full-tensor dumps from SGLang and Megatron."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


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
TAP_ORDER = {name: index for index, name in enumerate(TAP_SEQUENCE)}

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
    dumps: dict[tuple[str, str, int], tuple[torch.Tensor, dict]]
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


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = _align(left, right)
    diff = right - left
    left_norm = torch.linalg.vector_norm(left)
    denominator = max(float(left_norm), torch.finfo(torch.float32).tiny)
    cosine = torch.nn.functional.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rms_abs": float(diff.square().mean().sqrt()),
        "rel_l2": float(torch.linalg.vector_norm(diff)) / denominator,
        "cosine": float(cosine),
        "exact_fraction": float((left == right).float().mean()),
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
    common = sorted(left_grouped.keys() & right_grouped.keys())
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
        if key[0].endswith(".qkv"):
            record["status"] = "layout_opaque"
            record["error"] = "Packed QKV layouts are backend-specific; compare canonical Q/K taps"
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
    compared = [record for record in records if record["status"] == "compared"]
    compared_output_names = {record["name"] for record in compared if record["phase"] == "output"}
    missing_expected = [name for name in EXPECTED_OUTPUT_TAPS if name not in compared_output_names]
    ordered = sorted(
        compared,
        key=lambda record: (
            TAP_ORDER.get(record["name"], len(TAP_ORDER)),
            0 if record["phase"] == "input" else 1,
            record["name"],
        ),
    )
    nonzero = [record for record in ordered if record["max_abs"] > 0]
    worst = max(compared, key=lambda record: record["rel_l2"], default=None)
    summary = {
        "compared": len(compared),
        "shape_mismatches": sum(record["status"] == "shape_mismatch" for record in records),
        "missing_expected_output_taps": missing_expected,
        "first_nonzero": nonzero[0] if nonzero else None,
        "worst_relative_l2": worst,
    }
    return {
        "left": str(left_root),
        "right": str(right_root),
        "left_only": [list(key) for key in sorted(left.keys() - used_left)],
        "right_only": [list(key) for key in sorted(right.keys() - used_right)],
        "records": records,
        "summary": summary,
    }


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
                "backend-specific layout | - | - | - | - |"
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
