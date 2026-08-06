"""Run isolated BF16 non-MoE parity ablations on a two-layer Qwen3-30B-A3B.

The devbox reserves eight GPUs because that is the c1 scheduling unit, while
Ray intentionally exposes only two: one Megatron actor GPU and one SGLang
rollout GPU. Each variant starts from the same two-layer checkpoint.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import miles.utils.external_utils.command_utils as U
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_REPO = "fzyzcjy/Qwen3-30B-A3B-5layer"
MODEL_REVISION = "9c2ee37f22b7ef150675311b3d5e1c671838ffe1"
MEGATRON_IMAGE_REVISION = "4716f75475c78e2fc2c6f0d3af095f1681b770b4"
MEGATRON_PR_BASE_REVISION = "50ac48e87b8a31da7330de4a03d8ae42b985d9d2"
SGLANG_IMAGE_REVISION = "d218d6c7835307da50373f81704e61338b4e4847"
MILES_IMAGE_REVISION = "43d38ada230a431845338ed913f6c3a1b5f8355d"
SOURCE_MODEL_NAME = "Qwen3-30B-A3B-5layer"
MODEL_NAME = "Qwen3-30B-A3B-2layer"
MEGATRON_MODEL_TYPE = "qwen3-30B-A3B"
SOURCE_NUM_LAYERS = 5
NUM_LAYERS = 2
PROMPT_TOKENS = 8064
RESPONSE_TOKENS = 128
CONTEXT_TOKENS = PROMPT_TOKENS + RESPONSE_TOKENS
MODEL_MARKER = ".miles_two_layer_complete.json"
SOURCE_MARKER = ".miles_source_revision.json"
CONVERTED_MARKER = ".miles_source_model.json"


@dataclass(frozen=True)
class Variant:
    name: str
    kernels: str


VARIANTS = {
    "baseline": Variant("baseline", ""),
    "rmsnorm": Variant("rmsnorm", "rmsnorm"),
    "rmsnorm_qk": Variant("rmsnorm_qk", "rmsnorm,qk_rmsnorm"),
    "rmsnorm_qk_final": Variant("rmsnorm_qk_final", "rmsnorm,qk_rmsnorm,final_rmsnorm"),
}

DRIVER_MANAGED_ENV_VARS = (
    "MILES_SGLANG_BF16_KERNELS",
    "MILES_SGLANG_BF16_DEBUG_DIR",
    "SGLANG_BF16_NONMOE_PARITY",
    "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM",
)


def _run(command: list[str], **kwargs) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def _git_state(path: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head": head, "dirty": bool(status.strip())}


def _preflight_megatron_source(megatron_path: Path) -> dict[str, object]:
    import megatron.core

    from miles_megatron_plugins.sglang_bf16_kernels.selection import SUPPORTED_SGLANG_BF16_KERNELS

    source = Path(megatron.core.__file__).resolve()
    expected_root = megatron_path.resolve()
    if not source.is_relative_to(expected_root):
        raise RuntimeError(
            f"Imported Megatron from {source}, expected editable source under {expected_root}"
        )
    for revision, label in (
        (MEGATRON_IMAGE_REVISION, "image revision"),
        (MEGATRON_PR_BASE_REVISION, "PR base revision"),
    ):
        base_check = subprocess.run(
            ["git", "-C", str(megatron_path), "merge-base", "--is-ancestor", revision, "HEAD"],
            check=False,
        )
        if base_check.returncode != 0:
            raise RuntimeError(f"Megatron HEAD is not based on {label} {revision}")
    if SUPPORTED_SGLANG_BF16_KERNELS != ("rmsnorm", "qk_rmsnorm", "final_rmsnorm"):
        raise RuntimeError("Unexpected Megatron BF16 drop-in selection surface")
    return {
        "source": str(source),
        "image_base": MEGATRON_IMAGE_REVISION,
        "pr_base": MEGATRON_PR_BASE_REVISION,
        "supported_drop_ins": list(SUPPORTED_SGLANG_BF16_KERNELS),
        **_git_state(megatron_path),
    }


def _preflight_sglang_source(sglang_path: Path) -> dict[str, object]:
    import sglang
    from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm
    from sglang.srt.debug_utils.bf16_kernel_parity import BF16_NONMOE_PARITY_ENV

    source = Path(sglang.__file__).resolve()
    expected_root = (sglang_path / "python").resolve()
    if not source.is_relative_to(expected_root):
        raise RuntimeError(
            f"Imported SGLang from {source}, expected editable source under {expected_root}"
        )
    base_check = subprocess.run(
        [
            "git",
            "-C",
            str(sglang_path),
            "merge-base",
            "--is-ancestor",
            SGLANG_IMAGE_REVISION,
            "HEAD",
        ],
        check=False,
    )
    if base_check.returncode != 0:
        raise RuntimeError(f"SGLang HEAD is not based on image revision {SGLANG_IMAGE_REVISION}")
    return {
        "source": str(source),
        "image_base": SGLANG_IMAGE_REVISION,
        "rmsnorm_api": f"{true_on_policy_rms_norm.__module__}.{true_on_policy_rms_norm.__name__}",
        "nonmoe_parity_env": BF16_NONMOE_PARITY_ENV,
        **_git_state(sglang_path),
    }


def _preflight_miles_source(miles_path: Path) -> dict[str, object]:
    from miles.backends.megatron_utils.megatron_to_hf import qwen3moe

    source = Path(qwen3moe.__file__).resolve()
    expected_root = miles_path.resolve()
    if not source.is_relative_to(expected_root):
        raise RuntimeError(
            f"Imported Miles from {source}, expected editable source under {expected_root}"
        )
    base_check = subprocess.run(
        ["git", "-C", str(miles_path), "merge-base", "--is-ancestor", MILES_IMAGE_REVISION, "HEAD"],
        check=False,
    )
    if base_check.returncode != 0:
        raise RuntimeError(f"Miles HEAD is not based on image revision {MILES_IMAGE_REVISION}")

    parameter = object()
    converted = qwen3moe.convert_qwen3moe_to_hf(
        argparse.Namespace(hidden_size=4, kv_channels=2, num_attention_heads=2, num_query_groups=1),
        "module.module.decoder.layers.0.input_layernorm.weight",
        parameter,
    )
    if converted != [("model.layers.0.input_layernorm.weight", parameter)]:
        raise RuntimeError("Miles lacks explicit Qwen3-MoE input-layernorm weight sync support")
    return {
        "source": str(source),
        "image_base": MILES_IMAGE_REVISION,
        "explicit_input_layernorm_weight_sync": True,
        **_git_state(miles_path),
    }


def _copy_non_weight_files(source: Path, target: Path) -> None:
    for child in source.iterdir():
        if child.name == "config.json" or child.name == "model.safetensors.index.json":
            continue
        if child.suffix == ".safetensors":
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def _safetensors_total_size(model_dir: Path) -> int:
    total = 0
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                total += tensor.numel() * tensor.element_size()
    return total


def _expected_model_manifest() -> dict[str, object]:
    return {
        "source_repo": MODEL_REPO,
        "source_revision": MODEL_REVISION,
        "num_hidden_layers": NUM_LAYERS,
    }


def _expected_source_manifest() -> dict[str, object]:
    return {"source_repo": MODEL_REPO, "source_revision": MODEL_REVISION}


def _expected_converted_manifest() -> dict[str, object]:
    return {
        **_expected_model_manifest(),
        "megatron_model_type": MEGATRON_MODEL_TYPE,
        "megatron_pr_base": MEGATRON_PR_BASE_REVISION,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid parity checkpoint manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Parity checkpoint manifest must contain an object: {path}")
    return value


def _validate_truncated_model(target: Path) -> None:
    marker = target / MODEL_MARKER
    expected = _expected_model_manifest()
    if _read_manifest(marker) != expected:
        raise RuntimeError(
            f"Truncated checkpoint manifest does not match this experiment: {marker}"
        )

    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != NUM_LAYERS:
        raise RuntimeError(
            f"Expected {NUM_LAYERS} layers in {target / 'config.json'}, "
            f"got {config.get('num_hidden_layers')}"
        )
    index = json.loads((target / "model.safetensors.index.json").read_text(encoding="utf-8"))
    layer_ids = {
        int(match.group(1))
        for name in index["weight_map"]
        if (match := re.match(r"^model\.layers\.(\d+)\.", name)) is not None
    }
    if layer_ids != set(range(NUM_LAYERS)):
        raise RuntimeError(
            f"Expected exactly model layers {list(range(NUM_LAYERS))} in {target}, "
            f"got {sorted(layer_ids)}"
        )


def _validate_source_model(source: Path) -> None:
    if _read_manifest(source / SOURCE_MARKER) != _expected_source_manifest():
        raise RuntimeError(f"Source checkpoint revision is not verified: {source}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != SOURCE_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {SOURCE_NUM_LAYERS} source layers in {source}, "
            f"got {config.get('num_hidden_layers')}"
        )
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    layer_ids = {
        int(match.group(1))
        for name in index["weight_map"]
        if (match := re.match(r"^model\.layers\.(\d+)\.", name)) is not None
    }
    if layer_ids != set(range(SOURCE_NUM_LAYERS)):
        raise RuntimeError(
            f"Expected exactly source layers {list(range(SOURCE_NUM_LAYERS))} in {source}, "
            f"got {sorted(layer_ids)}"
        )


def _finalize_truncated_model(source: Path, target: Path) -> None:
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = NUM_LAYERS
    if "max_window_layers" in config:
        config["max_window_layers"] = NUM_LAYERS
    if "mlp_only_layers" in config:
        config["mlp_only_layers"] = [
            layer for layer in config["mlp_only_layers"] if layer < NUM_LAYERS
        ]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    _copy_non_weight_files(source, target)
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_size"] = _safetensors_total_size(target)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (target / MODEL_MARKER).write_text(
        json.dumps(_expected_model_manifest(), indent=2) + "\n", encoding="utf-8"
    )


def prepare_model(model_dir: Path, megatron_path: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    source = model_dir / SOURCE_MODEL_NAME
    target = model_dir / MODEL_NAME
    converted = model_dir / f"{MODEL_NAME}_torch_dist"

    source_marker = source / SOURCE_MARKER
    if not source_marker.exists() or _read_manifest(source_marker) != _expected_source_manifest():
        _run(
            ["hf", "download", MODEL_REPO, "--revision", MODEL_REVISION, "--local-dir", str(source)]
        )
        source_marker.write_text(
            json.dumps(_expected_source_manifest(), indent=2) + "\n", encoding="utf-8"
        )
    _validate_source_model(source)

    complete = target / MODEL_MARKER
    if not complete.exists():
        if target.exists():
            raise RuntimeError(
                f"Refusing to overwrite incomplete truncated checkpoint {target}; "
                "move it aside and retry"
            )
        temporary = Path(tempfile.mkdtemp(prefix=f".{MODEL_NAME}-", dir=model_dir))
        try:
            _run(
                [
                    "python",
                    "-m",
                    "sglang.srt.debug_utils.model_truncator",
                    "--input",
                    str(source),
                    "--output",
                    str(temporary),
                    "--keep-num-layers",
                    str(NUM_LAYERS),
                ]
            )
            _finalize_truncated_model(source, temporary)
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    _validate_truncated_model(target)

    os.environ["MODEL_ARGS_NUM_LAYERS"] = str(NUM_LAYERS)
    converted_tracker = converted / "latest_checkpointed_iteration.txt"
    converted_marker = converted / CONVERTED_MARKER
    if (
        converted_tracker.exists()
        and converted_tracker.read_text(encoding="utf-8").strip() == "release"
    ):
        if (
            not converted_marker.exists()
            or _read_manifest(converted_marker) != _expected_converted_manifest()
        ):
            raise RuntimeError(
                f"Refusing to reuse unverified converted checkpoint {converted}; move it aside and retry"
            )
    else:
        if converted.exists() and any(converted.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite incomplete converted checkpoint {converted}; "
                "move it aside and retry"
            )
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MEGATRON_MODEL_TYPE,
            num_gpus_per_node=1,
            dir_dst=str(model_dir),
            hf_checkpoint=str(target),
            megatron_path=str(megatron_path),
        )
        if (
            not converted_tracker.exists()
            or converted_tracker.read_text(encoding="utf-8").strip() != "release"
        ):
            raise RuntimeError(f"Megatron checkpoint conversion did not complete: {converted}")
        converted_marker.write_text(
            json.dumps(_expected_converted_manifest(), indent=2) + "\n", encoding="utf-8"
        )
    return target, converted


def _render_prompt_with_exact_tokens(tokenizer, target_tokens: int) -> str:
    prefix = (
        "Solve the following arithmetic problem. Give the final answer as a number. "
        "Compute 1 + 1. Before answering, silently check every intermediate step."
    )

    def render(repetitions: int) -> tuple[str, int]:
        content = prefix + " step" * repetitions
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, length

    low, high = 0, target_tokens
    while low <= high:
        middle = (low + high) // 2
        prompt, length = render(middle)
        if length == target_tokens:
            return prompt
        if length < target_tokens:
            low = middle + 1
        else:
            high = middle - 1

    for repetitions in range(max(0, high - 512), min(target_tokens, low + 512) + 1):
        prompt, length = render(repetitions)
        if length == target_tokens:
            return prompt
    raise RuntimeError(f"Could not render an exact {target_tokens}-token Qwen prompt")


def prepare_prompt(hf_checkpoint: Path, output_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    prompt = _render_prompt_with_exact_tokens(tokenizer, PROMPT_TOKENS)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_tokens != PROMPT_TOKENS:
        raise AssertionError(f"Expected {PROMPT_TOKENS} prompt tokens, got {actual_tokens}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"prompt": prompt, "label": "2"}, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _build_train_args(hf_checkpoint: Path, ref_load: Path, prompt_path: Path, run_dir: Path) -> str:
    checkpoint_args = f"--hf-checkpoint {hf_checkpoint} " f"--ref-load {ref_load} "
    rollout_args = (
        f"--prompt-data {prompt_path} "
        "--input-key prompt "
        "--label-key label "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 1 "
        "--n-samples-per-prompt 1 "
        "--global-batch-size 1 "
        f"--rollout-max-context-len {CONTEXT_TOKENS} "
        f"--rollout-max-prompt-len {PROMPT_TOKENS} "
        f"--rollout-max-response-len {RESPONSE_TOKENS} "
        "--rollout-temperature 1 "
        "--rollout-top-p 1 "
        "--rollout-top-k -1 "
        "--rollout-seed 42 "
        "--custom-generate-function-path "
        "miles_megatron_plugins.sglang_bf16_kernels.fixed_length_generate.generate_fixed_length "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    algorithm_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0 "
        "--kl-loss-coef 0 "
        "--entropy-coef 0 "
        "--eps-clip 0.2 "
        "--true-on-policy-mode "
        "--recompute-logprobs-via-prefill "
        "--use-rollout-routing-replay "
    )
    parallel_args = (
        "--bf16 "
        "--transformer-impl transformer_engine "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        f"--seq-length {CONTEXT_TOKENS} "
        "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {CONTEXT_TOKENS} "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-tp-size 1 "
        "--sglang-ep-size 1 "
        "--sglang-moe-runner-backend triton "
        "--sglang-attention-backend fa3 "
        "--sglang-kv-cache-dtype bf16 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-max-running-requests 1 "
        f"--sglang-chunked-prefill-size {CONTEXT_TOKENS} "
        "--sglang-disable-cuda-graph "
        "--sglang-disable-radix-cache "
        "--sglang-enable-deterministic-inference "
    )
    runtime_args = (
        "--attention-dropout 0 "
        "--hidden-dropout 0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--accumulate-allreduce-grads-in-fp32 "
        "--calculate-per-token-loss "
        "--deterministic-mode "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--num-gpus-per-node 2 "
        "--rollout-num-gpus 1 "
        "--update-weight-transfer-mode broadcast "
        f"--dump-details {run_dir / 'dump_details'} "
    )
    return "".join(
        (
            checkpoint_args,
            rollout_args,
            optimizer_args,
            algorithm_args,
            parallel_args,
            sglang_args,
            runtime_args,
        )
    )


def _read_metrics(metrics_dir: Path) -> dict[str, float]:
    wanted = {"train/train_rollout_logprob_abs_diff", "train/train_rollout_kl"}
    matches: dict[str, list[float]] = {metric: [] for metric in wanted}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("metric") in wanted and record.get("series"):
                value = record["series"][-1][1]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise RuntimeError(f"Non-finite metric marker for {record['metric']}: {value}")
                matches[record["metric"]].append(float(value))

    result = {}
    for metric, values in matches.items():
        if len(values) != 1:
            raise RuntimeError(
                f"Expected exactly one {metric} record under {metrics_dir}, found {len(values)}"
            )
        result[metric] = values[0]
    return result


def run_variant(
    variant: Variant,
    *,
    output_root: Path,
    hf_checkpoint: Path,
    ref_load: Path,
    prompt_path: Path,
    megatron_path: Path,
    sglang_path: Path,
    miles_path: Path,
) -> dict:
    run_dir = output_root / variant.name
    run_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = run_dir / "intermediates"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_args = _build_train_args(hf_checkpoint, ref_load, prompt_path, run_dir)
    extra_env_vars = {
        "MODEL_ARGS_NUM_LAYERS": str(NUM_LAYERS),
        "MILES_SGLANG_BF16_KERNELS": variant.kernels,
        "MILES_SGLANG_BF16_DEBUG_DIR": str(debug_dir),
        "MILES_SGLANG_BF16_DEBUG_MIN_TOKENS": str(PROMPT_TOKENS),
        "MILES_SGLANG_BF16_DEBUG_MAX_CALLS": "2",
        "MILES_CI_GATE_RECORD_DIR": str(metrics_dir),
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "PYTHONPATH": str(sglang_path / "python"),
        "SGLANG_BF16_NONMOE_PARITY": "1",
        "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM": "1",
        "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM_DEBUG": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "PYTHONUNBUFFERED": "1",
    }
    (run_dir / "launch.json").write_text(
        json.dumps(
            {
                "variant": variant.name,
                "kernels": variant.kernels,
                "train_args": train_args,
                "extra_env_vars": extra_env_vars,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        extra_env_vars=extra_env_vars,
        megatron_path=str(megatron_path),
    )

    comparison_dir = run_dir / "comparison"
    _run(
        [
            "python",
            "-m",
            "tools.sglang_bf16_parity.compare_intermediates",
            "--sglang",
            str(debug_dir / "sglang"),
            "--megatron",
            str(debug_dir / "megatron"),
            "--output",
            str(comparison_dir),
        ],
        cwd=megatron_path,
    )
    comparison = json.loads((comparison_dir / "intermediate_diff.json").read_text(encoding="utf-8"))

    result = {
        "variant": variant.name,
        "kernels": variant.kernels,
        "model_revision": MODEL_REVISION,
        "context_tokens": CONTEXT_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "response_tokens": RESPONSE_TOKENS,
        "megatron": _git_state(megatron_path),
        "sglang": _git_state(sglang_path),
        "miles": _git_state(miles_path),
        "metrics": _read_metrics(metrics_dir),
        "comparison_summary": comparison["summary"],
        "run_dir": str(run_dir),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"Comma-separated variants: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--run-id", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/root/models"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/root/shared_data/sglang-bf16-nonmoe")
    )
    parser.add_argument("--megatron-path", type=Path, default=Path("/root/Megatron-LM"))
    parser.add_argument("--sglang-path", type=Path, default=Path("/sgl-workspace/sglang"))
    parser.add_argument("--miles-path", type=Path, default=Path("/root/miles"))
    args = parser.parse_args()

    inherited = [name for name in DRIVER_MANAGED_ENV_VARS if os.environ.get(name)]
    if inherited:
        raise RuntimeError(
            "The harness manages these environment variables per variant; unset them first: "
            + ", ".join(inherited)
        )

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = set(names).difference(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")

    megatron_preflight = _preflight_megatron_source(args.megatron_path)
    sglang_preflight = _preflight_sglang_source(args.sglang_path)
    miles_preflight = _preflight_miles_source(args.miles_path)
    print(
        json.dumps(
            {
                "megatron_preflight": megatron_preflight,
                "sglang_preflight": sglang_preflight,
                "miles_preflight": miles_preflight,
            },
            indent=2,
        ),
        flush=True,
    )
    hf_checkpoint, ref_load = prepare_model(args.model_dir, args.megatron_path)
    prompt_path = args.output_root / "prompt_8064.jsonl"
    prepare_prompt(hf_checkpoint, prompt_path)
    if args.prepare_only:
        return

    run_root = args.output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "preflight.json").write_text(
        json.dumps(
            {"sglang": sglang_preflight, "miles": miles_preflight, "megatron": megatron_preflight},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results = []
    for name in names:
        results.append(
            run_variant(
                VARIANTS[name],
                output_root=run_root,
                hf_checkpoint=hf_checkpoint,
                ref_load=ref_load,
                prompt_path=prompt_path,
                megatron_path=args.megatron_path,
                sglang_path=args.sglang_path,
                miles_path=args.miles_path,
            )
        )
    baseline = next((result for result in results if result["variant"] == "baseline"), None)
    if baseline is not None:
        for result in results:
            result["metric_delta_from_baseline"] = {
                metric: value - baseline["metrics"][metric]
                for metric, value in result["metrics"].items()
            }
            (Path(result["run_dir"]) / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    (run_root / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Completed BF16 parity run: {run_root}", flush=True)


if __name__ == "__main__":
    main()
