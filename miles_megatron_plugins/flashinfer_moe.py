# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental FlashInfer routed-MoE forward with a BF16 surrogate backward.

This module deliberately targets the rollout routed-MoE contracts for:

* FlashInfer ``trtllm_fp4_block_scale_routed_moe`` with per-token NVFP4
* FlashInfer ``trtllm_fp8_block_scale_routed_moe`` with MXFP8
* contiguous expert-parallel expert ownership
* gated SwiGLU experts

Megatron BF16 parameters remain the checkpoint and optimizer source of truth.
TransformerEngine quantizes expert weights to match Miles weight sync, while
FlashInfer quantizes per-token activations and runs the fused MoE.
Backward recomputes a BF16 expert module and is intentionally not the
derivative of the quantized forward.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, get_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank


logger = logging.getLogger(__name__)

_ENV = "MILES_USE_FLASHINFER_MOE"
_QUANTIZATION_ENV = "MILES_FLASHINFER_MOE_QUANTIZATION"
_LOGGED_LAYERS: set[tuple[str, int]] = set()
_NVFP4_GROUP_SIZE = 16
_TE_NVFP4_ROW_ALIGNMENT = 16
_MXFP8_GROUP_SIZE = 32
_TE_MXFP8_ROW_ALIGNMENT = 32
_SUPPORTED_QUANTIZATIONS = ("nvfp4", "mxfp8")


def use_flashinfer_moe() -> bool:
    """Return whether the experimental routed-MoE path is enabled."""

    return os.environ.get(_ENV, "0") == "1"


def _flashinfer_moe_quantization(config) -> str:
    """Resolve one explicitly supported routed-MoE quantization."""

    override = os.environ.get(_QUANTIZATION_ENV, "").strip().lower()
    if override and override not in _SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported {_QUANTIZATION_ENV}={override!r}; supported values are "
            f"{', '.join(_SUPPORTED_QUANTIZATIONS)}"
        )

    configured = []
    fp8_recipe = getattr(config, "fp8_recipe", None)
    fp8_recipe = getattr(fp8_recipe, "value", fp8_recipe)
    if getattr(config, "fp8", None) is not None:
        if fp8_recipe != "mxfp8":
            raise ValueError(
                f"FlashInfer MoE does not support active FP8 recipe {fp8_recipe!r}; "
                "supported FP8 recipe: 'mxfp8'"
            )
        configured.append("mxfp8")

    fp4_recipe = getattr(config, "fp4_recipe", None)
    fp4_recipe = getattr(fp4_recipe, "value", fp4_recipe)
    if getattr(config, "fp4", None) is not None:
        if fp4_recipe != "nvfp4":
            raise ValueError(
                f"FlashInfer MoE does not support active FP4 recipe {fp4_recipe!r}; "
                "supported FP4 recipe: 'nvfp4'"
            )
        configured.append("nvfp4")

    if len(configured) > 1:
        raise ValueError(
            f"FlashInfer MoE requires exactly one quantization, got {configured}"
        )
    if override:
        if configured and configured[0] != override:
            raise ValueError(
                f"{_QUANTIZATION_ENV}={override!r} conflicts with active "
                f"{configured[0]!r} quantization"
            )
        return override
    if not configured:
        raise ValueError(
            "FlashInfer MoE requires an explicit supported quantization: active "
            "MXFP8/NVFP4 config or MILES_FLASHINFER_MOE_QUANTIZATION"
        )
    return configured[0]


class _FlashInferGroupedLinearParameters(torch.nn.Module):
    """Per-expert BF16 parameters with TEGroupedLinear-compatible names."""

    def __init__(
        self,
        *,
        num_experts,
        input_size,
        output_size,
        parallel_mode,
        config,
        init_method,
        pg_collection,
    ):
        super().__init__()
        if config.expert_tensor_parallel_size != 1:
            raise ValueError("FlashInfer MoE currently requires expert tensor parallel size 1")

        self.num_experts = num_experts
        self.parallel_mode = parallel_mode
        self._pg_collection = pg_collection
        self._tp_group = pg_collection.expt_tp
        partition_dim = 0 if parallel_mode == "column" else 1
        shape = (output_size, input_size)
        device = None if config.use_cpu_initialization else torch.cuda.current_device()

        for expert in range(num_experts):
            weight = Parameter(
                torch.empty(
                    *shape,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                if config.use_cpu_initialization:
                    _initialize_affine_weight_cpu(
                        weight,
                        output_size,
                        input_size,
                        shape[partition_dim],
                        partition_dim=partition_dim,
                        init_method=init_method,
                        params_dtype=config.params_dtype,
                        rank=0,
                        world_size=1,
                    )
                else:
                    _initialize_affine_weight_gpu(
                        weight,
                        init_method,
                        partition_dim=partition_dim,
                        is_expert=True,
                    )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=weight,
                    is_parallel=True,
                    dim=partition_dim,
                    stride=1,
                )
            setattr(weight, "allreduce", not (config.expert_model_parallel_size > 1))
            self.register_parameter(f"weight{expert}", weight)

    def sharded_state_dict(
        self, prefix="", sharded_offsets=(), metadata=None
    ):
        """Mirror TEGroupedLinear's per-expert EP/ETP checkpoint mapping."""

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = metadata.get("singleton_local_shards", False)
        num_global_experts = get_pg_size(self._pg_collection.ep) * self.num_experts
        local_expert_offset = get_pg_rank(self._pg_collection.ep) * self.num_experts
        ep_axis = len(sharded_offsets)
        tp_axis = 0 if self.parallel_mode == "column" else 1
        sharded_state_dict = {}

        for local_expert in range(self.num_experts):
            global_expert = local_expert_offset + local_expert
            state_dict = {
                f"{local_expert}.weight": getattr(
                    self, f"weight{local_expert}"
                ),
                f"{local_expert}._extra_state": None,
            }
            if singleton_local_shards:
                expert_prefix = f"{global_expert}.{prefix}"
                expert_offsets = sharded_offsets
            else:
                expert_prefix = prefix
                expert_offsets = (
                    *sharded_offsets,
                    (ep_axis, global_expert, num_global_experts),
                )
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict,
                "",
                {f"{local_expert}.weight": tp_axis},
                expert_offsets,
                tp_group=self._tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
            replace_prefix_for_sharding(
                sub_sd, f"{local_expert}.", expert_prefix
            )
            sharded_state_dict[f"{prefix}weight{local_expert}"] = sub_sd[
                f"{local_expert}.weight"
            ]
            extra_state_suffix = "" if local_expert == 0 else str(local_expert)
            sharded_state_dict[f"{prefix}_extra_state{extra_state_suffix}"] = (
                sub_sd[f"{local_expert}._extra_state"]
            )

        for sharded_value in sharded_state_dict.values():
            replica_id = sharded_value.replica_id
            if getattr(sharded_value, "is_data_parallel_fully_shard", False):
                expert_dp_rank = 0
            else:
                expert_dp_rank = get_pg_rank(self._pg_collection.expt_dp)
            sharded_value.replica_id = (*replica_id[:2], expert_dp_rank)
        return sharded_state_dict


class FlashInferGroupedMLP(MegatronModule):
    """TE-compatible BF16 master parameters without TE or grouped GEMM."""

    def __init__(self, num_local_experts, config, pg_collection=None):
        super().__init__(config=config)
        self.config = config
        self.num_local_experts = num_local_experts
        if pg_collection is None:
            raise ValueError("FlashInferGroupedMLP requires a ProcessGroupCollection")
        if config.add_bias_linear:
            raise ValueError("FlashInferGroupedMLP does not support expert bias")
        if config.moe_latent_size is not None:
            raise ValueError("FlashInferGroupedMLP does not support latent MoE projections")

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp
        fc1_output_size = config.moe_ffn_hidden_size
        if config.gated_linear_unit:
            fc1_output_size *= 2
        self.linear_fc1 = _FlashInferGroupedLinearParameters(
            num_experts=num_local_experts,
            input_size=config.hidden_size,
            output_size=fc1_output_size,
            parallel_mode="column",
            config=config,
            init_method=config.init_method,
            pg_collection=pg_collection,
        )
        self.linear_fc2 = _FlashInferGroupedLinearParameters(
            num_experts=num_local_experts,
            input_size=config.moe_ffn_hidden_size,
            output_size=config.hidden_size,
            parallel_mode="row",
            config=config,
            init_method=config.output_layer_init_method,
            pg_collection=pg_collection,
        )

        def remove_extra_states_check(_module, incompatible_keys):
            for key in list(incompatible_keys.unexpected_keys):
                if "_extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(self, *_args, **_kwargs):
        raise RuntimeError(
            "FlashInferGroupedMLP only owns BF16 master parameters; execution must "
            "go through the direct FlashInfer MoE path"
        )

    def backward_dw(self):
        """No delayed TE weight-gradient work exists for plain BF16 parameters."""

    def sharded_state_dict(
        self, prefix="", sharded_offsets=(), metadata=None
    ):
        """Match TEGroupedMLP's global expert checkpoint keys."""

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = metadata.get("singleton_local_shards", False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module,
                f"{name}.",
                sharded_offsets,
                metadata,
                tp_group=self.tp_group,
            )
            if name == "linear_fc1" and self.config.gated_linear_unit:
                num_global_experts = (
                    get_pg_size(self.ep_group) * self.num_local_experts
                )
                local_expert_offset = (
                    get_pg_rank(self.ep_group) * self.num_local_experts
                )
                ep_axis = len(sharded_offsets)
                for local_expert in range(self.num_local_experts):
                    if singleton_local_shards:
                        expert_offsets = sharded_offsets
                    else:
                        expert_offsets = (
                            *sharded_offsets,
                            (
                                ep_axis,
                                local_expert_offset + local_expert,
                                num_global_experts,
                            ),
                        )
                    key = f"{name}.weight{local_expert}"
                    sub_sd[key] = apply_swiglu_sharded_factory(
                        sub_sd[key], expert_offsets, singleton_local_shards
                    )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, "", f"{prefix}experts.")
            else:
                replace_prefix_for_sharding(
                    sub_sd, f"{name}.", f"{prefix}experts.{name}."
                )
            sharded_state_dict.update(
                {f"{prefix}{key}": value for key, value in sub_sd.items()}
            )
        return sharded_state_dict


def maybe_replace_flashinfer_moe_expert_spec(submodules):
    """Select non-TE BF16 expert parameters when the FlashInfer path is enabled."""

    if not use_flashinfer_moe():
        return submodules
    if submodules is None or submodules.experts is None:
        raise ValueError(f"{_ENV}=1 requires an MoE expert module specification")

    expert_module = get_module(submodules.experts)
    if expert_module is FlashInferGroupedMLP:
        return submodules

    replacement = copy.copy(submodules)
    replacement.experts = ModuleSpec(module=FlashInferGroupedMLP)
    return replacement


def _topk_from_dense_routing(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    top_k: int,
    ordered_topk_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover precomputed routed-kernel inputs from Megatron's dense router output."""

    if probs.ndim != 2 or routing_map.shape != probs.shape:
        raise ValueError(
            "FlashInfer MoE expects dense [tokens, experts] probabilities and routing map, "
            f"got {tuple(probs.shape)} and {tuple(routing_map.shape)}"
        )
    if routing_map.dtype != torch.bool:
        routing_map = routing_map.bool()
    selected_per_token = routing_map.sum(dim=-1)
    if not torch.all(selected_per_token == top_k):
        raise ValueError(
            "FlashInfer MoE does not support token dropping or padded routing; "
            f"expected {top_k} selected experts per token"
        )

    if ordered_topk_ids is not None:
        ordered_topk_ids = ordered_topk_ids.to(
            device=probs.device, dtype=torch.long, non_blocking=True
        )
        if ordered_topk_ids.shape != (probs.shape[0], top_k):
            raise ValueError(
                "Rollout-routing-replay IDs do not match the local routed tokens: "
                f"got {tuple(ordered_topk_ids.shape)}, expected {(probs.shape[0], top_k)}"
            )
        # Miles stores padding rows as all -1, then replaces them with a valid
        # deterministic range inside ReplayManager before building routing_map.
        # Apply the same replacement to the saved stream whose slot order we
        # recover here.
        all_invalid = (ordered_topk_ids == -1).all(dim=-1)
        if torch.any(all_invalid):
            padding_ids = (
                torch.arange(top_k, device=probs.device, dtype=torch.long)
                % probs.shape[1]
            )
            ordered_topk_ids = torch.where(
                all_invalid.unsqueeze(-1), padding_ids, ordered_topk_ids
            )
        if torch.any(ordered_topk_ids < 0) or torch.any(
            ordered_topk_ids >= probs.shape[1]
        ):
            raise ValueError("FlashInfer MoE found invalid rollout-routing-replay expert IDs")
        if not torch.all(routing_map.gather(1, ordered_topk_ids)):
            raise ValueError(
                "Rollout-routing-replay IDs disagree with Megatron's dense routing map"
            )
        topk_weights = probs.gather(1, ordered_topk_ids)
        return (
            topk_weights.contiguous(),
            ordered_topk_ids.to(torch.int32).contiguous(),
        )

    masked_probs = probs.masked_fill(~routing_map, torch.finfo(probs.dtype).min)
    topk_weights, topk_ids = torch.topk(masked_probs, k=top_k, dim=-1, sorted=True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def _rollout_replay_topk_ids(moe_layer) -> Optional[torch.Tensor]:
    """Return the exact top-k slot order consumed by Miles routing replay."""

    try:
        from miles.utils.replay_base import routing_replay_manager
    except ImportError:
        return None

    if not routing_replay_manager.enabled:
        return None
    replay = getattr(moe_layer.router, "routing_replay", None)
    if replay is None:
        raise RuntimeError(
            "Miles rollout-routing-replay was enabled after model construction; "
            "the FlashInfer MoE router has no registered replay stream"
        )

    if routing_replay_manager.stage == "replay_forward":
        index = replay.forward_index - 1
    elif routing_replay_manager.stage == "replay_backward":
        index = replay.backward_index - 1
    elif routing_replay_manager.stage == "record":
        index = len(replay.top_indices_list) - 1
    else:
        return None
    if index < 0 or index >= len(replay.top_indices_list):
        raise RuntimeError(
            "Miles rollout-routing-replay did not produce top-k IDs for the current MoE layer"
        )
    return replay.top_indices_list[index]


def _pack_topk_ids(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Mirror SGLang's ``PackTopkIds`` packed BF16 routing representation."""

    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    return ((topk_ids.to(torch.int32) << 16) | weight_bits).contiguous()


def _grouped_mlp_weights(
    experts: FlashInferGroupedMLP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Megatron-order BF16 weights as [E, 2I, H] and [E, H, I]."""

    w13 = torch.stack(
        [
            getattr(experts.linear_fc1, f"weight{expert}")
            for expert in range(experts.num_local_experts)
        ]
    )
    w2 = torch.stack(
        [
            getattr(experts.linear_fc2, f"weight{expert}")
            for expert in range(experts.num_local_experts)
        ]
    )
    return w13.contiguous(), w2.contiguous()


def _source_weight_key(experts: FlashInferGroupedMLP) -> tuple[int, ...]:
    """Best-effort invalidation key for optimizer-updated BF16 master weights."""

    key = []
    for linear in (experts.linear_fc1, experts.linear_fc2):
        for expert in range(experts.num_local_experts):
            weight = getattr(linear, f"weight{expert}")
            key.extend((weight.untyped_storage().data_ptr(), weight._version))
    return tuple(key)


class _PaddedEPAllGather(torch.autograd.Function):
    """Padded EP all-gather whose backward is a summing reduce-scatter."""

    @staticmethod
    def forward(ctx, local_tensor, max_tokens, token_counts, ep_group, ep_rank, ep_size):
        ctx.max_tokens = max_tokens
        ctx.token_counts = token_counts
        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size

        padded = local_tensor.new_zeros((max_tokens, *local_tensor.shape[1:]))
        padded[: local_tensor.shape[0]].copy_(local_tensor)
        gathered = [torch.empty_like(padded) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, padded, group=ep_group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunks = grad_output.contiguous().view(
            ctx.ep_size, ctx.max_tokens, *grad_output.shape[1:]
        )
        grad_local_padded = torch.empty_like(chunks[ctx.ep_rank])
        torch.distributed.reduce_scatter(
            grad_local_padded,
            [chunk.contiguous() for chunk in chunks.unbind(0)],
            group=ctx.ep_group,
        )
        local_tokens = ctx.token_counts[ctx.ep_rank]
        return grad_local_padded[:local_tokens], None, None, None, None, None


class _EPAllReduceSum(torch.autograd.Function):
    """EP sum for local-expert partials, with the same sum in backward."""

    @staticmethod
    def forward(ctx, local_output, ep_group):
        ctx.ep_group = ep_group
        output = local_output.clone()
        torch.distributed.all_reduce(output, group=ep_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad_input, group=ctx.ep_group)
        return grad_input, None


def _all_gather_token_counts(
    local_tokens: int, device: torch.device, ep_group, ep_size: int
) -> tuple[int, tuple[int, ...]]:
    local_count = torch.tensor([local_tokens], dtype=torch.int64, device=device)
    gathered_counts = [torch.empty_like(local_count) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered_counts, local_count, group=ep_group)
    counts = tuple(int(count.item()) for count in gathered_counts)
    return max(counts), counts


def _all_gather_padded_no_grad(
    local_tensor: torch.Tensor, max_tokens: int, ep_group, ep_size: int
) -> torch.Tensor:
    padded = local_tensor.new_zeros((max_tokens, *local_tensor.shape[1:]))
    padded[: local_tensor.shape[0]].copy_(local_tensor)
    gathered = [torch.empty_like(padded) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, padded, group=ep_group)
    return torch.cat(gathered, dim=0)


def _bf16_local_routed_experts(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_gate_up: torch.Tensor,
    w2: torch.Tensor,
    local_expert_offset: int,
) -> torch.Tensor:
    """High-precision surrogate for this rank's local routed experts."""

    output = hidden_states.new_zeros(hidden_states.shape)
    # Keep every custom-autograd input connected even if this EP rank owns no
    # selected experts. This makes every rank enter the matching backward
    # collectives with explicit zero gradients instead of returning ``None``.
    output = output + (
        hidden_states.sum()
        + topk_weights.sum()
        + w13_gate_up.sum()
        + w2.sum()
    ) * 0

    for local_expert in range(w13_gate_up.shape[0]):
        global_expert = local_expert_offset + local_expert
        token_indices, slots = torch.where(topk_ids == global_expert)
        if token_indices.numel() == 0:
            continue

        fc1 = F.linear(hidden_states[token_indices], w13_gate_up[local_expert])
        gate, up = fc1.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        # Megatron applies routing probabilities before FC2 and rounds back to BF16.
        activated = (
            activated * topk_weights[token_indices, slots].unsqueeze(-1)
        ).to(activated.dtype)
        expert_output = F.linear(activated, w2[local_expert])
        output = output.index_add(0, token_indices, expert_output)

    return output


@dataclass
class _PreparedWeights:
    gemm1_weights: torch.Tensor
    gemm1_scales: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_scales: torch.Tensor
    output1_scale: torch.Tensor
    output1_gate_scale: torch.Tensor
    output2_scale: torch.Tensor


def _te_nvfp4_weight_e4m3_max() -> int:
    use_4over6 = os.environ.get("NVTE_NVFP4_4OVER6", "").strip().lower()
    use_256 = os.environ.get(
        "NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"
    ).strip().lower()
    if use_4over6 in ("weights", "all") and use_256 in ("weights", "all"):
        return 256
    return 448


def _te_nvfp4_global_decode_scale(
    global_amax: torch.Tensor, e4m3_max: int
) -> torch.Tensor:
    encode_scale = torch.div(
        torch.tensor(
            float(e4m3_max * 6),
            device=global_amax.device,
            dtype=torch.float32,
        ),
        global_amax.to(torch.float32),
    )
    encode_scale = torch.minimum(
        encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_amax.device,
            dtype=torch.float32,
        ),
    )
    encode_scale = torch.where(
        encode_scale == 0.0, torch.ones_like(encode_scale), encode_scale
    )
    return torch.reciprocal(encode_scale)


def _te_nvfp4_quantize_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one expert matrix with Miles' TE weight-sync contract."""

    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    pad_rows = (-num_rows) % _TE_NVFP4_ROW_ALIGNMENT
    if pad_rows:
        weight = torch.cat(
            (
                weight,
                torch.zeros(
                    (pad_rows, num_cols),
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            ),
            dim=0,
        )

    use_4over6 = (
        os.environ.get("NVTE_NVFP4_4OVER6", "").strip().lower()
        in ("weights", "all")
    )
    e4m3_max = _te_nvfp4_weight_e4m3_max()
    err_mode = (
        os.environ.get("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE")
        .strip()
        .upper()
    )
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
        with_random_sign_mask=False,
    )
    quantized = quantizer.quantize(weight)
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[
        :num_rows, : num_cols // _NVFP4_GROUP_SIZE
    ].contiguous()
    global_amax = quantized._amax_rowwise.reshape(-1)[0]
    return (
        qweight,
        block_scale.view(torch.float8_e4m3fn),
        _te_nvfp4_global_decode_scale(global_amax, e4m3_max),
    )


def _te_nvfp4_quantize_gated_weight(
    gate_up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize shared-scale gate/up rows, then adapt to FlashInfer order."""

    qweight, block_scale, global_scale = _te_nvfp4_quantize_weight(
        gate_up_weight
    )
    gate_qweight, up_qweight = qweight.chunk(2, dim=0)
    gate_block_scale, up_block_scale = block_scale.chunk(2, dim=0)
    return (
        torch.cat((up_qweight, gate_qweight), dim=0),
        torch.cat((up_block_scale, gate_block_scale), dim=0),
        global_scale,
    )


@dataclass
class _PreparedMXFP8Weights:
    gemm1_weights: torch.Tensor
    gemm1_scales: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_scales: torch.Tensor


def _te_mxfp8_quantize_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one expert matrix with Miles' rowwise MXFP8 contract."""

    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.constants import TE_DType

    if weight.ndim != 2:
        raise ValueError(f"MXFP8 expert weight must be 2D, got {tuple(weight.shape)}")
    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    if num_cols % _MXFP8_GROUP_SIZE:
        raise ValueError(
            f"MXFP8 expert K={num_cols} must be divisible by {_MXFP8_GROUP_SIZE}"
        )
    pad_rows = (-num_rows) % _TE_MXFP8_ROW_ALIGNMENT
    if pad_rows:
        weight = torch.cat(
            (
                weight,
                torch.zeros(
                    (pad_rows, num_cols),
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            ),
            dim=0,
        )

    quantizer = MXFP8Quantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
    )
    quantized = quantizer.quantize(weight)
    qweight = quantized._rowwise_data[:num_rows, :num_cols]
    qweight = qweight.contiguous().view(torch.float8_e4m3fn)
    scale = quantized._rowwise_scale_inv[
        :num_rows, : num_cols // _MXFP8_GROUP_SIZE
    ].contiguous()
    return qweight, scale.view(torch.uint8)


def _te_mxfp8_quantize_gated_weight(
    gate_up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Megatron [gate, up] rows and adapt them to TRT-LLM [up, gate]."""

    qweight, scale = _te_mxfp8_quantize_weight(gate_up_weight)
    gate_qweight, up_qweight = qweight.chunk(2, dim=0)
    gate_scale, up_scale = scale.chunk(2, dim=0)
    return (
        torch.cat((up_qweight, gate_qweight), dim=0),
        torch.cat((up_scale, gate_scale), dim=0),
    )


class _FlashInferRunnerBase:
    """Common layer-local metadata and nonpersistent quantized-weight cache."""

    def __init__(
        self,
        *,
        num_experts: int,
        local_expert_offset: int,
        local_num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.num_experts = num_experts
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._weight_key: Optional[tuple[int, ...]] = None
        self._prepared: Optional[object] = None
        self._permute_cache: dict = {}

    def invalidate_weights(self) -> None:
        """Drop the forward mirror before the next optimizer-updated iteration."""

        self._weight_key = None
        self._prepared = None


class _FlashInferNVFP4Runner(_FlashInferRunnerBase):
    """NVFP4 exact-forward adapter."""

    quantization = "nvfp4"

    @staticmethod
    def _e4m3_max() -> float:
        return (
            256.0
            if os.environ.get("FLASHINFER_NVFP4_4OVER6") == "1"
            and os.environ.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256") == "1"
            else 448.0
        )

    def _prepare_weights(
        self, w13_gate_up: torch.Tensor, w2: torch.Tensor, weight_key: tuple[int, ...]
    ) -> _PreparedWeights:
        if self._prepared is not None and self._weight_key == weight_key:
            return self._prepared

        from flashinfer import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        gemm1_weights = []
        gemm1_scales = []
        gemm2_weights = []
        gemm2_scales = []
        output1_scales = []
        output2_scales = []
        epilogue_tile_m = 128

        for expert in range(self.local_num_experts):
            # Quantize Megatron's shared-scale [gate, up] matrix before adapting
            # its emitted rows to the TRT-LLM gated kernel's [up, gate] order.
            w13_q, w13_sf, w13_decode = _te_nvfp4_quantize_gated_weight(
                w13_gate_up[expert]
            )
            w2_q, w2_sf, w2_decode = _te_nvfp4_quantize_weight(w2[expert])
            w13_q = w13_q.reshape(
                2 * self.intermediate_size, self.hidden_size // 2
            ).view(torch.uint8)
            w13_sf = w13_sf.view(torch.float8_e4m3fn).reshape(
                2 * self.intermediate_size, self.hidden_size // 16
            )
            w2_q = w2_q.reshape(
                self.hidden_size, self.intermediate_size // 2
            ).view(torch.uint8)
            w2_sf = w2_sf.view(torch.float8_e4m3fn).reshape(
                self.hidden_size, self.intermediate_size // 16
            )

            weight_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache,
                w13_q,
                epilogue_tile_m,
                is_gated_act_gemm=True,
            ).to(w13_q.device)
            scale_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache,
                w13_sf.view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
                is_gated_act_gemm=True,
            ).to(w13_sf.device)
            gemm1_weights.append(w13_q[weight_indices].contiguous())
            gemm1_scales.append(
                nvfp4_block_scale_interleave(
                    w13_sf.view(torch.uint8)[scale_indices].contiguous()
                )
            )

            weight_indices = get_w2_permute_indices_with_cache(
                self._permute_cache, w2_q, epilogue_tile_m
            ).to(w2_q.device)
            scale_indices = get_w2_permute_indices_with_cache(
                self._permute_cache,
                w2_sf.view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            ).to(w2_sf.device)
            gemm2_weights.append(w2_q[weight_indices].contiguous())
            gemm2_scales.append(
                nvfp4_block_scale_interleave(
                    w2_sf.view(torch.uint8)[scale_indices].contiguous()
                )
            )
            output1_scales.append(w13_decode)
            output2_scales.append(w2_decode)

        output1_scale = torch.stack(output1_scales).to(torch.float32)
        prepared = _PreparedWeights(
            gemm1_weights=torch.stack(gemm1_weights),
            gemm1_scales=torch.stack(gemm1_scales)
            .view(torch.float8_e4m3fn)
            .reshape(
                self.local_num_experts,
                2 * self.intermediate_size,
                self.hidden_size // 16,
            ),
            gemm2_weights=torch.stack(gemm2_weights),
            gemm2_scales=torch.stack(gemm2_scales)
            .view(torch.float8_e4m3fn)
            .reshape(
                self.local_num_experts,
                self.hidden_size,
                self.intermediate_size // 16,
            ),
            output1_scale=output1_scale,
            output1_gate_scale=output1_scale.clone(),
            output2_scale=torch.stack(output2_scales).to(torch.float32),
        )
        self._weight_key = weight_key
        self._prepared = prepared
        if os.environ.get("MILES_FLASHINFER_MOE_DEBUG") == "1":
            logger.warning(
                "FlashInfer MoE materialized NVFP4 weights for experts [%d, %d)",
                self.local_expert_offset,
                self.local_expert_offset + self.local_num_experts,
            )
        return prepared

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_gate_up: torch.Tensor,
        w2: torch.Tensor,
        weight_key: tuple[int, ...],
    ) -> torch.Tensor:
        from flashinfer import ActivationType, SfLayout, nvfp4_quantize
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.utils import device_support_pdl

        prepared = self._prepare_weights(w13_gate_up, w2, weight_key)
        input_global_scale = torch.full(
            (1,),
            1.0 / (self._e4m3_max() * 6.0),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        hidden_fp4, hidden_scales, per_token_scale = nvfp4_quantize(
            hidden_states.contiguous(),
            input_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cuda",
        )
        hidden_fp4 = hidden_fp4.reshape(hidden_states.shape[0], self.hidden_size // 2)
        hidden_scales = hidden_scales.view(torch.float8_e4m3fn).reshape(
            hidden_states.shape[0], self.hidden_size // 16
        )
        packed_topk = _pack_topk_ids(topk_ids, topk_weights)
        tune_max_tokens = 1 << max(hidden_states.shape[0] - 1, 0).bit_length()

        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=hidden_fp4,
            hidden_states_scale=hidden_scales,
            gemm1_weights=prepared.gemm1_weights,
            gemm1_weights_scale=prepared.gemm1_scales,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=prepared.gemm2_weights,
            gemm2_weights_scale=prepared.gemm2_scales,
            gemm2_bias=None,
            output1_scale_scalar=prepared.output1_scale,
            output1_scale_gate_scalar=prepared.output1_gate_scale,
            output2_scale_scalar=prepared.output2_scale,
            per_token_scale=per_token_scale,
            num_experts=self.num_experts,
            top_k=topk_ids.shape[1],
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=tune_max_tokens,
            enable_pdl=hidden_states.shape[0] <= 8192
            and device_support_pdl(hidden_states.device),
        )[0]


class _FlashInferMXFP8Runner(_FlashInferRunnerBase):
    """MXFP8 exact-forward adapter matching Miles and SGLang layouts."""

    quantization = "mxfp8"

    def _prepare_weights(
        self, w13_gate_up: torch.Tensor, w2: torch.Tensor, weight_key: tuple[int, ...]
    ) -> _PreparedMXFP8Weights:
        if self._prepared is not None and self._weight_key == weight_key:
            return self._prepared

        from flashinfer import block_scale_interleave
        from flashinfer.fused_moe.core import (
            get_reorder_rows_for_gated_act_gemm_row_indices,
        )
        from flashinfer.utils import (
            get_shuffle_matrix_a_row_indices,
            get_shuffle_matrix_sf_a_row_indices,
        )

        gemm1_weights = []
        gemm1_scales = []
        gemm2_weights = []
        gemm2_scales = []
        epilogue_tile_m = 128

        for expert in range(self.local_num_experts):
            # Miles owns Megatron [gate, up] masters. FlashInfer consumes W3/W1
            # [up, gate] before its gated-row interleave and row shuffle.
            w13_q, w13_sf = _te_mxfp8_quantize_gated_weight(
                w13_gate_up[expert]
            )
            w2_q, w2_sf = _te_mxfp8_quantize_weight(w2[expert])
            w13_u8 = w13_q.reshape(
                2 * self.intermediate_size, self.hidden_size
            ).view(torch.uint8)
            w13_sf = w13_sf.reshape(
                2 * self.intermediate_size, self.hidden_size // _MXFP8_GROUP_SIZE
            )
            w2_u8 = w2_q.reshape(
                self.hidden_size, self.intermediate_size
            ).view(torch.uint8)
            w2_sf = w2_sf.reshape(
                self.hidden_size, self.intermediate_size // _MXFP8_GROUP_SIZE
            )

            cache_key = (
                tuple(w13_u8.shape),
                tuple(w2_u8.shape),
                tuple(w13_sf.shape),
                tuple(w2_sf.shape),
                w13_u8.device,
                epilogue_tile_m,
            )
            indices = self._permute_cache.get(cache_key)
            if indices is None:
                indices = (
                    get_reorder_rows_for_gated_act_gemm_row_indices(w13_u8).to(
                        w13_u8.device
                    ),
                    get_shuffle_matrix_a_row_indices(
                        w13_u8, epilogue_tile_m
                    ).to(w13_u8.device),
                    get_shuffle_matrix_a_row_indices(w2_u8, epilogue_tile_m).to(
                        w2_u8.device
                    ),
                    get_shuffle_matrix_sf_a_row_indices(
                        w13_sf, epilogue_tile_m
                    ).to(w13_sf.device),
                    get_shuffle_matrix_sf_a_row_indices(
                        w2_sf, epilogue_tile_m
                    ).to(w2_sf.device),
                )
                self._permute_cache[cache_key] = indices
            (
                gated_rows,
                w13_weight_rows,
                w2_weight_rows,
                w13_scale_rows,
                w2_scale_rows,
            ) = indices

            w13_u8 = w13_u8.index_select(0, gated_rows)
            w13_sf = w13_sf.index_select(0, gated_rows)
            gemm1_weights.append(
                w13_u8.index_select(0, w13_weight_rows).contiguous()
            )
            gemm1_scales.append(
                block_scale_interleave(
                    w13_sf.index_select(0, w13_scale_rows).contiguous()
                ).reshape_as(w13_sf)
            )
            gemm2_weights.append(
                w2_u8.index_select(0, w2_weight_rows).contiguous()
            )
            gemm2_scales.append(
                block_scale_interleave(
                    w2_sf.index_select(0, w2_scale_rows).contiguous()
                ).reshape_as(w2_sf)
            )

        prepared = _PreparedMXFP8Weights(
            gemm1_weights=torch.stack(gemm1_weights).view(torch.float8_e4m3fn),
            gemm1_scales=torch.stack(gemm1_scales).view(torch.uint8),
            gemm2_weights=torch.stack(gemm2_weights).view(torch.float8_e4m3fn),
            gemm2_scales=torch.stack(gemm2_scales).view(torch.uint8),
        )
        self._weight_key = weight_key
        self._prepared = prepared
        if os.environ.get("MILES_FLASHINFER_MOE_DEBUG") == "1":
            logger.warning(
                "FlashInfer MoE materialized MXFP8 weights for experts [%d, %d)",
                self.local_expert_offset,
                self.local_expert_offset + self.local_num_experts,
            )
        return prepared

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_gate_up: torch.Tensor,
        w2: torch.Tensor,
        weight_key: tuple[int, ...],
    ) -> torch.Tensor:
        from flashinfer import ActivationType, RoutingMethodType, mxfp8_quantize
        from flashinfer.fused_moe import (
            Fp8QuantizationType,
            trtllm_fp8_block_scale_routed_moe,
        )
        from flashinfer.tllm_enums import WeightLayout
        from flashinfer.utils import device_support_pdl

        prepared = self._prepare_weights(w13_gate_up, w2, weight_key)
        hidden_q, hidden_sf = mxfp8_quantize(
            hidden_states.contiguous(), False, backend="cute-dsl"
        )
        hidden_sf = hidden_sf.view(torch.uint8).reshape(
            hidden_states.shape[0], self.hidden_size // _MXFP8_GROUP_SIZE
        )
        packed_topk = _pack_topk_ids(topk_ids, topk_weights)
        tune_max_tokens = 1 << max(hidden_states.shape[0] - 1, 0).bit_length()
        output = trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=hidden_q,
            hidden_states_scale=hidden_sf,
            gemm1_weights=prepared.gemm1_weights,
            gemm1_weights_scale=prepared.gemm1_scales,
            gemm2_weights=prepared.gemm2_weights,
            gemm2_weights_scale=prepared.gemm2_scales,
            num_experts=self.num_experts,
            top_k=topk_ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=1.0,
            routing_method_type=RoutingMethodType.TopK.value,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK.value,
            do_finalize=True,
            enable_pdl=hidden_states.shape[0] <= 8192
            and device_support_pdl(hidden_states.device),
            tune_max_num_tokens=tune_max_tokens,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=ActivationType.Swiglu.value,
        )
        return output[0] if isinstance(output, list) else output


def _flashinfer_moe_runner_type(quantization: str):
    """Resolve a runner only after checking that its FlashInfer API is present."""

    if quantization == "nvfp4":
        try:
            from flashinfer import nvfp4_block_scale_interleave, nvfp4_quantize
            from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "FlashInfer NVFP4 routed MoE requires the NVFP4 quantizer, "
                "layout helpers, and TRT-LLM routed kernel"
            ) from exc
        del nvfp4_block_scale_interleave, nvfp4_quantize
        del trtllm_fp4_block_scale_routed_moe
        return _FlashInferNVFP4Runner
    elif quantization == "mxfp8":
        try:
            from flashinfer import block_scale_interleave, mxfp8_quantize
            from flashinfer.fused_moe import (
                Fp8QuantizationType,
                trtllm_fp8_block_scale_routed_moe,
            )
            from flashinfer.fused_moe.core import (
                get_reorder_rows_for_gated_act_gemm_row_indices,
            )
            from flashinfer.tllm_enums import WeightLayout
            from flashinfer.utils import (
                get_shuffle_matrix_a_row_indices,
                get_shuffle_matrix_sf_a_row_indices,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "FlashInfer MXFP8 routed MoE requires FlashInfer 0.6.14+ with "
                "MXFP8 quantization, shuffled-weight helpers, and the TRT-LLM "
                "FP8 routed kernel"
            ) from exc
        del block_scale_interleave, mxfp8_quantize
        del Fp8QuantizationType, trtllm_fp8_block_scale_routed_moe
        del get_reorder_rows_for_gated_act_gemm_row_indices, WeightLayout
        del get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices
        return _FlashInferMXFP8Runner
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no runner branch"
        )


def _flashinfer_moe_description(quantization: str) -> str:
    """Return the explicit log description for one supported quantization."""

    if quantization == "nvfp4":
        return (
            f"per-token NVFP4, 4-over-6="
            f"{os.environ.get('FLASHINFER_NVFP4_4OVER6', '0')}, "
            f"E4M3={int(_FlashInferNVFP4Runner._e4m3_max())}"
        )
    elif quantization == "mxfp8":
        return "MXFP8 weights and per-token activations"
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no log-description branch"
        )


class _FlashInferForwardBF16Backward(torch.autograd.Function):
    """Exact FlashInfer forward and module-level BF16 surrogate backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        topk_weights,
        topk_ids,
        w13_gate_up,
        w2,
        runner,
        weight_key,
    ):
        ctx.runner = runner
        ctx.save_for_backward(hidden_states, topk_weights, topk_ids, w13_gate_up, w2)
        return runner.forward(
            hidden_states, topk_weights, topk_ids, w13_gate_up, w2, weight_key
        )

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, topk_weights, topk_ids, w13_gate_up, w2 = ctx.saved_tensors
        needs = ctx.needs_input_grad

        with torch.enable_grad():
            hidden_ref = hidden_states.detach().requires_grad_(needs[0])
            weights_ref = topk_weights.detach().requires_grad_(needs[1])
            w13_ref = w13_gate_up.detach().requires_grad_(needs[3])
            w2_ref = w2.detach().requires_grad_(needs[4])
            output_ref = _bf16_local_routed_experts(
                hidden_ref,
                weights_ref,
                topk_ids,
                w13_ref,
                w2_ref,
                ctx.runner.local_expert_offset,
            )

        grad_inputs = [hidden_ref, weights_ref, w13_ref, w2_ref]
        requested_mask = (needs[0], needs[1], needs[3], needs[4])
        requested = [
            tensor for tensor, need in zip(grad_inputs, requested_mask) if need
        ]
        computed = torch.autograd.grad(
            output_ref,
            requested,
            grad_output,
            allow_unused=True,
        )
        computed_iter = iter(computed)
        grads = [
            next(computed_iter) if need else None for need in requested_mask
        ]
        # Megatron optimizers can update ``param.data`` without incrementing the
        # owning Parameter's version counter. Invalidate after every training
        # backward so the next forward cannot reuse pre-update quantized weights.
        ctx.runner.invalidate_weights()
        return grads[0], grads[1], None, grads[2], grads[3], None, None


def _validate_layer(moe_layer, hidden_states: torch.Tensor, intermediate_tensors) -> str:
    config = moe_layer.config
    experts = moe_layer.experts
    quantization = _flashinfer_moe_quantization(config)
    if not isinstance(experts, FlashInferGroupedMLP):
        raise TypeError(
            f"{_ENV}=1 requires FlashInferGroupedMLP BF16 parameters, got {type(experts)}"
        )
    if intermediate_tensors is not None:
        raise ValueError("FlashInfer MoE does not support partial-layer CUDA graph execution")
    if moe_layer.shared_expert_overlap:
        raise ValueError("FlashInfer MoE does not support shared-expert overlap")
    if config.moe_latent_size is not None:
        raise ValueError("FlashInfer MoE does not support latent MoE projections")
    if config.add_bias_linear:
        raise ValueError("FlashInfer MoE does not support expert bias")
    if not config.gated_linear_unit or config.activation_func is not F.silu:
        raise ValueError("FlashInfer MoE currently supports gated SwiGLU only")
    if config.activation_func_clamp_value is not None or config.glu_linear_offset != 0.0:
        raise ValueError("FlashInfer MoE does not support SwiGLU clamp or linear offset")
    if config.moe_expert_capacity_factor is not None:
        raise ValueError("FlashInfer MoE does not support expert capacity or token dropping")
    if config.moe_apply_probs_on_input:
        raise ValueError("FlashInfer MoE requires routing weights in the fused finalize")
    if experts.tp_group.size() != 1:
        raise ValueError("FlashInfer MoE currently requires expert tensor parallel size 1")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"FlashInfer MoE expects BF16 hidden states, got {hidden_states.dtype}")
    if any(
        weight.dtype != torch.bfloat16
        for linear in (experts.linear_fc1, experts.linear_fc2)
        for weight in linear.parameters()
    ):
        raise TypeError("FlashInfer MoE master weights must remain BF16")
    if quantization == "nvfp4":
        if hidden_states.shape[-1] % 16 or config.moe_ffn_hidden_size % 16:
            raise ValueError("FlashInfer NVFP4 dimensions must be multiples of 16")
    elif quantization == "mxfp8":
        if hidden_states.shape[-1] % 128 or config.moe_ffn_hidden_size % 128:
            raise ValueError(
                "FlashInfer MXFP8 hidden and intermediate dimensions must be multiples of 128"
            )
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no validation branch"
        )
    if not hidden_states.is_cuda or torch.cuda.get_device_capability(hidden_states.device)[0] < 10:
        raise RuntimeError("FlashInfer routed MoE requires NVIDIA Blackwell (SM100+)")
    return quantization


def run_flashinfer_moe(
    moe_layer,
    hidden_states: torch.Tensor,
    intermediate_tensors,
    padding_mask: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, None]:
    """Run the exact routed FlashInfer forward and attach the BF16 surrogate backward."""

    quantization = _validate_layer(moe_layer, hidden_states, intermediate_tensors)
    runner_type = _flashinfer_moe_runner_type(quantization)
    description = _flashinfer_moe_description(quantization)
    runner = getattr(moe_layer, "_flashinfer_moe_runner", None)
    if runner is None or runner.quantization != quantization:
        runner = runner_type(
            num_experts=moe_layer.config.num_moe_experts,
            local_expert_offset=moe_layer.local_expert_indices[0],
            local_num_experts=moe_layer.num_local_experts,
            hidden_size=moe_layer.config.hidden_size,
            intermediate_size=moe_layer.config.moe_ffn_hidden_size,
        )
        moe_layer._flashinfer_moe_runner = runner

    if padding_mask is not None and bool(padding_mask.any().item()):
        raise ValueError("FlashInfer MoE does not yet support padded tokens")

    shared_expert_output = moe_layer.shared_experts_compute(hidden_states)
    probs, routing_map = moe_layer.route(
        hidden_states, padding_mask=padding_mask, input_ids=input_ids
    )
    topk_weights, topk_ids = _topk_from_dense_routing(
        probs,
        routing_map,
        moe_layer.config.moe_router_topk,
        ordered_topk_ids=_rollout_replay_topk_ids(moe_layer),
    )

    hidden_shape = hidden_states.shape
    local_hidden = hidden_states.reshape(-1, hidden_shape[-1])
    ep_group = moe_layer.ep_group
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    max_tokens, token_counts = _all_gather_token_counts(
        local_hidden.shape[0], local_hidden.device, ep_group, ep_size
    )
    if max_tokens == 0:
        routed_output = torch.zeros_like(hidden_states)
        return (
            routed_output
            if shared_expert_output is None
            else routed_output + shared_expert_output,
            None,
        )

    if ep_size > 1:
        global_hidden = _PaddedEPAllGather.apply(
            local_hidden, max_tokens, token_counts, ep_group, ep_rank, ep_size
        )
        global_topk_weights = _PaddedEPAllGather.apply(
            topk_weights, max_tokens, token_counts, ep_group, ep_rank, ep_size
        )
        global_topk_ids = _all_gather_padded_no_grad(
            topk_ids, max_tokens, ep_group, ep_size
        )
    else:
        global_hidden = local_hidden
        global_topk_weights = topk_weights
        global_topk_ids = topk_ids

    w13_gate_up, w2 = _grouped_mlp_weights(moe_layer.experts)
    global_output = _FlashInferForwardBF16Backward.apply(
        global_hidden,
        global_topk_weights,
        global_topk_ids,
        w13_gate_up,
        w2,
        runner,
        _source_weight_key(moe_layer.experts),
    )
    if ep_size > 1:
        global_output = _EPAllReduceSum.apply(global_output, ep_group)

    local_start = ep_rank * max_tokens
    local_output = global_output[
        local_start : local_start + token_counts[ep_rank]
    ].contiguous()
    routed_output = local_output.view(hidden_shape)
    if shared_expert_output is not None:
        routed_output = routed_output + shared_expert_output

    layer_number = moe_layer.layer_number or -1
    log_key = (quantization, layer_number)
    if log_key not in _LOGGED_LAYERS:
        log_single_rank(
            logger,
            logging.WARNING,
            "Layer %d FlashInfer MoE path active: routed TRT-LLM, %s, "
            "BF16 surrogate backward (%s=1)",
            layer_number,
            description,
            _ENV,
        )
        _LOGGED_LAYERS.add(log_key)

    return routed_output, None


__all__ = [
    "FlashInferGroupedMLP",
    "maybe_replace_flashinfer_moe_expert_spec",
    "run_flashinfer_moe",
    "use_flashinfer_moe",
]
