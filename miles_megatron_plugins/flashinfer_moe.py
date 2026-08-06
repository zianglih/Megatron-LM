# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental FlashInfer routed-MoE forward with a BF16 surrogate backward.

This module deliberately targets the rollout routed-MoE contracts for:

* FlashInfer ``trtllm_bf16_routed_moe`` for plain BF16 models and TE-selected
  BF16 layer contexts in quantized models
* FlashInfer ``trtllm_fp8_block_scale_routed_moe`` with MXFP8
* FlashInfer ``trtllm_fp4_block_scale_routed_moe`` with per-token NVFP4
* contiguous expert-parallel expert ownership
* gated SwiGLU experts

Megatron BF16 parameters remain the checkpoint and optimizer source of truth.
TransformerEngine's live module decision selects BF16 or the model's configured
quantization for each routed layer. Backward recomputes a BF16 expert module
from either the original BF16 operands or lazy dequantization of forward-derived
visible quantized operands. Neither operand mode is the derivative of the fused
quantized forward.
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Sequence

import torch
import torch.nn.functional as F

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.fp4_utils import get_fp4_recipe
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.transformer.moe.experts import TEGroupedMLP, _MoEActivationInFP32
from megatron.core.transformer.spec_utils import ModuleSpec, get_module
from megatron.core.utils import (
    get_pg_rank,
    log_single_rank,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)

_ENV = "MILES_USE_FLASHINFER_MOE"
_BACKWARD_OVERRIDE_ENV = "NVTE_BACKWARD_OVERRIDE"
_NVTX_MESSAGE = "flashinfer_moe"
HIGH_PRECISION_BACKWARD = "high_precision"
DEQUANTIZED_BACKWARD = "dequantized"
_LOGGED_LAYERS: set[tuple[str, str, str]] = set()
_MXFP8_GROUP_SIZE = 32
_TE_MXFP8_ROW_ALIGNMENT = 32
_NVFP4_GROUP_SIZE = 16
_TE_NVFP4_ROW_ALIGNMENT = 16
# The replay uses shared stateless TE shells, and ``functional_call``
# temporarily rebinds their parameters. This does not make the runner generally
# thread-safe; it only keeps one replay's shell state internally consistent.
_BF16_SURROGATE_LOCK = threading.Lock()


@contextmanager
def _flashinfer_nvtx_range(suffix: str):
    """Use Megatron's opt-in NVTX switch without a plugin-specific control."""

    nvtx_range_push(msg=_NVTX_MESSAGE, suffix=suffix)
    try:
        yield
    finally:
        nvtx_range_pop(msg=_NVTX_MESSAGE, suffix=suffix)


@dataclass(frozen=True)
class _FlashInferModules:
    api: ModuleType
    fused_moe: ModuleType
    fused_moe_core: ModuleType
    tllm_enums: ModuleType
    utils: ModuleType


@cache
def _flashinfer_modules() -> _FlashInferModules:
    """Load the opt-in FlashInfer runtime without making it a Megatron dependency."""

    return _FlashInferModules(
        api=importlib.import_module("flashinfer"),
        fused_moe=importlib.import_module("flashinfer.fused_moe"),
        fused_moe_core=importlib.import_module("flashinfer.fused_moe.core"),
        tllm_enums=importlib.import_module("flashinfer.tllm_enums"),
        utils=importlib.import_module("flashinfer.utils"),
    )


@dataclass(frozen=True)
class _TransformerEngineModules:
    api: ModuleType
    constants: ModuleType
    fp8: ModuleType


@cache
def _transformer_engine_modules() -> _TransformerEngineModules:
    """Load the exact TE surface only after the FlashInfer path is selected."""

    return _TransformerEngineModules(
        api=importlib.import_module("transformer_engine.pytorch"),
        constants=importlib.import_module("transformer_engine.pytorch.constants"),
        fp8=importlib.import_module("transformer_engine.pytorch.fp8"),
    )


@cache
def _te_quantized_tensor() -> ModuleType:
    return importlib.import_module("transformer_engine.pytorch.quantized_tensor")


def flashinfer_moe_backward_mode() -> str:
    """Resolve the explicit surrogate-backward operand mode."""

    value = os.environ.get(_BACKWARD_OVERRIDE_ENV)
    if value == HIGH_PRECISION_BACKWARD:
        return HIGH_PRECISION_BACKWARD
    elif value == DEQUANTIZED_BACKWARD:
        return DEQUANTIZED_BACKWARD
    else:
        raise ValueError(
            f"FlashInfer MoE requires {_BACKWARD_OVERRIDE_ENV} to be exactly "
            f"{HIGH_PRECISION_BACKWARD!r} or {DEQUANTIZED_BACKWARD!r}; got {value!r}"
        )


def _padded_linear_scales(scales: torch.Tensor, *, rows: int, scale_columns: int) -> torch.Tensor:
    """Adapt compact FlashInfer scales to TE's row-padded linear layout."""

    if scales.numel() != rows * scale_columns:
        raise ValueError(
            "FlashInfer MoE scale payload has the wrong size: "
            f"got {scales.numel()}, expected {rows * scale_columns}"
        )
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((scale_columns + 3) // 4) * 4
    compact = scales.view(torch.uint8).reshape(rows, scale_columns)
    if rows == padded_rows and scale_columns == padded_columns:
        return compact.contiguous()
    padded = torch.empty((padded_rows, padded_columns), device=scales.device, dtype=torch.uint8)
    padded[:rows, :scale_columns].copy_(compact)
    return padded


def _dequantize_weight_payload(
    payload: object, shape: torch.Size, *, dtype: torch.dtype
) -> torch.Tensor:
    """Decode one forward-QDQ weight and discard any quantization padding."""

    decoded = payload.dequantize(dtype=dtype)
    if len(decoded.shape) != len(shape) or any(
        decoded_size < output_size for decoded_size, output_size in zip(decoded.shape, shape)
    ):
        raise ValueError(
            "Decoded FlashInfer MoE weight is smaller than its master shape: "
            f"got {tuple(decoded.shape)}, expected at least {tuple(shape)}"
        )
    return decoded[tuple(slice(0, size) for size in shape)].contiguous()


def _quantized_storage_shell(storage):
    """Clone TE storage metadata without copying or consuming its payloads."""

    shell = object.__new__(type(storage))
    shell.__dict__.update(storage.__dict__)
    return shell


class _QDQWithIdentityGradient(torch.autograd.Function):
    """Use a decoded forward value while passing its gradient to the QDQ source."""

    @staticmethod
    def forward(ctx, source: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        ctx.source_dtype = source.dtype
        return decoded

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.to(ctx.source_dtype), None


def use_flashinfer_moe() -> bool:
    """Return whether the experimental routed-MoE path is enabled."""

    return os.environ.get(_ENV, "0") == "1"


def flashinfer_moe_dispatch_mode(config) -> str:
    """Resolve one explicit communication mode for the FlashInfer MoE path."""

    if config.moe_combine_in_fp32:
        raise ValueError(
            "FlashInfer MoE does not support FP32 combine because router gradients "
            "would bypass the BF16 surrogate boundary"
        )
    dispatcher = config.moe_token_dispatcher_type
    if dispatcher == "allgather":
        return "allgather"
    elif dispatcher == "alltoall":
        return "alltoall"
    elif dispatcher == "flex":
        raise NotImplementedError(
            "FlashInfer MoE does not yet support the flex token dispatcher "
            f"with {config.moe_flex_dispatcher_backend!r} backend"
        )
    else:
        raise NotImplementedError(
            f"FlashInfer MoE token dispatcher {dispatcher!r} has no execution branch"
        )


def _flashinfer_moe_quantization(config) -> str:
    """Resolve one explicitly supported routed-MoE execution precision."""

    if config.fp8 is not None and config.fp4 is not None:
        raise ValueError("FlashInfer MoE cannot enable FP8 and FP4 together")

    if config.fp8 is None and config.fp4 is None:
        return "bf16"
    elif config.fp8 is not None:
        if config.fp8_recipe != Fp8Recipe.mxfp8:
            raise ValueError(
                f"FlashInfer MoE does not support active FP8 recipe {config.fp8_recipe!r}; "
                "supported FP8 recipe: 'mxfp8'"
            )
        return "mxfp8"
    elif config.fp4 is not None:
        if config.fp4_recipe != Fp4Recipe.nvfp4:
            raise ValueError(
                f"FlashInfer MoE does not support active FP4 recipe {config.fp4_recipe!r}; "
                "supported FP4 recipe: 'nvfp4'"
            )
        return "nvfp4"
    else:
        raise NotImplementedError(
            "FlashInfer MoE precision configuration has no execution branch"
        )


def _flashinfer_moe_execution_precision(experts) -> str:
    """Match the FlashInfer runner to the TE decision for this routed expert."""

    context_quantized = _transformer_engine_modules().fp8.FP8GlobalStateManager.is_fp8_enabled()
    fc1_quantized = experts.linear_fc1.will_execute_quantized(context_quantized)
    fc2_quantized = experts.linear_fc2.will_execute_quantized(context_quantized)
    if fc1_quantized != fc2_quantized:
        raise ValueError(
            "FlashInfer MoE requires routed expert FC1 and FC2 to use the same precision"
        )
    if fc1_quantized:
        if experts._flashinfer_moe_quantization == "bf16":
            raise ValueError(
                "FlashInfer MoE cannot execute quantized routed experts without an "
                "MXFP8 or NVFP4 model precision"
            )
        return experts._flashinfer_moe_quantization
    return "bf16"


def _flashinfer_moe_effective_backward_mode(
    execution_precision: str, requested_backward_mode: str
) -> str:
    """Resolve whether this forward produced quantized operands to decode."""

    if requested_backward_mode not in (HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD):
        raise ValueError(f"Unsupported FlashInfer MoE backward mode {requested_backward_mode!r}")
    if execution_precision == "bf16":
        return HIGH_PRECISION_BACKWARD
    elif execution_precision in ("mxfp8", "nvfp4"):
        return requested_backward_mode
    else:
        raise NotImplementedError(
            f"FlashInfer MoE execution precision {execution_precision!r} has no backward branch"
        )


def _validate_flashinfer_nvfp4_recipe(config) -> None:
    """Require the row-scaled NVFP4 recipe implemented by the FlashInfer runner."""

    recipe = get_fp4_recipe(config)
    if not recipe.disable_rht:
        raise ValueError(
            "FlashInfer NVFP4 does not support random Hadamard transforms (RHT); "
            "set NVTE_NVFP4_DISABLE_RHT=1 before launching Python"
        )
    if not recipe.disable_stochastic_rounding:
        raise ValueError(
            "FlashInfer NVFP4 does not support stochastic rounding (SR); "
            "set NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1 before launching Python"
        )
    if not recipe.disable_2d_quantization:
        raise ValueError(
            "FlashInfer NVFP4 does not support 2D 16x16 weight scaling; "
            "set NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 before launching Python"
        )
    if not recipe.row_scaled_activation:
        raise ValueError(
            "FlashInfer NVFP4 supports only row-scaled activations; "
            "set NVTE_NVFP4_ROW_SCALED_ACTIVATION=1 before launching Python"
        )


def _validate_flashinfer_moe_config(config) -> str:
    """Validate model-static constraints before allocating expert parameters."""

    flashinfer_moe_dispatch_mode(config)
    quantization = _flashinfer_moe_quantization(config)
    if config.moe_shared_expert_overlap:
        raise ValueError("FlashInfer MoE does not support shared-expert overlap")
    if config.delay_wgrad_compute:
        raise ValueError("FlashInfer MoE does not support delayed expert weight gradients")
    if config.moe_latent_size is not None:
        raise ValueError("FlashInfer MoE does not support latent MoE projections")
    if config.add_bias_linear:
        raise ValueError("FlashInfer MoE does not support expert bias")
    if not config.gated_linear_unit or config.activation_func is not F.silu:
        raise ValueError("FlashInfer MoE currently supports gated SwiGLU only")
    if config.use_te_activation_func:
        raise ValueError("FlashInfer MoE does not support Transformer Engine activation modules")
    if config.fine_grained_activation_offloading:
        offloaded_expert_stages = {"expert_fc1", "moe_act"}.intersection(config.offload_modules)
        if offloaded_expert_stages:
            raise ValueError(
                "FlashInfer MoE does not support fine-grained activation offloading for "
                f"{sorted(offloaded_expert_stages)}"
            )
    if config.activation_func_clamp_value is not None or config.glu_linear_offset != 0.0:
        raise ValueError("FlashInfer MoE does not support SwiGLU clamp or linear offset")
    if config.moe_expert_capacity_factor is not None:
        raise ValueError("FlashInfer MoE does not support expert capacity or token dropping")
    if config.moe_router_padding_for_quantization:
        raise ValueError("FlashInfer MoE does not support router padding")
    if config.moe_apply_probs_on_input:
        raise ValueError("FlashInfer MoE requires routing weights in the fused finalize")
    if config.expert_tensor_parallel_size != 1:
        raise ValueError("FlashInfer MoE currently requires expert tensor parallel size 1")
    if config.params_dtype != torch.bfloat16 or config.fp8_param or config.fp4_param:
        raise TypeError("FlashInfer MoE master weights must remain BF16")
    if config.num_moe_experts > 2048:
        raise ValueError("FlashInfer routed MoE supports at most 2048 global experts")
    if quantization == "bf16":
        if config.hidden_size % 128 or config.moe_ffn_hidden_size % 128:
            raise ValueError(
                "FlashInfer BF16 hidden and intermediate dimensions must be multiples of 128"
            )
    elif quantization == "mxfp8":
        if config.num_moe_experts % 4 or config.num_moe_experts <= 1:
            raise ValueError(
                "FlashInfer MXFP8 requires global experts divisible by 4 and greater "
                f"than dispatched kernel top-k 1, got {config.num_moe_experts}"
            )
        if config.hidden_size % 128 or config.moe_ffn_hidden_size % 128:
            raise ValueError(
                "FlashInfer MXFP8 hidden and intermediate dimensions must be multiples of 128"
            )
    elif quantization == "nvfp4":
        _validate_flashinfer_nvfp4_recipe(config)
        if config.hidden_size % 16 or config.moe_ffn_hidden_size % 16:
            raise ValueError("FlashInfer NVFP4 dimensions must be multiples of 16")
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no validation branch"
        )
    return quantization


class FlashInferGroupedMLP(TEGroupedMLP):
    """Megatron TE expert parameters with a FlashInfer compute boundary."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None):
        self._flashinfer_moe_backward_mode = flashinfer_moe_backward_mode()
        self._flashinfer_moe_quantization = _validate_flashinfer_moe_config(config)
        self._flashinfer_moe_dispatch_mode = flashinfer_moe_dispatch_mode(config)
        self._flashinfer_moe_activation_in_fp32 = config.moe_activation_in_fp32
        self._flashinfer_moe_fused_activation = config.bias_activation_fusion
        self._flashinfer_moe_runner = None
        if pg_collection is None:
            raise ValueError("FlashInferGroupedMLP requires a ProcessGroupCollection")
        super().__init__(num_local_experts, config, submodules, pg_collection=pg_collection)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Run FlashInfer on assignments already dispatched to local experts."""

        with _flashinfer_nvtx_range("routed_experts_forward"):
            return _run_dispatched_flashinfer_moe(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

    def backward_dw(self):
        """Custom autograd produces weight gradients without delayed TE work."""

        return None


def maybe_replace_flashinfer_moe_expert_spec(submodules):
    """Replace only expert compute while retaining Megatron's TE parameter contract."""

    if not use_flashinfer_moe():
        return submodules
    if submodules is None or submodules.experts is None:
        raise ValueError(f"{_ENV}=1 requires an MoE expert module specification")
    if (
        submodules.shared_experts is not None
        and get_module(submodules.shared_experts) is FlashInferGroupedMLP
    ):
        raise ValueError("FlashInfer MoE replaces routed experts only, never shared experts")

    expert_module = get_module(submodules.experts)
    if expert_module is FlashInferGroupedMLP:
        return submodules

    replacement = copy.copy(submodules)
    if not isinstance(submodules.experts, ModuleSpec):
        raise TypeError(f"{_ENV}=1 requires experts to use Megatron ModuleSpec")
    expert_spec = copy.copy(submodules.experts)
    expert_spec.module = FlashInferGroupedMLP
    if expert_spec.submodules is None:
        raise ValueError(
            "FlashInfer MoE requires explicit Transformer Engine grouped FC1/FC2 submodules"
        )
    linear_fc1 = expert_spec.submodules.linear_fc1
    linear_fc2 = expert_spec.submodules.linear_fc2
    if (
        get_module(linear_fc1) is not TEColumnParallelGroupedLinear
        or get_module(linear_fc2) is not TERowParallelGroupedLinear
    ):
        raise ValueError("FlashInfer MoE requires Transformer Engine grouped FC1/FC2 submodules")
    replacement.experts = expert_spec
    return replacement


def _pack_topk_ids(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Mirror SGLang's ``PackTopkIds`` packed BF16 routing representation."""

    weight_bits = topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return ((topk_ids.to(torch.int32) << 16) | weight_bits).contiguous()


def _grouped_mlp_weight_parameters(
    experts: FlashInferGroupedMLP,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Return expert parameters without materializing a stacked BF16 copy."""

    w13 = tuple(
        getattr(experts.linear_fc1, f"weight{expert}")
        for expert in range(experts.num_local_experts)
    )
    w2 = tuple(
        getattr(experts.linear_fc2, f"weight{expert}")
        for expert in range(experts.num_local_experts)
    )
    return w13, w2


def _dispatched_topk_inputs(
    permuted_probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    local_expert_offset: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build singleton routed-kernel inputs for expert-dispatched assignments."""

    if permuted_probs.ndim != 1:
        raise ValueError(
            "FlashInfer MoE expects one routing probability per dispatched row, "
            f"got {tuple(permuted_probs.shape)}"
        )
    if tokens_per_expert.ndim != 1 or tokens_per_expert.numel() != num_local_experts:
        raise ValueError(
            "FlashInfer MoE expects one token count per local expert, "
            f"got {tuple(tokens_per_expert.shape)} for {num_local_experts} experts"
        )

    counts = tokens_per_expert.to(device=permuted_probs.device, dtype=torch.long, non_blocking=True)
    local_expert_ids = torch.arange(
        local_expert_offset,
        local_expert_offset + num_local_experts,
        device=permuted_probs.device,
        dtype=torch.int32,
    )
    topk_ids = torch.repeat_interleave(
        local_expert_ids, counts, output_size=permuted_probs.shape[0]
    ).unsqueeze(1)
    topk_weights = permuted_probs.unsqueeze(1)
    return topk_weights.contiguous(), topk_ids.contiguous()


def _source_weight_key(*weight_groups: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Best-effort invalidation key for optimizer-updated BF16 master weights."""

    key = []
    for weights in weight_groups:
        for weight in weights:
            key.extend((weight.untyped_storage().data_ptr(), weight._version))
    return tuple(key)


@dataclass
class _FlashInferForwardResult:
    output: torch.Tensor
    backward_hidden_states: object | None
    backward_w13: tuple[torch.Tensor, ...] | None
    backward_w2: tuple[torch.Tensor, ...] | None


@dataclass
class _PreparedBF16Weights:
    gemm1_weights: torch.Tensor
    gemm2_weights: torch.Tensor


@dataclass
class _PreparedMXFP8Weights:
    gemm1_weights: torch.Tensor
    gemm1_scales: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_scales: torch.Tensor
    backward_w13: tuple[torch.Tensor, ...] | None = None
    backward_w2: tuple[torch.Tensor, ...] | None = None


@dataclass
class _PreparedNVFP4Weights:
    gemm1_weights: torch.Tensor
    gemm1_scales: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_scales: torch.Tensor
    output1_scale: torch.Tensor
    output1_gate_scale: torch.Tensor
    output2_scale: torch.Tensor
    backward_w13: tuple[torch.Tensor, ...] | None = None
    backward_w2: tuple[torch.Tensor, ...] | None = None


class _TEBF16GroupedLinear:
    """Storage-free TE GroupedLinear with functional expert weights."""

    def __init__(
        self, *, num_gemms: int, in_features: int, out_features: int, device: torch.device
    ) -> None:
        te = _transformer_engine_modules().api
        self.num_gemms = num_gemms
        self.device = torch.device(device)
        with te.autocast(enabled=False):
            self.op = te.GroupedLinear(
                num_gemms=num_gemms,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                return_bias=False,
                params_dtype=torch.bfloat16,
                device="meta",
            )

        if self.op.primary_weights_in_fp8:
            raise RuntimeError(
                "FlashInfer MoE BF16 surrogate was constructed with quantized parameters"
            )
        expected_parameters = tuple(f"weight{index}" for index in range(num_gemms))
        actual_parameters = tuple(name for name, _ in self.op.named_parameters())
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                "FlashInfer MoE BF16 surrogate requires separate TE grouped weights; "
                f"got parameters {actual_parameters}"
            )

        # TE saves bias placeholders even with bias disabled. Keep the empty
        # tensors off meta so its legacy autograd path can save them.
        for index in range(num_gemms):
            setattr(
                self.op, f"bias{index}", torch.empty(0, dtype=torch.bfloat16, device=self.device)
            )

    def __call__(
        self, hidden_states: torch.Tensor, splits: torch.Tensor, weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        te = _transformer_engine_modules().api
        weights = tuple(weights)
        if len(weights) != self.num_gemms:
            raise ValueError(
                "FlashInfer MoE BF16 surrogate expected one weight per local expert, "
                f"got {len(weights)} for {self.num_gemms} experts"
            )
        if hidden_states.device != self.device:
            raise ValueError(
                "FlashInfer MoE BF16 surrogate device changed from "
                f"{self.device} to {hidden_states.device}"
            )
        if hidden_states.dtype != torch.bfloat16 or any(
            weight.dtype != torch.bfloat16 for weight in weights
        ):
            raise TypeError("FlashInfer MoE BF16 surrogate requires BF16 activations and weights")
        if any(weight.device != self.device for weight in weights):
            raise ValueError("FlashInfer MoE BF16 surrogate weights must share the input device")
        if any(weight.requires_grad != weights[0].requires_grad for weight in weights[1:]):
            raise ValueError(
                "FlashInfer MoE BF16 surrogate requires a uniform expert-weight grad state"
            )

        functional_weights = {f"weight{index}": weight for index, weight in enumerate(weights)}
        with te.autocast(enabled=False):
            return torch.func.functional_call(
                self.op, functional_weights, args=(hidden_states, splits), strict=True
            )


class _BF16GroupedMLPSurrogate:
    """Two grouped BF16 GEMMs with Megatron's routed SwiGLU between them."""

    def __init__(self, *, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._device: torch.device | None = None
        self._fc1: _TEBF16GroupedLinear | None = None
        self._fc2: _TEBF16GroupedLinear | None = None

    def _initialize(self, device: torch.device) -> None:
        device = torch.device(device)
        if self._device is None:
            fc1 = _TEBF16GroupedLinear(
                num_gemms=self.num_experts,
                in_features=self.hidden_size,
                out_features=2 * self.intermediate_size,
                device=device,
            )
            fc2 = _TEBF16GroupedLinear(
                num_gemms=self.num_experts,
                in_features=self.intermediate_size,
                out_features=self.hidden_size,
                device=device,
            )
            self._fc1, self._fc2, self._device = fc1, fc2, device
        if device != self._device:
            raise ValueError(
                f"FlashInfer MoE BF16 surrogate device changed from {self._device} to {device}"
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        tokens_per_expert: Sequence[int],
        *,
        activation_in_fp32: bool,
        fused_activation: bool,
        fc2_input_qdq=None,
    ) -> torch.Tensor:
        if len(tokens_per_expert) != self.num_experts:
            raise ValueError(
                "FlashInfer MoE BF16 surrogate expected one token count per local expert, "
                f"got {len(tokens_per_expert)} for {self.num_experts} experts"
            )
        if sum(tokens_per_expert) != hidden_states.shape[0]:
            raise ValueError(
                "FlashInfer MoE dispatched token counts do not match hidden rows: "
                f"{sum(tokens_per_expert)} != {hidden_states.shape[0]}"
            )
        if topk_weights.shape != (hidden_states.shape[0], 1):
            raise ValueError(
                "FlashInfer MoE BF16 surrogate expects one routing weight per row, "
                f"got {tuple(topk_weights.shape)}"
            )

        # Reuse one CPU split tensor for both grouped GEMMs. Passing the Python
        # sequence would make TE materialize this metadata once per GEMM.
        splits = torch.tensor(tokens_per_expert, dtype=torch.int64, device="cpu")
        with _BF16_SURROGATE_LOCK:
            self._initialize(hidden_states.device)
            assert self._fc1 is not None and self._fc2 is not None
            with _flashinfer_nvtx_range("surrogate_fc1"):
                fc1_output = self._fc1(hidden_states, splits, w13_gate_up)
            with _flashinfer_nvtx_range("surrogate_activation"):
                if fc2_input_qdq is not None:
                    activation_dtype = (
                        torch.float32
                        if activation_in_fp32
                        else fc2_input_qdq.fc2_input_qdq_source_dtype
                    )
                    gate, up = fc1_output.to(activation_dtype).chunk(2, dim=-1)
                    activated = (F.silu(gate) * up).to(
                        fc2_input_qdq.fc2_input_qdq_source_dtype
                    )
                elif activation_in_fp32:
                    activated = _MoEActivationInFP32.apply(
                        fc1_output, torch.ones_like(topk_weights), 0.0
                    )
                elif fused_activation:
                    activated = weighted_bias_swiglu_impl(
                        fc1_output,
                        None,
                        torch.ones_like(topk_weights),
                        fp8_input_store=False,
                    )
                else:
                    gate, up = fc1_output.chunk(2, dim=-1)
                    activated = (F.silu(gate) * up).to(fc1_output.dtype)
            if fc2_input_qdq is not None:
                with _flashinfer_nvtx_range("surrogate_fc2_input_qdq"):
                    activated = fc2_input_qdq.qdq_fc2_input(activated)
            with _flashinfer_nvtx_range("surrogate_fc2"):
                output = self._fc2(activated, splits, w2)
            with _flashinfer_nvtx_range("surrogate_finalize"):
                return output * topk_weights.to(output.dtype)


class _FlashInferRunnerBase:
    """Common layer-local metadata and nonpersistent quantized-weight cache."""

    fc2_input_qdq_source_dtype: torch.dtype | None = None

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
        self._weight_key: tuple[int, ...] | None = None
        self._prepared_backward_mode: str | None = None
        self._prepared: (
            _PreparedBF16Weights | _PreparedMXFP8Weights | _PreparedNVFP4Weights | None
        ) = None
        self._permute_cache: dict = {}
        self._bf16_surrogate = _BF16GroupedMLPSurrogate(
            num_experts=local_num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def invalidate_weights(self) -> None:
        """Drop the forward mirror before the next optimizer-updated iteration."""

        self._weight_key = None
        self._prepared_backward_mode = None
        self._prepared = None

    def bf16_surrogate(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        tokens_per_expert: Sequence[int],
        *,
        activation_in_fp32: bool,
        fused_activation: bool,
        dequantized_backward: bool,
    ) -> torch.Tensor:
        """Run the shared grouped-BF16 surrogate without retaining expert weights."""

        if dequantized_backward and self.fc2_input_qdq_source_dtype is None:
            raise NotImplementedError(
                f"FlashInfer {self.quantization} MoE has no FC2-input QDQ implementation"
            )

        return self._bf16_surrogate(
            hidden_states,
            topk_weights,
            w13_gate_up,
            w2,
            tokens_per_expert,
            activation_in_fp32=activation_in_fp32,
            fused_activation=fused_activation,
            fc2_input_qdq=self if dequantized_backward else None,
        )

    def qdq_fc2_input(self, activation: torch.Tensor) -> torch.Tensor:
        """Return the BF16 QDQ input consumed by surrogate FC2."""

        raise NotImplementedError(
            f"FlashInfer {self.quantization} MoE has no FC2-input QDQ implementation"
        )


class _FlashInferBF16Runner(_FlashInferRunnerBase):
    """BF16 exact-forward adapter for Megatron's unquantized layer contexts."""

    quantization = "bf16"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError(
                "FlashInfer BF16 hidden and intermediate dimensions must be multiples of 128"
            )

    def _prepare_weights(
        self,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
    ) -> _PreparedBF16Weights:
        if self._prepared is not None and self._weight_key == weight_key:
            return self._prepared
        self.invalidate_weights()

        flashinfer = _flashinfer_modules()
        gemm1_weights = None
        gemm2_weights = None
        epilogue_tile_m = 128
        block_k_bytes = 128
        for expert in range(self.local_num_experts):
            gate, up = w13_gate_up[expert].chunk(2, dim=0)
            up_gate = torch.cat((up, gate), dim=0).contiguous().view(torch.uint8)
            fc2 = w2[expert].contiguous().view(torch.uint8)
            gemm1_rows = flashinfer.fused_moe_core._maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache, up_gate, epilogue_tile_m, is_gated_act_gemm=True
            ).to(up_gate.device)
            gemm2_rows = flashinfer.fused_moe_core.get_w2_permute_indices_with_cache(
                self._permute_cache, fc2, epilogue_tile_m
            ).to(fc2.device)
            gemm1 = flashinfer.fused_moe.convert_to_block_layout(
                up_gate.index_select(0, gemm1_rows).contiguous(), block_k_bytes
            ).view(torch.bfloat16)
            gemm2 = flashinfer.fused_moe.convert_to_block_layout(
                fc2.index_select(0, gemm2_rows).contiguous(), block_k_bytes
            ).view(torch.bfloat16)
            if gemm1_weights is None:
                gemm1_weights = torch.empty(
                    (self.local_num_experts, *gemm1.shape), device=gemm1.device, dtype=gemm1.dtype
                )
                gemm2_weights = torch.empty(
                    (self.local_num_experts, *gemm2.shape), device=gemm2.device, dtype=gemm2.dtype
                )
            gemm1_weights[expert].copy_(gemm1)
            gemm2_weights[expert].copy_(gemm2)

        if gemm1_weights is None or gemm2_weights is None:
            raise ValueError("FlashInfer BF16 routed MoE requires at least one local expert")

        prepared = _PreparedBF16Weights(gemm1_weights=gemm1_weights, gemm2_weights=gemm2_weights)
        self._weight_key = weight_key
        self._prepared_backward_mode = HIGH_PRECISION_BACKWARD
        self._prepared = prepared
        if os.environ.get("MILES_FLASHINFER_MOE_DEBUG") == "1":
            logger.warning(
                "FlashInfer MoE materialized BF16 weights for experts [%d, %d)",
                self.local_expert_offset,
                self.local_expert_offset + self.local_num_experts,
            )
        return prepared

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
        backward_mode: str,
    ) -> _FlashInferForwardResult:
        flashinfer = _flashinfer_modules()
        if backward_mode != HIGH_PRECISION_BACKWARD:
            raise ValueError(
                "FlashInfer BF16 MoE has no dequantized operands; "
                f"got backward mode {backward_mode!r}"
            )
        with _flashinfer_nvtx_range("weight_prepare_bf16"):
            prepared = self._prepare_weights(w13_gate_up, w2, weight_key)
        with _flashinfer_nvtx_range("kernel_input_pack_bf16"):
            packed_topk = _pack_topk_ids(topk_ids, topk_weights)
            tune_max_tokens = 1 << max(hidden_states.shape[0] - 1, 0).bit_length()
        with _flashinfer_nvtx_range("fused_kernel_bf16"):
            output = flashinfer.fused_moe.trtllm_bf16_routed_moe(
                topk_ids=packed_topk,
                hidden_states=hidden_states,
                gemm1_weights=prepared.gemm1_weights,
                gemm2_weights=prepared.gemm2_weights,
                num_experts=self.num_experts,
                top_k=topk_ids.shape[1],
                n_group=None,
                topk_group=None,
                intermediate_size=self.intermediate_size,
                local_expert_offset=self.local_expert_offset,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=flashinfer.api.RoutingMethodType.TopK.value,
                use_shuffled_weight=True,
                weight_layout=flashinfer.tllm_enums.WeightLayout.BlockMajorK.value,
                do_finalize=True,
                enable_pdl=(
                    hidden_states.shape[0] <= 8192
                    and flashinfer.utils.device_support_pdl(hidden_states.device)
                ),
                tune_max_num_tokens=tune_max_tokens,
                activation_type=flashinfer.api.ActivationType.Swiglu.value,
            )
        return _FlashInferForwardResult(
            output=output, backward_hidden_states=None, backward_w13=None, backward_w2=None
        )


class _FlashInferMXFP8Runner(_FlashInferRunnerBase):
    """MXFP8 exact-forward adapter matching Miles and SGLang layouts."""

    quantization = "mxfp8"
    fc2_input_qdq_source_dtype = torch.bfloat16

    @staticmethod
    @cache
    def _te_storage() -> ModuleType:
        return importlib.import_module(
            "transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage"
        )

    @classmethod
    def _activation_storage(
        cls, data: torch.Tensor, scales: torch.Tensor, *, dtype: torch.dtype
    ):
        """Wrap the exact FlashInfer rowwise MXFP8 payload for deferred decode."""

        te = _transformer_engine_modules()
        if data.ndim != 2 or data.shape[1] % _MXFP8_GROUP_SIZE:
            raise ValueError(
                "FlashInfer MXFP8 activation must be [M, K] with K divisible by 32, "
                f"got {tuple(data.shape)}"
            )
        rows, columns = data.shape
        if rows == 0:
            raise ValueError("FlashInfer MXFP8 storage requires at least one row")
        linear_scales = _padded_linear_scales(
            scales, rows=rows, scale_columns=columns // _MXFP8_GROUP_SIZE
        )
        return cls._te_storage().MXFP8TensorStorage(
            data.view(torch.uint8),
            linear_scales,
            None,
            None,
            te.constants.DType.kFloat8E4M3,
            None,
            False,
            fake_dtype=dtype,
        )

    @classmethod
    def dequantize_activation(
        cls, data: torch.Tensor, scales: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """GPU-dequantize the exact rowwise MXFP8 activation used by FlashInfer."""

        if data.shape[0] == 0:
            return torch.empty_like(data, dtype=dtype)
        return cls._activation_storage(data, scales, dtype=dtype).dequantize(dtype=dtype)

    @classmethod
    def qdq_fc2_input(cls, activation: torch.Tensor) -> torch.Tensor:
        """QDQ the configured activation output for the BF16 FC2 replay."""

        if activation.dtype != cls.fc2_input_qdq_source_dtype:
            raise TypeError(
                "FlashInfer MXFP8 FC2-input QDQ requires activation dtype "
                f"{cls.fc2_input_qdq_source_dtype}, got {activation.dtype}"
            )
        with _flashinfer_nvtx_range("surrogate_fc2_input_quantize_mxfp8"):
            data, scales = cls._quantize_activation(activation.detach())
        with _flashinfer_nvtx_range("surrogate_fc2_input_dequantize_mxfp8"):
            decoded = cls.dequantize_activation(data, scales, dtype=torch.bfloat16)
        return _QDQWithIdentityGradient.apply(activation, decoded)

    @staticmethod
    def _quantize_activation(activation: torch.Tensor):
        """Quantize one activation with FlashInfer's rowwise MXFP8 contract."""

        if activation.ndim != 2 or activation.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer MXFP8 activation quantization requires a 2D BF16 tensor, "
                f"got shape={tuple(activation.shape)}, dtype={activation.dtype}"
            )
        flashinfer = _flashinfer_modules()
        return flashinfer.api.mxfp8_quantize(
            activation.contiguous(), False, backend="cute-dsl"
        )

    @staticmethod
    def _quantize_weight(weight: torch.Tensor, *, return_quantized: bool = False):
        """Quantize one expert matrix with Miles' rowwise MXFP8 contract."""

        te = _transformer_engine_modules()
        if weight.ndim != 2:
            raise ValueError(f"MXFP8 expert weight must be 2D, got {tuple(weight.shape)}")
        weight = weight.contiguous()
        num_rows, num_cols = weight.shape
        if num_cols % _MXFP8_GROUP_SIZE:
            raise ValueError(f"MXFP8 expert K={num_cols} must be divisible by {_MXFP8_GROUP_SIZE}")
        pad_rows = (-num_rows) % _TE_MXFP8_ROW_ALIGNMENT
        if pad_rows:
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(
                        (pad_rows, num_cols), device=weight.device, dtype=weight.dtype
                    ),
                ),
                dim=0,
            )

        quantizer = te.api.MXFP8Quantizer(
            fp8_dtype=te.constants.TE_DType[torch.float8_e4m3fn],
            rowwise=True,
            columnwise=False,
        )
        quantizer.internal = True
        quantized = quantizer.quantize(weight)
        qweight = quantized._rowwise_data[:num_rows, :num_cols]
        qweight = qweight.contiguous().view(torch.float8_e4m3fn)
        scale = quantized._rowwise_scale_inv[
            :num_rows, : num_cols // _MXFP8_GROUP_SIZE
        ].contiguous()
        result = (qweight, scale.view(torch.uint8))
        if return_quantized:
            return (*result, quantized)
        return result

    @classmethod
    def _quantize_gated_weight(
        cls, gate_up_weight: torch.Tensor, *, return_quantized: bool = False
    ):
        """Quantize Megatron [gate, up] rows and adapt them to TRT-LLM [up, gate]."""

        result = cls._quantize_weight(gate_up_weight, return_quantized=return_quantized)
        if return_quantized:
            qweight, scale, quantized = result
        else:
            qweight, scale = result
        gate_qweight, up_qweight = qweight.chunk(2, dim=0)
        gate_scale, up_scale = scale.chunk(2, dim=0)
        reordered = (
            torch.cat((up_qweight, gate_qweight), dim=0),
            torch.cat((up_scale, gate_scale), dim=0),
        )
        if return_quantized:
            return (*reordered, quantized)
        return reordered

    def _prepare_weights(
        self,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
        backward_mode: str = HIGH_PRECISION_BACKWARD,
    ) -> _PreparedMXFP8Weights:
        if backward_mode not in (HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD):
            raise ValueError(f"Unsupported FlashInfer MoE backward mode {backward_mode!r}")
        if (
            self._prepared is not None
            and self._weight_key == weight_key
            and self._prepared_backward_mode == backward_mode
        ):
            return self._prepared

        flashinfer = _flashinfer_modules()
        gemm1_weights = []
        gemm1_scales = []
        gemm2_weights = []
        gemm2_scales = []
        dequantized_backward = backward_mode == DEQUANTIZED_BACKWARD
        backward_w13 = [] if dequantized_backward else None
        backward_w2 = [] if dequantized_backward else None
        epilogue_tile_m = 128

        for expert in range(self.local_num_experts):
            # Miles owns Megatron [gate, up] masters. FlashInfer consumes W3/W1
            # [up, gate] before its gated-row interleave and row shuffle.
            w13_result = self._quantize_gated_weight(
                w13_gate_up[expert], return_quantized=dequantized_backward
            )
            w2_result = self._quantize_weight(
                w2[expert], return_quantized=dequantized_backward
            )
            if dequantized_backward:
                w13_q, w13_sf, w13_quantized = w13_result
                w2_q, w2_sf, w2_quantized = w2_result
                backward_w13.append(
                    _dequantize_weight_payload(
                        w13_quantized, w13_gate_up[expert].shape, dtype=w13_gate_up[expert].dtype
                    )
                )
                backward_w2.append(
                    _dequantize_weight_payload(
                        w2_quantized, w2[expert].shape, dtype=w2[expert].dtype
                    )
                )
                del w13_quantized, w2_quantized
            else:
                w13_q, w13_sf = w13_result
                w2_q, w2_sf = w2_result
            w13_u8 = w13_q.reshape(2 * self.intermediate_size, self.hidden_size).view(torch.uint8)
            w13_sf = w13_sf.reshape(
                2 * self.intermediate_size, self.hidden_size // _MXFP8_GROUP_SIZE
            )
            w2_u8 = w2_q.reshape(self.hidden_size, self.intermediate_size).view(torch.uint8)
            w2_sf = w2_sf.reshape(self.hidden_size, self.intermediate_size // _MXFP8_GROUP_SIZE)

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
                    flashinfer.fused_moe_core.get_reorder_rows_for_gated_act_gemm_row_indices(
                        w13_u8
                    ).to(w13_u8.device),
                    flashinfer.utils.get_shuffle_matrix_a_row_indices(w13_u8, epilogue_tile_m).to(
                        w13_u8.device
                    ),
                    flashinfer.utils.get_shuffle_matrix_a_row_indices(w2_u8, epilogue_tile_m).to(
                        w2_u8.device
                    ),
                    flashinfer.utils.get_shuffle_matrix_sf_a_row_indices(
                        w13_sf, epilogue_tile_m
                    ).to(w13_sf.device),
                    flashinfer.utils.get_shuffle_matrix_sf_a_row_indices(w2_sf, epilogue_tile_m).to(
                        w2_sf.device
                    ),
                )
                self._permute_cache[cache_key] = indices
            gated_rows, w13_weight_rows, w2_weight_rows, w13_scale_rows, w2_scale_rows = indices

            w13_u8 = w13_u8.index_select(0, gated_rows)
            w13_sf = w13_sf.index_select(0, gated_rows)
            gemm1_weights.append(w13_u8.index_select(0, w13_weight_rows).contiguous())
            gemm1_scales.append(
                flashinfer.api.block_scale_interleave(
                    w13_sf.index_select(0, w13_scale_rows).contiguous()
                ).reshape_as(w13_sf)
            )
            gemm2_weights.append(w2_u8.index_select(0, w2_weight_rows).contiguous())
            gemm2_scales.append(
                flashinfer.api.block_scale_interleave(
                    w2_sf.index_select(0, w2_scale_rows).contiguous()
                ).reshape_as(w2_sf)
            )

        prepared = _PreparedMXFP8Weights(
            gemm1_weights=torch.stack(gemm1_weights).view(torch.float8_e4m3fn),
            gemm1_scales=torch.stack(gemm1_scales).view(torch.uint8),
            gemm2_weights=torch.stack(gemm2_weights).view(torch.float8_e4m3fn),
            gemm2_scales=torch.stack(gemm2_scales).view(torch.uint8),
            backward_w13=tuple(backward_w13) if backward_w13 is not None else None,
            backward_w2=tuple(backward_w2) if backward_w2 is not None else None,
        )
        self._weight_key = weight_key
        self._prepared_backward_mode = backward_mode
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
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
        backward_mode: str,
    ) -> _FlashInferForwardResult:
        flashinfer = _flashinfer_modules()
        with _flashinfer_nvtx_range("weight_prepare_mxfp8"):
            prepared = self._prepare_weights(w13_gate_up, w2, weight_key, backward_mode)
        with _flashinfer_nvtx_range("activation_quantize_mxfp8"):
            hidden_q, hidden_sf = self._quantize_activation(hidden_states)
            hidden_sf = hidden_sf.view(torch.uint8).reshape(
                hidden_states.shape[0], self.hidden_size // _MXFP8_GROUP_SIZE
            )
        with _flashinfer_nvtx_range("kernel_input_pack_mxfp8"):
            packed_topk = _pack_topk_ids(topk_ids, topk_weights)
            tune_max_tokens = 1 << max(hidden_states.shape[0] - 1, 0).bit_length()
        with _flashinfer_nvtx_range("fused_kernel_mxfp8"):
            output = flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
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
                routing_method_type=flashinfer.api.RoutingMethodType.TopK.value,
                use_shuffled_weight=True,
                weight_layout=flashinfer.tllm_enums.WeightLayout.MajorK.value,
                do_finalize=True,
                enable_pdl=(
                    hidden_states.shape[0] <= 8192
                    and flashinfer.utils.device_support_pdl(hidden_states.device)
                ),
                tune_max_num_tokens=tune_max_tokens,
                fp8_quantization_type=flashinfer.fused_moe.Fp8QuantizationType.MxFp8,
                activation_type=flashinfer.api.ActivationType.Swiglu.value,
            )
        backward_hidden_states = None
        if backward_mode == DEQUANTIZED_BACKWARD:
            with _flashinfer_nvtx_range("qdq_capture_mxfp8"):
                backward_hidden_states = self._activation_storage(
                    hidden_q, hidden_sf, dtype=hidden_states.dtype
                )
        return _FlashInferForwardResult(
            output=output,
            backward_hidden_states=backward_hidden_states,
            backward_w13=prepared.backward_w13,
            backward_w2=prepared.backward_w2,
        )


class _FlashInferNVFP4Runner(_FlashInferRunnerBase):
    """NVFP4 exact-forward adapter."""

    quantization = "nvfp4"
    fc2_input_qdq_source_dtype = torch.bfloat16

    @staticmethod
    @cache
    def _te_storage() -> ModuleType:
        return importlib.import_module(
            "transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage"
        )

    @staticmethod
    @cache
    def _te_tensor() -> ModuleType:
        return importlib.import_module("transformer_engine.pytorch.tensor.nvfp4_tensor")

    @classmethod
    def _activation_storage(
        cls,
        data: torch.Tensor,
        scales: torch.Tensor,
        per_token_scale: torch.Tensor,
        *,
        dtype: torch.dtype,
        e4m3_max: int,
        use_4over6: bool,
    ):
        """Wrap exact NVFP4 q/block scales with equivalent TE row metadata."""

        te = _transformer_engine_modules()
        if data.ndim != 2:
            raise ValueError(
                f"FlashInfer NVFP4 activation must be [M, K/2], got {tuple(data.shape)}"
            )
        rows, packed_columns = data.shape
        columns = packed_columns * 2
        if columns % _NVFP4_GROUP_SIZE:
            raise ValueError(f"FlashInfer NVFP4 activation K={columns} must be divisible by 16")
        if per_token_scale.numel() != rows:
            raise ValueError(
                "FlashInfer NVFP4 per-token scale count mismatch: "
                f"got {per_token_scale.numel()}, expected {rows}"
            )
        if rows == 0:
            raise ValueError("FlashInfer NVFP4 storage requires at least one row")
        linear_scales = _padded_linear_scales(
            scales, rows=rows, scale_columns=columns // _NVFP4_GROUP_SIZE
        )
        # TE derives the row decode scale as amax / (E4M3_MAX * E2M1_MAX).
        # FlashInfer returns that decode scale directly, so reconstruct the amax.
        row_amax = per_token_scale.to(torch.float32).reshape(rows) * float(e4m3_max * 6)
        return cls._te_storage().NVFP4TensorStorage(
            data.view(torch.uint8),
            linear_scales,
            None,
            None,
            row_amax,
            None,
            te.constants.DType.kFloat4E2M1,
            None,
            False,
            fake_dtype=dtype,
            row_scaled_nvfp4=True,
            nvfp4_use_4over6=use_4over6,
            nvfp4_e4m3_max=e4m3_max,
        )

    @classmethod
    def dequantize_activation(
        cls,
        data: torch.Tensor,
        scales: torch.Tensor,
        per_token_scale: torch.Tensor,
        *,
        dtype: torch.dtype,
        e4m3_max: int,
        use_4over6: bool,
    ) -> torch.Tensor:
        """Decode FlashInfer NVFP4 q data through equivalent TE row metadata."""

        if data.shape[0] == 0:
            return torch.empty((0, data.shape[1] * 2), device=data.device, dtype=dtype)
        return cls._activation_storage(
            data,
            scales,
            per_token_scale,
            dtype=dtype,
            e4m3_max=e4m3_max,
            use_4over6=use_4over6,
        ).dequantize(dtype=dtype)

    @classmethod
    def qdq_fc2_input(cls, activation: torch.Tensor) -> torch.Tensor:
        """QDQ the BF16 activation with FlashInfer's row-scaled NVFP4 contract."""

        if activation.dtype != cls.fc2_input_qdq_source_dtype:
            raise TypeError(
                "FlashInfer NVFP4 FC2-input QDQ requires activation dtype "
                f"{cls.fc2_input_qdq_source_dtype}, got {activation.dtype}"
            )
        with _flashinfer_nvtx_range("surrogate_fc2_input_quantize_nvfp4"):
            data, scales, per_token_scale, e4m3_max, use_4over6 = cls._quantize_activation(
                activation.detach()
            )
        with _flashinfer_nvtx_range("surrogate_fc2_input_dequantize_nvfp4"):
            decoded = cls.dequantize_activation(
                data,
                scales,
                per_token_scale,
                dtype=torch.bfloat16,
                e4m3_max=e4m3_max,
                use_4over6=use_4over6,
            )
        return _QDQWithIdentityGradient.apply(activation, decoded)

    @classmethod
    def _quantize_activation(cls, activation: torch.Tensor):
        """Quantize one BF16 activation with FlashInfer's row-scaled NVFP4 contract."""

        if activation.ndim != 2 or activation.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer NVFP4 activation quantization requires a 2D BF16 tensor, "
                f"got shape={tuple(activation.shape)}, dtype={activation.dtype}"
            )
        flashinfer = _flashinfer_modules()
        rows, columns = activation.shape
        e4m3_max = int(cls._e4m3_max())
        use_4over6 = os.environ.get("FLASHINFER_NVFP4_4OVER6") == "1"
        input_global_scale = torch.full(
            (1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32, device=activation.device
        )
        data, scales, per_token_scale = flashinfer.api.nvfp4_quantize(
            activation.contiguous(),
            input_global_scale,
            sfLayout=flashinfer.api.SfLayout.layout_linear,
            per_token_activation=True,
            backend="cuda",
        )
        data = data.reshape(rows, columns // 2)
        scales = scales.view(torch.float8_e4m3fn).reshape(rows, columns // _NVFP4_GROUP_SIZE)
        return data, scales, per_token_scale, e4m3_max, use_4over6

    @staticmethod
    def _te_weight_e4m3_max() -> int:
        use_4over6 = os.environ.get("NVTE_NVFP4_4OVER6", "").strip().lower()
        use_256 = os.environ.get("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all").strip().lower()
        if use_4over6 in ("weights", "all") and use_256 in ("weights", "all"):
            return 256
        return 448

    @staticmethod
    def _global_decode_scale(global_amax: torch.Tensor, e4m3_max: int) -> torch.Tensor:
        encode_scale = torch.div(
            torch.tensor(float(e4m3_max * 6), device=global_amax.device, dtype=torch.float32),
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

    @classmethod
    def _quantize_weight(cls, weight: torch.Tensor, *, return_quantized: bool = False):
        """Quantize one expert matrix with Miles' TE weight-sync contract."""

        weight = weight.contiguous()
        num_rows, num_cols = weight.shape
        pad_rows = (-num_rows) % _TE_NVFP4_ROW_ALIGNMENT
        if pad_rows:
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(
                        (pad_rows, num_cols), device=weight.device, dtype=weight.dtype
                    ),
                ),
                dim=0,
            )

        use_4over6 = os.environ.get("NVTE_NVFP4_4OVER6", "").strip().lower() in (
            "weights",
            "all",
        )
        e4m3_max = cls._te_weight_e4m3_max()
        err_mode = os.environ.get("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()
        quantizer = cls._te_tensor().NVFP4Quantizer(
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
        # This quantization is an implementation detail of the custom autograd
        # boundary. Ask TE for lightweight storage directly instead of a Tensor
        # subclass whose autograd wrapper owns zero-sized base storage.
        quantizer.internal = True
        quantized = quantizer.quantize(weight)
        qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
        block_scale = quantized._rowwise_scale_inv[
            :num_rows, : num_cols // _NVFP4_GROUP_SIZE
        ].contiguous()
        global_amax = quantized._amax_rowwise.reshape(-1)[0]
        result = (
            qweight,
            block_scale.view(torch.float8_e4m3fn),
            cls._global_decode_scale(global_amax, e4m3_max),
        )
        if return_quantized:
            return (*result, quantized)
        return result

    @classmethod
    def _quantize_gated_weight(
        cls, gate_up_weight: torch.Tensor, *, return_quantized: bool = False
    ):
        """Quantize shared-scale gate/up rows, then adapt to FlashInfer order."""

        result = cls._quantize_weight(gate_up_weight, return_quantized=return_quantized)
        if return_quantized:
            qweight, block_scale, global_scale, quantized = result
        else:
            qweight, block_scale, global_scale = result
        gate_qweight, up_qweight = qweight.chunk(2, dim=0)
        gate_block_scale, up_block_scale = block_scale.chunk(2, dim=0)
        reordered = (
            torch.cat((up_qweight, gate_qweight), dim=0),
            torch.cat((up_block_scale, gate_block_scale), dim=0),
            global_scale,
        )
        if return_quantized:
            return (*reordered, quantized)
        return reordered

    @staticmethod
    def _e4m3_max() -> float:
        return (
            256.0
            if os.environ.get("FLASHINFER_NVFP4_4OVER6") == "1"
            and os.environ.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256") == "1"
            else 448.0
        )

    def _prepare_weights(
        self,
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
        backward_mode: str = HIGH_PRECISION_BACKWARD,
    ) -> _PreparedNVFP4Weights:
        if backward_mode not in (HIGH_PRECISION_BACKWARD, DEQUANTIZED_BACKWARD):
            raise ValueError(f"Unsupported FlashInfer MoE backward mode {backward_mode!r}")
        if (
            self._prepared is not None
            and self._weight_key == weight_key
            and self._prepared_backward_mode == backward_mode
        ):
            return self._prepared

        flashinfer = _flashinfer_modules()
        gemm1_weights = []
        gemm1_scales = []
        gemm2_weights = []
        gemm2_scales = []
        output1_scales = []
        output2_scales = []
        dequantized_backward = backward_mode == DEQUANTIZED_BACKWARD
        backward_w13 = [] if dequantized_backward else None
        backward_w2 = [] if dequantized_backward else None
        epilogue_tile_m = 128

        for expert in range(self.local_num_experts):
            # Quantize Megatron's shared-scale [gate, up] matrix before adapting
            # its emitted rows to the TRT-LLM gated kernel's [up, gate] order.
            w13_result = self._quantize_gated_weight(
                w13_gate_up[expert], return_quantized=dequantized_backward
            )
            w2_result = self._quantize_weight(
                w2[expert], return_quantized=dequantized_backward
            )
            if dequantized_backward:
                w13_q, w13_sf, w13_decode, w13_quantized = w13_result
                w2_q, w2_sf, w2_decode, w2_quantized = w2_result
                backward_w13.append(
                    _dequantize_weight_payload(
                        w13_quantized, w13_gate_up[expert].shape, dtype=w13_gate_up[expert].dtype
                    )
                )
                backward_w2.append(
                    _dequantize_weight_payload(
                        w2_quantized, w2[expert].shape, dtype=w2[expert].dtype
                    )
                )
                del w13_quantized, w2_quantized
            else:
                w13_q, w13_sf, w13_decode = w13_result
                w2_q, w2_sf, w2_decode = w2_result
            w13_q = w13_q.reshape(2 * self.intermediate_size, self.hidden_size // 2).view(
                torch.uint8
            )
            w13_sf = w13_sf.view(torch.float8_e4m3fn).reshape(
                2 * self.intermediate_size, self.hidden_size // 16
            )
            w2_q = w2_q.reshape(self.hidden_size, self.intermediate_size // 2).view(torch.uint8)
            w2_sf = w2_sf.view(torch.float8_e4m3fn).reshape(
                self.hidden_size, self.intermediate_size // 16
            )

            weight_indices = flashinfer.fused_moe_core._maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache, w13_q, epilogue_tile_m, is_gated_act_gemm=True
            ).to(w13_q.device)
            scale_indices = flashinfer.fused_moe_core._maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache,
                w13_sf.view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
                is_gated_act_gemm=True,
            ).to(w13_sf.device)
            gemm1_weights.append(w13_q[weight_indices].contiguous())
            gemm1_scales.append(
                flashinfer.api.nvfp4_block_scale_interleave(
                    w13_sf.view(torch.uint8)[scale_indices].contiguous()
                )
            )

            weight_indices = flashinfer.fused_moe_core.get_w2_permute_indices_with_cache(
                self._permute_cache, w2_q, epilogue_tile_m
            ).to(w2_q.device)
            scale_indices = flashinfer.fused_moe_core.get_w2_permute_indices_with_cache(
                self._permute_cache, w2_sf.view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16
            ).to(w2_sf.device)
            gemm2_weights.append(w2_q[weight_indices].contiguous())
            gemm2_scales.append(
                flashinfer.api.nvfp4_block_scale_interleave(
                    w2_sf.view(torch.uint8)[scale_indices].contiguous()
                )
            )
            output1_scales.append(w13_decode)
            output2_scales.append(w2_decode)

        output1_scale = torch.stack(output1_scales).to(torch.float32)
        prepared = _PreparedNVFP4Weights(
            gemm1_weights=torch.stack(gemm1_weights),
            gemm1_scales=torch.stack(gemm1_scales)
            .view(torch.float8_e4m3fn)
            .reshape(self.local_num_experts, 2 * self.intermediate_size, self.hidden_size // 16),
            gemm2_weights=torch.stack(gemm2_weights),
            gemm2_scales=torch.stack(gemm2_scales)
            .view(torch.float8_e4m3fn)
            .reshape(self.local_num_experts, self.hidden_size, self.intermediate_size // 16),
            output1_scale=output1_scale,
            output1_gate_scale=output1_scale.clone(),
            output2_scale=torch.stack(output2_scales).to(torch.float32),
            backward_w13=tuple(backward_w13) if backward_w13 is not None else None,
            backward_w2=tuple(backward_w2) if backward_w2 is not None else None,
        )
        self._weight_key = weight_key
        self._prepared_backward_mode = backward_mode
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
        w13_gate_up: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_key: tuple[int, ...],
        backward_mode: str,
    ) -> _FlashInferForwardResult:
        flashinfer = _flashinfer_modules()
        with _flashinfer_nvtx_range("weight_prepare_nvfp4"):
            prepared = self._prepare_weights(w13_gate_up, w2, weight_key, backward_mode)
        with _flashinfer_nvtx_range("activation_quantize_nvfp4"):
            (hidden_fp4, hidden_scales, per_token_scale, e4m3_max, use_4over6) = (
                self._quantize_activation(hidden_states)
            )
        with _flashinfer_nvtx_range("kernel_input_pack_nvfp4"):
            packed_topk = _pack_topk_ids(topk_ids, topk_weights)
            tune_max_tokens = 1 << max(hidden_states.shape[0] - 1, 0).bit_length()

        with _flashinfer_nvtx_range("fused_kernel_nvfp4"):
            output = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
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
                activation_type=flashinfer.api.ActivationType.Swiglu.value,
                tune_max_num_tokens=tune_max_tokens,
                enable_pdl=(
                    hidden_states.shape[0] <= 8192
                    and flashinfer.utils.device_support_pdl(hidden_states.device)
                ),
            )[0]
        backward_hidden_states = None
        if backward_mode == DEQUANTIZED_BACKWARD:
            with _flashinfer_nvtx_range("qdq_capture_nvfp4"):
                backward_hidden_states = self._activation_storage(
                    hidden_fp4,
                    hidden_scales,
                    per_token_scale,
                    dtype=hidden_states.dtype,
                    e4m3_max=e4m3_max,
                    use_4over6=use_4over6,
                )
        return _FlashInferForwardResult(
            output=output,
            backward_hidden_states=backward_hidden_states,
            backward_w13=prepared.backward_w13,
            backward_w2=prepared.backward_w2,
        )


def _flashinfer_moe_runner_type(quantization: str):
    """Resolve the runner for one explicitly supported quantization."""

    if quantization == "bf16":
        return _FlashInferBF16Runner
    elif quantization == "mxfp8":
        return _FlashInferMXFP8Runner
    elif quantization == "nvfp4":
        return _FlashInferNVFP4Runner
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no runner branch"
        )


def _flashinfer_moe_description(quantization: str) -> str:
    """Return the explicit log description for one supported quantization."""

    if quantization == "bf16":
        return "BF16 weights and activations"
    elif quantization == "mxfp8":
        return "MXFP8 weights and per-token activations"
    elif quantization == "nvfp4":
        return (
            f"per-token NVFP4, 4-over-6="
            f"{os.environ.get('FLASHINFER_NVFP4_4OVER6', '0')}, "
            f"E4M3={int(_FlashInferNVFP4Runner._e4m3_max())}"
        )
    else:
        raise NotImplementedError(
            f"FlashInfer MoE quantization {quantization!r} has no log-description branch"
        )


class _FlashInferForwardBF16Backward(torch.autograd.Function):
    """Exact FlashInfer forward with selectable BF16 surrogate operands."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        topk_weights,
        topk_ids,
        tokens_per_expert,
        runner,
        weight_key,
        backward_mode,
        activation_in_fp32,
        fused_activation,
        *expert_weights,
    ):
        ctx.runner = runner
        ctx.activation_in_fp32 = activation_in_fp32
        ctx.fused_activation = fused_activation
        num_local_experts = runner.local_num_experts
        if len(expert_weights) != 2 * num_local_experts:
            raise ValueError(
                "FlashInfer MoE expected two weights per local expert, got "
                f"{len(expert_weights)} for {num_local_experts} experts"
            )
        ctx.num_local_experts = num_local_experts
        ctx.tokens_per_expert = tuple(int(count) for count in tokens_per_expert)
        if len(ctx.tokens_per_expert) != num_local_experts:
            raise ValueError(
                "FlashInfer MoE expected one token count per local expert, got "
                f"{len(ctx.tokens_per_expert)} for {num_local_experts} experts"
            )
        ctx.saved_quantized_operands = False
        w13_gate_up = expert_weights[:num_local_experts]
        w2 = expert_weights[num_local_experts:]
        if hidden_states.shape[0] == 0:
            # FlashInfer's routed launchers do not accept M=0. Keep this rank in
            # the custom-autograd graph so backward still emits explicit zero
            # gradients for inputs and locally owned expert weights.
            ctx.save_for_backward(hidden_states, topk_weights, *expert_weights)
            return torch.empty_like(hidden_states)
        result = runner.forward(
            hidden_states, topk_weights, topk_ids, w13_gate_up, w2, weight_key, backward_mode
        )
        if backward_mode == HIGH_PRECISION_BACKWARD:
            ctx.save_for_backward(hidden_states, topk_weights, *expert_weights)
        elif backward_mode == DEQUANTIZED_BACKWARD:
            with _flashinfer_nvtx_range("qdq_save"):
                backward_hidden_states = result.backward_hidden_states
                backward_w13 = result.backward_w13
                backward_w2 = result.backward_w2
                if any(
                    tensor is None for tensor in (backward_hidden_states, backward_w13, backward_w2)
                ):
                    raise RuntimeError(
                        "FlashInfer MoE dequantized backward is missing a forward operand"
                    )
                ctx.backward_dtype = hidden_states.dtype
                quantized_operands = (
                    _quantized_storage_shell(backward_hidden_states),
                    topk_weights,
                )
                (tensors_to_save, tensor_objects) = _te_quantized_tensor().prepare_for_saving(
                    *quantized_operands
                )
                ctx.quantized_tensor_count = len(tensors_to_save)
                ctx.save_for_backward(*tensors_to_save, *backward_w13, *backward_w2)
                ctx.tensor_objects = tensor_objects
                ctx.saved_quantized_operands = True
        else:
            raise ValueError(f"Unsupported FlashInfer MoE backward mode {backward_mode!r}")
        return result.output

    @staticmethod
    def backward(ctx, grad_output):
        with _flashinfer_nvtx_range("surrogate_backward"):
            return _FlashInferForwardBF16Backward._backward(ctx, grad_output)

    @staticmethod
    def _backward(ctx, grad_output):
        # Forward mirrors are not backward operands. Release them before the
        # BF16 grouped replay to reduce its live-memory peak and ensure a
        # failed replay cannot leave optimizer-stale prepared weights cached.
        with _flashinfer_nvtx_range("backward_operand_restore"):
            ctx.runner.invalidate_weights()
            if ctx.saved_quantized_operands:
                with _flashinfer_nvtx_range("qdq_restore"):
                    tensor_objects = [
                        None if tensor_object is None else _quantized_storage_shell(tensor_object)
                        for tensor_object in ctx.tensor_objects
                    ]
                    restored = _te_quantized_tensor().restore_from_saved(
                        tensor_objects, ctx.saved_tensors[: ctx.quantized_tensor_count]
                    )
                    hidden_storage, topk_weights = restored
                    hidden_states = hidden_storage.dequantize(dtype=ctx.backward_dtype)
                    expert_weights = ctx.saved_tensors[ctx.quantized_tensor_count :]
            else:
                hidden_states, topk_weights, *expert_weights = ctx.saved_tensors
            needs = ctx.needs_input_grad
            weight_needs = needs[9:]
        if hidden_states.shape[0] == 0:
            # Avoid launching a full BF16 expert recompute just to manufacture
            # explicit zeros on an EP rank with no dispatched assignments.
            return (
                torch.zeros_like(hidden_states) if needs[0] else None,
                torch.zeros_like(topk_weights) if needs[1] else None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                *(
                    torch.zeros_like(weight) if need else None
                    for weight, need in zip(expert_weights, weight_needs)
                ),
            )

        with torch.enable_grad():
            with _flashinfer_nvtx_range("surrogate_input_prep"):
                hidden_ref = hidden_states.detach().requires_grad_(needs[0])
                weights_ref = topk_weights.detach().requires_grad_(needs[1])
                w13_needs = weight_needs[: ctx.num_local_experts]
                w2_needs = weight_needs[ctx.num_local_experts :]
                w13_ref = tuple(
                    weight.detach().requires_grad_(any(w13_needs))
                    for weight in expert_weights[: ctx.num_local_experts]
                )
                w2_ref = tuple(
                    weight.detach().requires_grad_(any(w2_needs))
                    for weight in expert_weights[ctx.num_local_experts :]
                )
            with _flashinfer_nvtx_range("surrogate_recompute"):
                output_ref = ctx.runner.bf16_surrogate(
                    hidden_ref,
                    weights_ref,
                    w13_ref,
                    w2_ref,
                    ctx.tokens_per_expert,
                    activation_in_fp32=ctx.activation_in_fp32,
                    fused_activation=ctx.fused_activation,
                    dequantized_backward=ctx.saved_quantized_operands,
                )

        expert_weight_refs = (*w13_ref, *w2_ref)
        grad_inputs = (hidden_ref, weights_ref, *expert_weight_refs)
        requested_mask = (needs[0], needs[1], *weight_needs)
        requested = [tensor for tensor, need in zip(grad_inputs, requested_mask) if need]
        with _flashinfer_nvtx_range("surrogate_autograd_grad"):
            computed = torch.autograd.grad(output_ref, requested, grad_output, allow_unused=True)
        computed_iter = iter(computed)
        grads = []
        for tensor, need in zip(grad_inputs, requested_mask):
            if not need:
                grads.append(None)
                continue
            grad = next(computed_iter)
            grads.append(torch.zeros_like(tensor) if grad is None else grad)
        return grads[0], grads[1], None, None, None, None, None, None, None, *grads[2:]


def _run_flashinfer_forward_with_surrogate(
    hidden_states,
    topk_weights,
    topk_ids,
    tokens_per_expert,
    w13_gate_up,
    w2,
    runner,
    weight_key,
    backward_mode,
    *,
    activation_in_fp32=False,
    fused_activation=False,
):
    """Attach custom autograd only when this invocation can require backward."""

    expert_weights = (*w13_gate_up, *w2)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (hidden_states, topk_weights, *expert_weights)
    )
    if needs_backward:
        return _FlashInferForwardBF16Backward.apply(
            hidden_states,
            topk_weights,
            topk_ids,
            tokens_per_expert,
            runner,
            weight_key,
            backward_mode,
            activation_in_fp32,
            fused_activation,
            *expert_weights,
        )

    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)

    # The output is independent of the backward operand mode. Avoid retaining
    # backward-only QDQ buffers when no backward can consume them.
    return runner.forward(
        hidden_states, topk_weights, topk_ids, w13_gate_up, w2, weight_key, HIGH_PRECISION_BACKWARD
    ).output


def _get_flashinfer_runner(experts: FlashInferGroupedMLP, quantization: str, device: torch.device):
    """Return a cached runner for one module's contiguous local expert shard."""

    runner = experts._flashinfer_moe_runner
    if runner is not None and runner.quantization == quantization:
        return runner

    if torch.cuda.get_device_capability(device)[0] < 10:
        raise RuntimeError("FlashInfer routed MoE requires NVIDIA Blackwell (SM100+)")
    runner_type = _flashinfer_moe_runner_type(quantization)
    if any(
        weight.dtype != torch.bfloat16
        for linear in (experts.linear_fc1, experts.linear_fc2)
        for weight in linear.parameters()
    ):
        raise TypeError("FlashInfer MoE master weights must remain BF16")
    runner = runner_type(
        num_experts=experts.config.num_moe_experts,
        local_expert_offset=get_pg_rank(experts.ep_group) * experts.num_local_experts,
        local_num_experts=experts.num_local_experts,
        hidden_size=experts.config.hidden_size,
        intermediate_size=experts.config.moe_ffn_hidden_size,
    )
    experts._flashinfer_moe_runner = runner
    return runner


def _run_dispatched_flashinfer_moe(
    experts: FlashInferGroupedMLP,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """Run local experts after Megatron's native dispatcher has sorted assignments."""

    with _flashinfer_nvtx_range("input_prep"):
        dispatch_mode = experts._flashinfer_moe_dispatch_mode
        execution_precision = _flashinfer_moe_execution_precision(experts)
        requested_backward_mode = experts._flashinfer_moe_backward_mode
        backward_mode = _flashinfer_moe_effective_backward_mode(
            execution_precision, requested_backward_mode
        )
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"FlashInfer MoE expects BF16 hidden states, got {hidden_states.dtype}")
        if not hidden_states.is_cuda:
            raise RuntimeError("FlashInfer routed MoE requires NVIDIA Blackwell (SM100+)")
        runner = _get_flashinfer_runner(experts, execution_precision, hidden_states.device)
        topk_weights, topk_ids = _dispatched_topk_inputs(
            permuted_probs,
            tokens_per_expert,
            local_expert_offset=runner.local_expert_offset,
            num_local_experts=experts.num_local_experts,
        )
        if hidden_states.shape[0] != topk_weights.shape[0]:
            raise ValueError(
                "FlashInfer MoE dispatched hidden/probability row counts disagree: "
                f"{hidden_states.shape[0]} != {topk_weights.shape[0]}"
            )

        token_counts = tuple(int(count) for count in tokens_per_expert.tolist())
        w13_gate_up, w2 = _grouped_mlp_weight_parameters(experts)
        weight_key = _source_weight_key(w13_gate_up, w2)
    output = _run_flashinfer_forward_with_surrogate(
        hidden_states,
        topk_weights,
        topk_ids,
        token_counts,
        w13_gate_up,
        w2,
        runner,
        weight_key,
        backward_mode,
        activation_in_fp32=experts._flashinfer_moe_activation_in_fp32,
        fused_activation=experts._flashinfer_moe_fused_activation,
    )
    if output.shape != hidden_states.shape:
        raise RuntimeError(
            "FlashInfer MoE local output shape mismatch: "
            f"{tuple(output.shape)} != {tuple(hidden_states.shape)}"
        )

    log_key = (execution_precision, dispatch_mode, requested_backward_mode)
    if log_key not in _LOGGED_LAYERS:
        log_single_rank(
            logger,
            logging.WARNING,
            "FlashInfer MoE path active: Megatron %s dispatch, routed TRT-LLM "
            "top-k=1 local assignments, %s, %s BF16 surrogate backward "
            "(%s=1, %s=%s)",
            dispatch_mode,
            _flashinfer_moe_description(execution_precision),
            backward_mode,
            _ENV,
            _BACKWARD_OVERRIDE_ENV,
            requested_backward_mode,
        )
        _LOGGED_LAYERS.add(log_key)

    return output, None


__all__ = [
    "FlashInferGroupedMLP",
    "flashinfer_moe_dispatch_mode",
    "maybe_replace_flashinfer_moe_expert_spec",
    "use_flashinfer_moe",
]
