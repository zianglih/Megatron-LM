# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grouped BF16 surrogate used by the FlashInfer routed-MoE autograd boundary."""

from __future__ import annotations

import threading
from typing import Sequence

import torch


class _TEBF16GroupedLinear:
    """Storage-free TE GroupedLinear with functional expert weights."""

    def __init__(
        self, *, num_gemms: int, in_features: int, out_features: int, device: torch.device
    ) -> None:
        import transformer_engine.pytorch as te

        self.num_gemms = num_gemms
        self.device = torch.device(device)
        self._call_lock = threading.RLock()
        with te.autocast(enabled=False):
            self.op = te.GroupedLinear(
                num_gemms=num_gemms,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                return_bias=False,
                params_dtype=torch.bfloat16,
                device="meta",
                tp_group=None,
                tp_size=1,
                parallel_mode=None,
                sequence_parallel=False,
                fuse_wgrad_accumulation=False,
                delay_wgrad_compute=False,
                save_original_input=False,
            )

        if getattr(self.op, "primary_weights_in_fp8", False):
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
        # TE keeps bias placeholders even with bias=False and saves them in its
        # autograd context. Keep those zero-sized tensors off the meta device.
        for index in range(num_gemms):
            setattr(
                self.op, f"bias{index}", torch.empty(0, dtype=torch.bfloat16, device=self.device)
            )

    def __call__(
        self, hidden_states: torch.Tensor, splits: Sequence[int], weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        import transformer_engine.pytorch as te

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
        if len(splits) != self.num_gemms or any(not isinstance(split, int) for split in splits):
            raise ValueError("FlashInfer MoE BF16 surrogate requires one integer split per expert")

        functional_weights = {f"weight{index}": weight for index, weight in enumerate(weights)}
        with self._call_lock, te.autocast(enabled=False):
            return torch.func.functional_call(
                self.op,
                functional_weights,
                args=(hidden_states, splits),
                kwargs={"is_first_microbatch": None},
                strict=True,
            )


class _BF16GroupedMLPSurrogate:
    """Two grouped BF16 GEMMs with Megatron's routed SwiGLU between them."""

    def __init__(self, *, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._initialize_lock = threading.RLock()
        self._device: torch.device | None = None
        self._fc1: _TEBF16GroupedLinear | None = None
        self._fc2: _TEBF16GroupedLinear | None = None

    def _initialize(self, device: torch.device) -> None:
        device = torch.device(device)
        with self._initialize_lock:
            if self._device is not None:
                if device != self._device:
                    raise ValueError(
                        "FlashInfer MoE BF16 surrogate device changed from "
                        f"{self._device} to {device}"
                    )
                return
            self._fc1 = _TEBF16GroupedLinear(
                num_gemms=self.num_experts,
                in_features=self.hidden_size,
                out_features=2 * self.intermediate_size,
                device=device,
            )
            self._fc2 = _TEBF16GroupedLinear(
                num_gemms=self.num_experts,
                in_features=self.intermediate_size,
                out_features=self.hidden_size,
                device=device,
            )
            self._device = device

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

        self._initialize(hidden_states.device)
        assert self._fc1 is not None and self._fc2 is not None
        # The repository-pinned TE accepts Python split lists; newer TE releases
        # retain that API and convert it to host metadata without a device sync.
        splits = list(tokens_per_expert)
        fc1_output = self._fc1(hidden_states, splits, w13_gate_up)
        if activation_in_fp32:
            from megatron.core.transformer.moe.experts import _MoEActivationInFP32

            activated = _MoEActivationInFP32.apply(fc1_output, topk_weights, 0.0)
        elif fused_activation:
            from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

            activated = weighted_bias_swiglu_impl(
                fc1_output, None, topk_weights, fp8_input_store=False
            )
        else:
            gate, up = fc1_output.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate) * up * topk_weights).to(fc1_output.dtype)
        return self._fc2(activated, splits, w2)
