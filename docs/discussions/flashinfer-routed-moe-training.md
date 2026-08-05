# FlashInfer routed MoE training

This extension keeps Megatron BF16 parameters, checkpoints, routing, token
dispatch, and token combine as the source of truth while replacing the local
routed-expert forward with a model-neutral FlashInfer BF16, MXFP8, or NVFP4
kernel. It composes with the reusable FP32 MoE activation option introduced in
[radixark/Megatron-LM#68](https://github.com/radixark/Megatron-LM/pull/68), but
does not require that option.

FlashInfer forward integration, its custom-autograd boundary, and the BF16
replay live together in `miles_megatron_plugins/flashinfer_moe.py`. The plugin
delegates grouped-GEMM and activation backward logic to Transformer Engine and
Megatron rather than reproducing either implementation.

## Enablement

Set:

```bash
export MILES_USE_FLASHINFER_MOE=1
```

Select exactly one quantization through the canonical Megatron precision
arguments:

```text
MXFP8: --fp8-format e4m3 --fp8-recipe mxfp8
NVFP4: --fp4-format e2m1 --fp4-recipe nvfp4
```

The plugin infers the model's quantized runner from the active format and
recipe. At execution time it follows Transformer Engine's precision decision
for the routed-expert FC1 and FC2 modules: a quantized layer uses that runner,
while a layer that TE keeps in BF16 uses FlashInfer's BF16 routed kernel. FC1
and FC2 must agree on the execution precision. There is no implicit NVFP4
fallback, and unsupported or simultaneous recipes are rejected explicitly.

This directly supports Megatron's first/last-layer BF16 controls, for example:

```text
--first-last-layers-bf16
--num-layers-at-start-in-bf16 N
--num-layers-at-end-in-bf16 N
```

Megatron's layer-scoped FP8 or FP4 context remains the source of truth, so this
works across pipeline and virtual-pipeline placement without a second layer
index calculation in the plugin.

Select the surrogate operands explicitly with the same Transformer Engine
setting used by the rest of the model:

```bash
export NVTE_BACKWARD_OVERRIDE=high_precision
# or
export NVTE_BACKWARD_OVERRIDE=dequantized
```

`high_precision` uses the original BF16 operands, while `dequantized` uses
BF16 values decoded from the exact forward quantized operands. The setting is
required; unset, empty, and unsupported values are rejected. This follows
[NVIDIA/TransformerEngine#2644](https://github.com/NVIDIA/TransformerEngine/pull/2644).
When TE selects BF16 for a routed layer, there are no quantized operands to
decode, so a model-wide `dequantized` setting uses the `high_precision`
surrogate for that layer. Quantized layers continue to use dequantized forward
operands.

## Dispatch and expert ownership

Both supported dispatchers use Megatron's normal lifecycle:

```text
router -> Megatron dispatch -> FlashInfer local experts -> Megatron combine
```

`allgather` uses `MoEAllGatherTokenDispatcher` unchanged. `alltoall` uses
`MoEAlltoAllTokenDispatcher` unchanged. The plugin does not implement a
collective, reconstruct routing, pad ranks, or reproduce the MoE layer forward.

Megatron hands the expert module rows already sorted by local expert, their
`tokens_per_expert`, and the corresponding differentiable routing
probabilities. The plugin converts those rows to a routed-kernel top-k of one;
the model router can still use top-k greater than one because each assignment
has already been expanded by the dispatcher.

The replacement is deliberately routed-expert-only. The existing
`shared_experts` spec is preserved without copying or replacement, and a
FlashInfer grouped routed-expert module is rejected in that slot. Shared-expert
construction, execution, and precision therefore remain native Megatron/TE
behavior; model recipes that keep the shared expert in BF16 continue to do so.

The routed expert class inherits Megatron's `TEGroupedMLP`, so parameter
creation, names, initialization, optimizer ownership, and
distributed-checkpoint mapping remain owned by Megatron. The surrogate's raw
Transformer Engine operators are storage-free helpers rather than registered
expert modules, and gradients from the custom-autograd boundary are returned
to the original Megatron parameters.

## Forward and backward

For each local routed-expert shard, the quantized forward path:

1. reads the existing per-expert BF16 `weightN` parameters without stacking a
   second BF16 copy;
2. quantizes through Transformer Engine using the same rowwise contracts as
   Miles weight sync;
3. adapts gate/up order and FlashInfer's shuffled weight layouts; and
4. invokes the explicit MXFP8 or NVFP4 TRT-LLM routed kernel.

When TE keeps the routed layer in BF16, the same BF16 master weights are
materialized in FlashInfer's block-major layout and passed to its BF16 TRT-LLM
routed kernel. This path is selected from TE's live execution context rather
than a separate FlashInfer precision flag.

Prepared quantized weights, or the BF16 block-major mirror, are cached until
backward starts. Backward releases that forward-only mirror before the grouped
BF16 replay, reducing its live-memory peak and ensuring optimizer updates
through `param.data` cannot leave a stale mirror for the next forward.

Both effective backward modes recompute a grouped BF16 SwiGLU expert graph and
differentiate that surrogate:

- `high_precision` saves references to the original BF16 activation and master
  parameters.
- `dequantized` saves the compact forward activation payload and reuses BF16
  weights decoded from the quantized operands produced for that forward. It
  does not replace or mutate the BF16 master parameters.

For BF16 routed execution, both model-wide settings use the original BF16
operands because that forward produces no quantized payload.

The replay consists of a raw Transformer Engine `GroupedLinear` for FC1, the
routed activation, and a second raw
`GroupedLinear` for FC2. The operators are constructed without persistent
weight storage, and `torch.func.functional_call` supplies either the original
or QDQ-selected per-expert weights. Transformer Engine therefore owns the
grouped forward, dgrad, and wgrad GEMMs; there is no Python loop over experts.

The FP32-activation path reuses Megatron's `_MoEActivationInFP32`, and the
fused-activation BF16 path reuses Megatron's weighted SwiGLU helper. The replay
does not use Transformer Engine `LayerNormMLP`: dispatched expert rows require
only the two grouped linear operations and the intervening routed activation,
and there is no expert-local layer normalization to reproduce.

Token counts are materialized once as a CPU `int64` split tensor and reused by
both grouped linears, keeping split metadata on the host without duplicate
conversion.

The surrogate is intentionally not the mathematical derivative of the fused
quantized forward. The choice only controls which forward operands seed the
BF16 recomputation.

## Supported surface

- NVIDIA Blackwell (SM100 or newer)
- BF16 master parameters and BF16 dispatched hidden states
- BF16 routed execution selected by TE for first/last layers in MXFP8 and NVFP4
  models
- gated SwiGLU without bias, clamp, or linear offset
- expert parallelism with `expert_tensor_parallel_size=1`
- Megatron `allgather` and `alltoall` token dispatchers
- dropless routing without router quantization padding
- up to 2,048 global experts

The extension rejects unsupported branches before the kernel launch. In
particular, delayed expert-weight gradients, Transformer Engine activation
modules (`use_te_activation_func`), ETP greater than one, and flex dispatch
backends such as DeepEP and HybridEP have no fallback branch. Shared-expert
overlap, expert activation offloading, latent MoE projections, expert bias,
non-SwiGLU activations, capacity/drop routing, FP32 combine, and unknown
quantization recipes are also rejected rather than silently redirected to
another implementation.

## Validation

The focused tests compare the grouped surrogate with an independent BF16
expert reference, including empty local experts, the supported activation
paths, shared-expert preservation, and TE-driven runner selection. The
distributed numerical test exercises both quantizations, BF16 boundary-layer
execution, both Megatron dispatchers, and both backward operand settings.

Run the focused suite with:

```bash
python3 -m pytest -q tests/unit_tests/extension/test_flashinfer_moe.py

NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m pytest -q -x tests/unit_tests/extension/test_flashinfer_moe.py \
  -k test_flashinfer_routed_forward_and_surrogate_backward
```

`tests/manual_tests/flashinfer_moe_backward_profile.py` can measure end-to-end
step latency and CUDA memory for the dispatcher, quantization, backward-mode,
and outstanding-forward configuration under review. Its existing MXFP8 and
NVFP4 cases explicitly enter Megatron's layer-scoped quantization context, so
they measure quantized execution rather than the BF16 boundary runner. No
single-run performance or memory result is treated as a guarantee in this
design discussion.
