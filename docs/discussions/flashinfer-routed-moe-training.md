# FlashInfer routed MoE training

This extension keeps Megatron BF16 parameters, checkpoints, routing, token
dispatch, and token combine as the source of truth while replacing the local
expert forward with a FlashInfer low-precision routed-MoE kernel. It generalizes
the FP32-activation work in
[radixark/Megatron-LM#68](https://github.com/radixark/Megatron-LM/pull/68)
to model-neutral NVFP4 and MXFP8 expert execution.

FlashInfer forward integration and its custom-autograd boundary live in
`miles_megatron_plugins/flashinfer_moe.py`. The BF16 replay is isolated in
`miles_megatron_plugins/flashinfer_moe_surrogate.py`, so the quantized-kernel
adapter does not need to reproduce grouped-GEMM or activation backward logic.

## Enablement

Set:

```bash
export MILES_USE_FLASHINFER_MOE=1
```

Select exactly one quantization through the active Megatron recipe or an
explicit override:

```bash
export MILES_FLASHINFER_MOE_QUANTIZATION=nvfp4  # or mxfp8
```

If an active `fp4=nvfp4` or `fp8=mxfp8` recipe and the override are both
present, they must agree. There is no implicit NVFP4 fallback.

Backward uses the original BF16 operands by default. To recompute with BF16
values decoded from the exact forward quantized operands, set:

```bash
export MILES_FLASHINFER_MOE_DEQUANTIZED=1
```

This is analogous to Transformer Engine's `NVTE_BACKWARD_OVERRIDE=dequantized`
mode from
[NVIDIA/TransformerEngine#2644](https://github.com/NVIDIA/TransformerEngine/pull/2644).

## Dispatch and expert ownership

Both supported dispatchers use Megatron's normal lifecycle:

```text
router -> Megatron dispatch -> FlashInfer local experts -> Megatron combine
```

`alltoall` uses `MoEAlltoAllTokenDispatcher` unchanged. `allgather` uses
`MoEAllGatherTokenDispatcher` unchanged. The plugin does not implement a
collective, reconstruct routing, pad ranks, or reproduce the MoE layer forward.

Megatron hands the expert module rows already sorted by local expert, their
`tokens_per_expert`, and the corresponding differentiable routing
probabilities. The plugin converts those rows to a routed-kernel top-k of one;
the model router can still use top-k greater than one because each assignment
has already been expanded by the dispatcher.

The expert class inherits Megatron's `TEGroupedMLP`, so parameter creation,
names, initialization, optimizer ownership, and distributed-checkpoint mapping
remain owned by Megatron. The surrogate's raw Transformer Engine operators are
storage-free helpers rather than registered expert modules, and gradients from
the custom-autograd boundary are returned to the original Megatron parameters.

## Forward and backward

For each local expert shard, the forward path:

1. reads the existing per-expert BF16 `weightN` parameters without stacking a
   second BF16 copy;
2. quantizes through Transformer Engine using the same rowwise contracts as
   Miles weight sync;
3. adapts gate/up order and FlashInfer's shuffled weight layouts; and
4. invokes the explicit NVFP4 or MXFP8 TRT-LLM routed kernel.

The quantized weights are cached until backward. The cache is conservatively
invalidated after every training backward because optimizer updates through
`param.data` are not guaranteed to advance a parameter version counter.

Both backward modes recompute a grouped BF16 SwiGLU expert graph and
differentiate that surrogate:

- `high_precision` saves references to the original BF16 activation and master
  parameters.
- `dequantized` saves the compact forward activation payload and reuses BF16
  weights decoded from the quantized operands produced for that forward. It
  does not replace or mutate the BF16 master parameters.

The replay in `flashinfer_moe_surrogate.py` consists of a raw Transformer Engine
`GroupedLinear` for FC1, the routed activation, and a second raw
`GroupedLinear` for FC2. The operators are constructed without persistent
weight storage, and `torch.func.functional_call` supplies either the original
or QDQ-selected per-expert weights. Transformer Engine therefore owns the
grouped forward, dgrad, and wgrad GEMMs; there is no Python loop over experts.

The FP32-activation path reuses Megatron's `_MoEActivationInFP32`, and the
fused-activation BF16 path reuses Megatron's weighted SwiGLU helper. The replay
does not use Transformer Engine `LayerNormMLP`: dispatched expert rows require
only the two grouped linear operations and the intervening routed activation, and there is
no expert-local layer normalization to reproduce.

Token counts are passed to both grouped linears as a Python split list. This is
the default path supported by the repository-pinned Transformer Engine revision
and keeps split metadata on the host.

The surrogate is intentionally not the mathematical derivative of the fused
quantized forward. The choice only controls which forward operands seed the
BF16 recomputation.

## Supported surface

- NVIDIA Blackwell (SM100 or newer)
- BF16 master parameters and BF16 dispatched hidden states
- NVFP4 and MXFP8
- gated SwiGLU without bias, clamp, or linear offset
- expert parallelism with `expert_tensor_parallel_size=1`
- Megatron `alltoall` and `allgather` token dispatchers
- dropless routing without router quantization padding
- up to 2,048 global experts

The extension rejects unsupported branches before the kernel launch. In
particular, delayed expert-weight gradients, Transformer Engine activation
modules (`use_te_activation_func`), ETP greater than one, and flex dispatch
backends such as DeepEP and HybridEP have no fallback branch. Shared-expert
overlap, latent MoE projections, expert bias, non-SwiGLU activations,
capacity/drop routing, FP32 combine, and unknown quantization recipes are also
rejected rather than silently redirected to another implementation.

## Validation

The focused tests compare the grouped surrogate with an independent BF16
expert reference, including empty local experts and the supported activation
paths. The distributed numerical test exercises both quantizations, both
Megatron dispatchers, and both backward operand modes.

On a bare 8xB200 `radixark/miles:dev-202608041247` devbox, the focused file
completed with 54 passed and 8 torchrun-only cases skipped. The eight-rank run
then passed all eight combinations of NVFP4/MXFP8, all-to-all/all-gather, and
8/4,096 input tokens at 32 experts, hidden size 7,168, intermediate size 2,048,
and router top-k 8. The largest observed forward relative L2 was 0.215 for
NVFP4 and 0.071 for MXFP8; every surrogate hidden-gradient relative L2 was
below 0.004.

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
and outstanding-forward configuration under review. No single-run performance
or memory result is treated as a guarantee in this design discussion.

One same-image, three-iteration A/B against the previous per-expert-loop commit
(`428ad508f`) produced the following non-gating result. The grouped revision
used the repository-compatible non-fused Transformer Engine `GroupedLinear`
path with `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0`; the loop baseline did
not use that Transformer Engine path.

| Dispatcher | Quantization | High precision, loop -> grouped (ms) | Dequantized, loop -> grouped (ms) | Grouped peak delta, high/dequantized (MiB) |
| --- | --- | ---: | ---: | ---: |
| all-to-all | NVFP4 | 105.209 -> 101.711 | 106.915 -> 104.042 | +36.97 / +35.50 |
| all-to-all | MXFP8 | 99.302 -> 97.623 | 100.082 -> 97.652 | +35.38 / +35.38 |
| all-gather | NVFP4 | 117.823 -> 114.781 | 120.059 -> 117.163 | +34.34 / +33.67 |
| all-gather | MXFP8 | 116.284 -> 114.661 | 119.156 -> 117.903 | +35.37 / +34.53 |

This sample shows a 1.1-3.3% end-to-end latency reduction with a roughly
34-37 MiB increase in peak workspace. It is a narrow regression/profile check,
not a performance guarantee.
