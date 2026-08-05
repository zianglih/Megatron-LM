# FlashInfer routed MoE training

This extension keeps Megatron BF16 parameters, checkpoints, routing, token
dispatch, and token combine as the source of truth while replacing the local
expert forward with a FlashInfer low-precision routed-MoE kernel. It generalizes
the FP32-activation work in
[radixark/Megatron-LM#68](https://github.com/radixark/Megatron-LM/pull/68)
to model-neutral NVFP4 and MXFP8 expert execution.

The implementation lives in `miles_megatron_plugins/flashinfer_moe.py`. The only
Megatron-core integration is expert-spec selection during `MoELayer`
construction.

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

The override must agree with an active `fp4=nvfp4` or `fp8=mxfp8` recipe. There
is no implicit NVFP4 fallback.

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
names, initialization, and distributed-checkpoint mapping remain owned by
Megatron. Only `forward` and delayed-weight-gradient handling are replaced.

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

Both backward modes recompute a BF16 SwiGLU expert graph and differentiate that
surrogate:

- `high_precision` saves references to the original BF16 activation and master
  parameters.
- `dequantized` saves the compact forward activation payload and eagerly decodes
  each weight immediately after it is quantized. The decoded per-expert weights
  are shared by outstanding forward graphs, and the temporary canonical
  quantized weight storage is released expert by expert. Activation decode is
  deferred until its backward to limit memory growth with pipeline overlap.

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
particular, flex dispatch (`deepep` or `hybridep`), ETP greater than one, shared
expert overlap, latent MoE projections, expert bias, non-SwiGLU activations,
capacity/drop routing, FP32 combine, and unknown quantization recipes are not
silently redirected to another implementation.

## Validation

The distributed numerical test is parameterized over:

- 8 and 4,096 input tokens;
- 32 experts, hidden size 7,168, intermediate size 2,048, router top-k 8;
- NVFP4 and MXFP8; and
- Megatron all-to-all and all-gather.

On one 8xB200 bare devbox, all eight distributed cases passed. The fused
forward was compared with a distributed BF16 reference, and both backward modes
were compared with their corresponding ordinary-autograd BF16 surrogate. The
worst observed forward relative L2 was below 0.215 for NVFP4 and 0.071 for
MXFP8; surrogate hidden-gradient relative L2 was below 0.004, with router and
parameter-gradient comparisons matching at the reported precision.

The focused single-process suite passed 38 tests with 8 distributed cases
skipped outside torchrun:

```bash
python3 -m pytest -q tests/unit_tests/extension/test_flashinfer_moe.py

NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m pytest -q -x tests/unit_tests/extension/test_flashinfer_moe.py \
  -k test_flashinfer_routed_forward_and_surrogate_backward
```

The non-gating profile in
`tests/manual_tests/flashinfer_moe_backward_profile.py` reports end-to-end step
latency, peak allocated memory, post-forward live memory, and peak memory with
one and four outstanding forward graphs for every dispatcher/quantization pair.

One B200 sample at 4,096 tokens produced:

| Dispatcher | Quantization | High-precision ms | Dequantized ms | 1-graph peak delta MiB | 4-graph live delta MiB | 4-graph peak delta MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all-to-all | NVFP4 | 105.05 | 112.01 | +480.18 | -1,135.45 | -470.87 |
| all-to-all | MXFP8 | 99.48 | 99.66 | +600.03 | -656.01 | -143.99 |
| all-gather | NVFP4 | 117.65 | 119.51 | +480.07 | -1,134.70 | -526.73 |
| all-gather | MXFP8 | 127.51 | 119.32 | +600.03 | -655.99 | -143.99 |

The dequantized mode pays for one cached BF16 QDQ weight shard, so its
single-graph peak is higher. With four outstanding graphs, retaining compact
activation payloads instead of BF16 dispatched activations outweighed that
fixed cost in every measured case. These are non-gating single-run profile
numbers, not a performance guarantee.
