# FlashInfer routed MoE in Megatron training

## Goal

Evaluate a stronger rollout-parity boundary than matching activation functions
individually: use the same FlashInfer routed MoE forward in Megatron that SGLang
uses for rollout, while retaining ordinary BF16 Megatron parameters and an
explicitly approximate training backward.

This experiment targets:

- `trtllm_fp4_block_scale_routed_moe`
- per-token NVFP4 activation quantization
- NVFP4 4-over-6 weight and activation quantization
- `trtllm_fp8_block_scale_routed_moe`
- MXFP8 weights and per-token MXFP8 activation quantization
- routed SwiGLU experts
- BF16 module input and output
- expert tensor parallel size 1

It does not introduce a weight or activation layout. It materializes the
layouts already required by the FlashInfer TRT-LLM kernel.

## Design

The opt-in environment variable is:

```bash
MILES_USE_FLASHINFER_MOE=1
```

It defaults to disabled. When enabled, the adapter requires exactly one
supported quantization. It resolves an active Megatron MXFP8 or NVFP4 recipe,
or experiments can select either path explicitly:

```bash
MILES_FLASHINFER_MOE_QUANTIZATION=nvfp4  # or mxfp8
```

There is no default quantization fallback. Other active recipes and missing or
ambiguous selections fail before the routed kernel runs, and an environment
selection cannot override a conflicting active recipe. New formats must add an
explicit resolver, capability, validation, runner, and log-description branch.

When enabled, the MoE layer:

1. Replaces `TEGroupedMLP` with a BF16 holder that preserves its per-expert
   parameter naming contract during model construction. The concrete
   `FlashInferGroupedMLP` is a plain per-expert parameter holder with
   `linear_fc1.weightN` and `linear_fc2.weightN` parameters; it does not depend
   on TE or the `grouped_gemm` package. Checkpoint, optimizer, and Miles raw
   weight-sync ownership remain in Megatron.
2. Runs the normal Megatron router, including Miles rollout-routing replay.
   When replay is active, the adapter reads the current replay stream's original
   top-k IDs so the FlashInfer packed input preserves rollout slot order; routing
   weights remain differentiable gathers from Megatron's dense probabilities.
3. Gathers BF16 tokens, routing weights, and routing IDs across the expert
   parallel group.
4. Quantizes each current BF16 expert with the matching direct TE quantizer so
   the values and scales match Miles weight sync. NVFP4 uses per-tensor decode
   scales; MXFP8 uses compact rowwise UE8M0 scales. Both paths adapt Megatron's
   `[gate, up]` rows to FlashInfer's W3/W1 contract and apply the same
   format-specific weight/scale layout transforms used by SGLang.
5. Quantizes activations per token with FlashInfer in the linear scale layout.
6. Calls the pre-routed FlashInfer NVFP4 or MXFP8 operation with packed BF16
   routing weights, local expert offset/count, `do_finalize=True`, and SwiGLU.
7. Sums local-expert partial outputs across expert parallel ranks and slices
   this rank's original tokens.

FlashInfer does not expose autograd for this fused forward. The custom autograd
bridge therefore recomputes a whole local routed-expert module in BF16 during
backward and obtains gradients with `torch.autograd.grad`. The surrogate uses
Megatron's `[gate, up]`, `silu(gate) * up`, probability-before-FC2 formula.
It returns gradients for BF16 input, routing weights, and BF16 master weights.
Expert IDs and forward quantization are intentionally nondifferentiable.

The expert-parallel communication remains part of the autograd boundary:

- padded all-gather backward is a summing reduce-scatter for input and router
  gradients;
- output all-reduce backward is another sum;
- local expert parameters receive only their local gradients.

This backward is not the derivative of the quantized FlashInfer forward.
Gradient fidelity is an accepted limitation of this experiment.

## Exact NVFP4 rollout settings

The controlled GLM-5.2 experiment uses:

```bash
NVTE_NVFP4_4OVER6=all
NVTE_NVFP4_4OVER6_E4M3_USE_256=all
NVTE_NVFP4_4OVER6_ERR_MODE=MSE
NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH=0
NVTE_USE_FAST_MATH=0
FLASHINFER_NVFP4_4OVER6=1
FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1
FLASHINFER_NVFP4_4OVER6_ERR_MODE=MSE
FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH=0
FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1
SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1
TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
```

The `NVTE_*` settings own per-tensor weight quantization. The
`FLASHINFER_*` settings own per-token activation quantization. The module does
not enter TE autocast, a TE recipe, or a global TE quantization context.

The initial comparison used SGLang `flashinfer_trtllm_routed`, SGL-kernel DSA
top-k, Miles Torch top-k, and fixed rollout-routing replay. The activation-only
toggle is disabled on both sides because this path bypasses Megatron's
routed-expert activation implementation. A later controlled run using
FlashInfer DSA top-k on both sides is recorded below.

The MXFP8 path instead uses TE's rowwise `MXFP8Quantizer` for BF16 master
weights and FlashInfer's `mxfp8_quantize` for per-token activations. The kernel
receives token-major activation scales, shuffled W3/W1 and W2 weights, and
interleaved UE8M0 weight scales, matching SGLang's
`flashinfer_trtllm_routed` backend.

## Environment

Local workspace:

```text
/Users/ziangli/playground/cute-dsl-nvfp4
```

Megatron branch and starting revision:

```text
agent-flashinfer-fast-swiglu-megatron
5096ccd6c
```

Required remote image and hardware:

```text
radixark/miles:dev-202607281246
queue: hell
8 x NVIDIA B200
```

No FlashInfer source refresh or modification is required. The requested image
initially provided FlashInfer 0.6.12 and was later upgraded to the published
0.6.13 wheels without a source checkout or build. Miles serializes the rollout
weights with TE, while SGLang uses FlashInfer's CUDA NVFP4 quantizer for
per-token activations. The adapter mirrors that division of ownership.

## Initial limitations

- The quantized weight mirror uses a parameter storage/version cache. The
  surrogate backward invalidates it before the optimizer can update BF16 master
  weights, because Megatron optimizer copies through `param.data` do not
  necessarily increment `Parameter._version`. A production implementation
  should refresh persistent buffers in place instead of reallocating them.
- Without Miles routing replay, top-k slot order is recovered from Megatron's
  dense selected probabilities. Models whose correction bias changes selected
  expert ordering independently of routing weights need an explicit router
  top-k side channel for the non-replay case.
- FlashInfer tactic selection/cache parity with the rollout process is not yet
  controlled.
- A nonempty Megatron `padding_mask`, token dropping/capacity, shared-expert
  overlap, latent MoE, expert bias, SwiGLU clamp/offset, and expert tensor
  parallelism greater than one are rejected. Miles' all-`-1` replay padding
  rows are supported by applying the same deterministic expert-ID replacement
  as `ReplayManager`.
- The current MXFP8 path requires hidden and expert-intermediate dimensions to
  be multiples of 128. Shape padding can relax this in a later iteration.
- The trainer run used EP4 while rollout data came from EP2. Ordinary NCCL
  all-reduce does not preserve the same partial-sum association across those
  topologies, so reusing the FlashInfer expert kernel alone cannot guarantee
  bitwise module parity.
- The replay side channel assumes the router consumes a replay entry. Hash
  routing and MTP paths that bypass Miles replay are outside this experiment.
- The parameter names match Miles raw weight sync, but live SGLang weight
  transfer and optimizer-state resume have not yet been exercised.
- Partial MoE CUDA graph capture is rejected. Full-layer activation
  checkpointing may re-run the FlashInfer forward and must be measured.

## Validation log

Implementation and tests are intentionally left unstaged while this design is
evaluated.

The first parameter-holder attempt reused legacy `GroupedMLP`. The requested
image does not contain the optional `grouped_gemm` package, so that candidate
failed during model construction before checkpoint loading. The final design
uses the plain per-expert holder described above. It preserves the TE-compatible
checkpoint names without constructing TE or grouped-GEMM layers; TE is called
only as the standalone weight quantizer.

The initial remote environment was:

```text
devbox: flashinfer-moe-parity
image: radixark/miles:dev-202607281246
queue: hell
hardware: 8 x NVIDIA B200
PyTorch: 2.11.0+cu130
FlashInfer: 0.6.12
SGLang: 0.5.16.dev34+gf7ea06e
Miles: 5cc5ff96413b82a2d14befcde0376ece6e36dce2
Megatron base: 5096ccd6c
```

The local Megatron implementation, MoE hook, test, and Miles comparison script
were SHA-256 checked against `/root/Megatron-LM` and `/root/miles` immediately
before the run. No FlashInfer source was synchronized or rebuilt.

Targeted Blackwell validation:

```text
16 passed, 27 warnings in 1.39s
```

This includes the exact FlashInfer per-token NVFP4 + 4-over-6 forward, a
nonzero-loss BF16 surrogate backward with finite gradients, replay slot-order
preservation, replay padding normalization, and explicit zero gradients when an
EP rank has no locally routed tokens. It also bitwise-compares the local TE
helper with Miles for odd-row padding and paired gate/up quantization.

The MXFP8 extension was validated separately on the bare
`radixark/miles:dev-202608041247` image with FlashInfer 0.6.14 and an NVIDIA
B200. The full routed-MoE unit file passed without installing or rebuilding any
packages:

```text
24 passed, 25 warnings in 2.19s
```

This adds bitwise TE-versus-Miles MXFP8 weight and scale checks, explicit
selection/error coverage for NVFP4 and MXFP8, bitwise prepared-layout parity
with SGLang, and a routed SwiGLU MXFP8 forward with finite BF16 surrogate
gradients at 128 global experts, 4 local experts, hidden size 2048, and
intermediate size 768.

### Quantized-weight contract

Miles raw weight sync jointly quantizes gate/up BF16 tensors with TE's
`NVFP4Quantizer`, then SGLang consumes those serialized packed weights, block
scales, and per-tensor decode scales. A B200 audit of FlashInfer 0.6.13 and TE
2.17 under strict 4-over-6/MSE, E4M3 max 256, and disabled candidate fast math
confirmed that the producer contract differs by quantization mode:

```text
Per-token, (32, 128), 200 samples: 0 packed/scales/decode mismatches
Per-token, (128, 2048), 40 samples: 0 packed/scales/decode mismatches
Per-tensor, (32, 128), 200 samples: 2 samples mismatched
Per-tensor, (128, 2048), 40 samples: 11 samples mismatched
```

The small per-tensor case differed by 16 packed bytes and two block-scale bytes
across 51,200 blocks. The strict implementations are mathematically equivalent,
but their global-decode arithmetic ordering can change a near-tie by a few
FP32 ULPs. The module therefore uses FlashInfer only for aligned per-token
activation quantization and TE for per-tensor weights. FC1 is quantized once in
`[gate, up]` order before its packed data and scales are reordered to
`[up, gate]`; no new memory layout is introduced.

Miles' complete NVFP4 quantizer unit file also passed on the same environment:

```text
633 passed, 24 warnings in 6.59s
```

Audit artifacts:

```text
logs/nvfp4_te_fi_contract_20260729_191500/run.log
logs/miles_nvfp4_quantizer_unit_20260729_192000/run.log
logs/flashinfer_moe_te_weights_unit_final_20260729_184400/run.log
```

### Paired model run

This is the historical FlashInfer-0.6.12, FlashInfer-quantized-weight result
before the weight producer was corrected to match Miles.

The paired GLM-5.2 five-layer run used two byte-identical fixed rollout files,
TP4/EP4/ETP1, full activation recomputation, Miles routing replay, SGL-kernel
rollout top-k, Torch trainer top-k, and zero BF16 boundary layers. Both variants
loaded the original distributed checkpoint and completed both training
backwards. The one-time runtime marker confirmed the FlashInfer path was active
on both routed-MoE layers.

| Metric | Megatron mean | FlashInfer mean | Relative delta |
| --- | ---: | ---: | ---: |
| `train/train_rollout_logprob_abs_diff` | 0.0152137894 | 0.0152012068 | -0.083% |
| `train/train_rollout_kl` | 0.0002127190 | 0.0002114670 | -0.589% |
| `train/kl_loss` | 0.0001833101 | 0.0001843832 | +0.585% |

The result is numerically neutral at this sample size: it demonstrates that the
standalone FlashInfer-forward/BF16-backward module works with routing replay,
not that it closes rollout parity. The fixed data has zero advantages and the
run used `--debug-disable-optimizer`; model-level backward control flow ran, but
this comparison does not validate optimizer updates, live weight sync, or
quantized-cache invalidation after an update.

Local artifacts:

```text
logs/flashinfer_moe_unit_final_20260729_174500/run.log
logs/flashinfer_moe_weight_quant_audit_20260729_175100/run.log
logs/flashinfer_moe_numerical_final_20260729_174700/comparison/results.json
logs/flashinfer_moe_numerical_final_20260729_174700/comparison/flashinfer/run.log
logs/flashinfer_moe_numerical_final_20260729_174700/comparison/megatron/run.log
```

### FlashInfer 0.6.13 wheel rerun

This is the historical FlashInfer-quantized-weight rerun before the weight
producer was corrected to match Miles.

FlashInfer was replaced using only the published CUDA 13 wheels:

```bash
python3 -m pip uninstall -y \
    flashinfer flashinfer-python flashinfer-cubin flashinfer-jit-cache
python3 -m pip install \
    'flashinfer-python[cu13]==0.6.13' \
    'flashinfer-cubin==0.6.13'
python3 -m pip install 'flashinfer-jit-cache==0.6.13' \
    --index-url https://flashinfer.ai/whl/cu130
python3 -m pip install --no-deps nvidia-cudnn-cu13==9.22.0.52
flashinfer show-config
```

Preflight immediately before the rerun reported:

```text
FlashInfer Python: 0.6.13
FlashInfer cubin: 0.6.13
FlashInfer JIT cache: 0.6.13+cu130
PyTorch: 2.11.0+cu130
cuDNN package: 9.22.0.52
torch.backends.cudnn.version(): 92200
```

No FlashInfer repository was synchronized, installed editable, or built. The
same two fixed rollout files and the same experiment configuration were reused,
including strict 4-over-6 candidate scoring
(`NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH=0` and
`FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH=0`). Both variants completed both
backwards, and the one-time markers again confirmed the FlashInfer routed MoE
path on layers 4 and 5.

| Metric | Megatron mean | FlashInfer 0.6.13 mean | Relative delta |
| --- | ---: | ---: | ---: |
| `train/train_rollout_logprob_abs_diff` | 0.0152137894 | 0.0152615779 | +0.314% |
| `train/train_rollout_kl` | 0.0002127190 | 0.0002142042 | +0.698% |
| `train/kl_loss` | 0.0001833101 | 0.0001843008 | +0.540% |

This two-rollout comparison does not show a numerical improvement from the
FlashInfer-forward variant after the wheel upgrade. The sample remains too
small for a general quality conclusion, but all three tracked means moved
slightly away from the rollout baseline in this controlled replay. Quantizer
contract debugging is intentionally deferred.

Local artifacts:

```text
logs/flashinfer_0613_install_20260729_180000/run.log
logs/flashinfer_moe_numerical_0613_20260729_185000/run.log
logs/flashinfer_moe_numerical_0613_20260729_185000/comparison/results.json
logs/flashinfer_moe_numerical_0613_20260729_185000/comparison/flashinfer/run.log
logs/flashinfer_moe_numerical_0613_20260729_185000/comparison/megatron/run.log
```

### TE weight-quantizer rerun

The final prototype directly uses TE 2.17 for each expert's per-tensor weights,
keeps FlashInfer 0.6.13 for per-token activations and fused forward, and does
not use TE quantization context. The same two fixed rollout files, TP4/EP4/ETP1
topology, routing replay, zero BF16 boundary layers, and disabled optimizer were
used. Runtime markers confirmed the FlashInfer path on both routed layers, and
both forwards and surrogate backwards completed.

| Metric | Megatron mean | TE-weight FlashInfer mean | Relative delta |
| --- | ---: | ---: | ---: |
| `train/train_rollout_logprob_abs_diff` | 0.0152137894 | 0.0151186958 | -0.625% |
| `train/train_rollout_kl` | 0.0002127190 | 0.0002107913 | -0.906% |
| `train/kl_loss` | 0.0001833101 | 0.0001805730 | -1.493% |

All three tracked means improved in this paired replay. The sample is only two
rollouts, so this is evidence that the corrected weight contract helps this
controlled case, not a general quality conclusion. It still does not isolate
the remaining GEMM, expert-parallel reduction, permutation, or topology
differences.

Final artifacts:

```text
logs/flashinfer_moe_te_weights_numerical_final_20260729_184700/run.log
logs/flashinfer_moe_te_weights_numerical_final_20260729_184700/comparison/results.json
logs/flashinfer_moe_te_weights_numerical_final_20260729_184700/comparison/flashinfer/run.log
logs/flashinfer_moe_te_weights_numerical_final_20260729_184700/comparison/megatron/run.log
```

### FlashInfer DSA top-k with large tie-break

The general numerical script was changed to use FlashInfer DSA top-k for both
rollout and trainer:

```text
--sglang-dsa-topk-backend flashinfer
--miles-dsa-topk-backend flashinfer
SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=large
SGLANG_DSA_FUSE_TOPK=0
```

`large` maps to FlashInfer `tie_break=2` in both owning adapters. Disabling the
SGLang fused path makes rollout call the unfused `flashinfer.top_k`, avoiding
the known fused SGLang integration bug. A B200 preflight with eight equal
scores returned indices `[6, 7]` from both the Miles and SGLang adapters,
confirming that the large-index tie-break was active.

A fixed replay cannot exercise SGLang, so the experiment used three fresh
processes:

1. A live Megatron-MoE run generated two rollout files with SGLang FlashInfer
   top-k, large tie-break, and top-k fusion disabled.
2. The files were replayed through the ordinary Megatron MoE with Miles
   FlashInfer top-k.
3. The byte-identical files were replayed through the FlashInfer MoE with Miles
   FlashInfer top-k.

The two fixture SHA-256 values were:

```text
0.pt  85780c802ed12b91ea9ec73fdeec2666af9ee020accee3e7eca6daa9b20f9715
1.pt  5e2af94cc7cd80484894fb7edc00753111503029e47ee6b4c9bd3758bc94ae3a
```

Both trainer variants used TP4/EP4/ETP1, routing replay, zero BF16 boundary
layers, and a disabled optimizer. Indexer replay remained disabled so Miles
actually executed FlashInfer top-k. The candidate log emitted the one-time
runtime marker for both routed layers, confirming the FlashInfer routed
TRT-LLM, per-token NVFP4, 4-over-6 forward and BF16 surrogate backward.

| Metric | Megatron mean | FlashInfer mean | Relative delta |
| --- | ---: | ---: | ---: |
| `train/train_rollout_logprob_abs_diff` | 0.0150603754 | 0.0151270670 | +0.443% |
| `train/train_rollout_kl` | 0.0002101068 | 0.0002090524 | -0.502% |
| `train/kl_loss` | 0.0001882156 | 0.0001842414 | -2.112% |

This paired result is mixed: the log-probability absolute-difference mean
regressed slightly, while both KL means improved. The earlier SGL-kernel/Torch
top-k replay improved all three means by 0.625%, 0.906%, and 1.493%,
respectively. Those percentages are each valid within their own fixed fixture,
but comparing absolute means across the two experiments is only contextual:
the generated rollout tensors and top-k configuration differ, so the
cross-fixture movement cannot be attributed to FlashInfer MoE or top-k alone.

The run used FlashInfer 0.6.13 published CUDA 13 wheels, PyTorch 2.11.0+cu130,
and the same TE 2.17 per-tensor weight quantizer. Immediately before the paired
run, the numerical script and FlashInfer MoE implementation hashes were:

```text
tools/run_numerical_comparison.py
  d199d23f8792242b08ac42e4bdf9b9992cc3561512860b30d808d10670fc88d9
miles_megatron_plugins/flashinfer_moe.py
  2ac1a9786f5d9c1c230669addb7070bf93916ba7aad43cd62e81431a8f006765
```

Artifacts:

```text
logs/topk_large_rollout_fixture_20260729_191100/run.log
logs/topk_large_rollout_fixture_20260729_191100/fixture/megatron/dump_details/rollout_data/0.pt
logs/topk_large_rollout_fixture_20260729_191100/fixture/megatron/dump_details/rollout_data/1.pt
logs/flashinfer_moe_topk_large_numerical_20260729_191500/run.log
logs/flashinfer_moe_topk_large_numerical_20260729_191500/comparison/results.json
```
