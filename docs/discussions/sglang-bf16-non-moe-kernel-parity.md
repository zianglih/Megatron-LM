# SGLang BF16 non-MoE kernel parity experiments

This document is the live experiment ledger for isolated SGLang BF16 kernel
drop-ins in Megatron. The goal is to measure how each replacement changes
`train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl`, and matched
intermediate tensors before deciding which adapters are worth retaining.

## Scope

The experimental contract is intentionally narrower than the existing broad
true-on-policy backend:

- BF16 only; FP8, MXFP8, NVFP4, and mixed first/last-layer recipes are out of
  scope for this PR.
- The routed MoE, shared experts, router, dispatcher, combine, and probability
  placement remain native Megatron. SGLang uses its requested Triton MoE runner.
- The LM head remains native Megatron. Logits are a terminal observation, not a
  replacement target.
- Each kernel is selected independently through
  `MILES_SGLANG_BF16_KERNELS`.
- Unset or empty selection leaves the original block specs unchanged.
- Once a kernel is selected, missing SGLang APIs and unsupported layouts fail
  explicitly; there is no silent fallback.

The current selections are:

| Selection | Replaced boundary | Preserved boundary |
| --- | --- | --- |
| `rmsnorm` | Attention input RMSNorm and pre-MLP/pre-MoE RMSNorm | Attention, MoE, residual add, dense GEMMs |
| `qk_rmsnorm` | Q and K RMSNorm | QKV projection, RoPE, attention |
| `final_rmsnorm` | Final block RMSNorm | LM head and logit processing |

For TE layers, selecting `rmsnorm` decomposes only the fused input
`TELayerNormColumnParallelLinear` into a standalone SGLang RMSNorm followed by
`TEColumnParallelLinear`. The distributed-checkpoint mapping preserves the
canonical `self_attention.linear_qkv.layer_norm_` key. MoE specs are never
traversed or replaced.

The explicit attention-input norm also changes the live Megatron parameter
name from `self_attention.linear_qkv.layer_norm_weight` to
`input_layernorm.weight`. The image's Miles Qwen3-MoE weight converter does not
recognize that spelling. Validation therefore uses the same one-line alias and
focused test already present in the companion Miles side of
[radixark/Megatron-LM#30](https://github.com/radixark/Megatron-LM/pull/30)
([radixark/miles#1059](https://github.com/radixark/miles/pull/1059)). The harness
preflight executes the conversion probe before downloading or launching
anything. This integration-only Miles change is kept out of the Megatron PR.

## Numerical and backward contract

The forward calls SGLang's `true_on_policy_rms_norm` Triton kernel. Its dtype
contract is explicit per site because SGLang does not use one common boundary:

- Block input and pre-MoE norms reduce and apply the FP32 affine in FP32, then
  store BF16 for the following Megatron dense operation.
- Q/K norms reduce in FP32, round the normalized value to BF16, apply the FP32
  affine, and retain FP32 through RoPE. One small Megatron hook casts Q/K to
  BF16 immediately after RoPE and before attention, matching SGLang's dense
  attention boundary.
- Final norm rounds both the normalized value and effective weight to BF16 and
  returns BF16 to the native Megatron LM head.

Norm parameters are stored in FP32, matching the reference Megatron adapter and
the SGLang block/QK contract. The final-norm kernel explicitly casts its
effective weight to BF16. This storage choice is part of the experiment because
it can also affect optimizer updates after the first step.

The fused SGLang API is forward-only. Backward recomputes the same per-site
SGLang expression under PyTorch autograd; it is a surrogate, not the TE RMSNorm
backward. This avoids executing and retaining a second PyTorch graph during
forward while keeping the experiment isolated. Backward-kernel parity is not
claimed.

The image's native final norm has a BF16 weight and receives a BF16 hidden
tensor plus the FP32 residual carried by the block stack. The SGLang experiment
branch enables the fused kernel for exactly this already-FP32-promoted residual
case, so the final arm runs the same fused forward on rollout and training. It
does not fuse narrower residual additions, where loading both operands as FP32
would skip a native BF16 addition rounding point.

## Two-layer 8192-token harness

The harness is
`tools/sglang_bf16_parity/run_qwen3_30b_2layer.py`. It truncates the published
five-layer debug checkpoint at revision
`9c2ee37f22b7ef150675311b3d5e1c671838ffe1` to two layers, converts the exact
two-layer checkpoint to Megatron torch-dist, and exposes only two GPUs to Ray:

- GPU 0: one-GPU Megatron BF16 training.
- GPU 1: one-GPU SGLang rollout with `--sglang-moe-runner-backend triton`.

The prompt is pre-rendered to exactly 8064 tokens and generation is forced to
128 tokens, producing an exact 8192-token scoring prefill without paying for an
8192-token decode. Routing selections are replayed. SGLang prefill recomputes
rollout log probabilities. Radix prefix-cache reuse is disabled so the scoring
request actually executes all 8192 tokens instead of reusing the preceding
generation prefix; active-request KV state still serves decode. The image only
registers the dense
`qwen3_dense_true_on_policy_v1` contract, which is not valid for Qwen3-MoE and
would also switch the router away from its normal Triton `TopK` path. The local
SGLang experiment branch therefore uses the narrow
`SGLANG_BF16_NONMOE_PARITY=1` environment gate for only the non-MoE Qwen3
norm/cast boundaries. It adds the upstream fused RMSNorm API, the corresponding
Qwen3 norm/cast wiring, and debug hooks. The Triton MoE implementation and its
normal routing path are unchanged. Megatron does not enable its broad
`true_on_policy_contract`; the selected drop-in set is the only Megatron model
construction difference between variants.

The requested validation image is:

```text
radixark/miles@sha256:d9e01378d8820afd88824c798ea628b3b6cb87a6c6911db5165ef9d98187db55
```

The devbox is `c1/infra/bf16-nonmoe-parity`, a bare 8xB200 c1 allocation in
namespace `infra` on the default `earth` queue, without `--queue hell`. c1
requires the 8-GPU scheduling unit even though this harness uses only two GPUs.

B200 validation uses SGLang FlashInfer attention because the image's FA3
backend rejects SM100. The harness sets
`SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296`; the default 2 GiB planner
workspace overflowed during the exact 8192-token scoring prefill. The SGLang
experiment patch preserves this larger caller override.

Run all ablations from the image Megatron checkout after syncing this branch:

```bash
mkdir -p /root/shared_data/sglang-bf16-nonmoe
cd /root/Megatron-LM
set -o pipefail
python tools/sglang_bf16_parity/run_qwen3_30b_2layer.py \
  --variants baseline,rmsnorm,rmsnorm_qk,rmsnorm_qk_final \
  2>&1 | tee /root/shared_data/sglang-bf16-nonmoe/harness.log
```

Run one arm during iteration:

```bash
python tools/sglang_bf16_parity/run_qwen3_30b_2layer.py \
  --variants rmsnorm_qk
```

Each invocation creates a timestamped run directory so prior tensors and metric
records cannot be reused accidentally. The harness records full tensors for
the 8064-token generation prefill and the later scoring prefill, scalar
summaries, raw JSON/Markdown tensor comparisons, operation-ordered
`op_progression.json` and `op_progression.md` reports, CI-history metric
records, and one `result.json` per variant. The comparator pairs calls by
canonical name, phase, and compatible squeezed shape, then selects the largest
match; call ordinals are process-local and are never treated as cross-backend
identities.

## Experiment ledger

All deltas are relative to the native baseline from the same image, checkpoint,
prompt, seed, and topology. Lower absolute log-probability difference and lower
train/rollout KL are better. A result remains `pending` until the run has both
metrics and matched intermediate reports.

| Variant | Drop-ins | Context | Log-prob abs diff | Delta | Train/rollout KL | Delta | First useful divergence | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `baseline` | none | 8192 | 0.0223388672 | - | 0.0007065088 | - | Fused input norm + QKV region: joint Q/K/V relative L2 3.20e-7 | complete |
| `rmsnorm` | block + pre-MoE RMSNorm | 8192 | 0.0230712891 | +0.0007324219 | 0.0008305465 | +0.0001240377 | Layer 0 Q-norm output: max 0.0703125, relative L2 0.00231965 | complete |
| `rmsnorm_qk` | previous + Q/K RMSNorm | 8192 | 0.0213623047 | -0.0009765625 | 0.0007387963 | +0.0000322876 | Layer 0 attention output: max 0.000488281, relative L2 0.00116990 | complete |
| `rmsnorm_qk_final` | previous + final RMSNorm | 8192 | 0.0219726562 | -0.0003662109 | 0.0007983116 | +0.0000918028 | Layer 0 attention output: max 0.000488281, relative L2 0.00116990 | complete |

Run `20260806-infra-b200-v3` completed all four arms. Every arm produced its
metric record and intermediate report; the comparator matched 36 tensors for
the fused native baseline and 42 for each drop-in arm, with no canonical shape
mismatches or missing expected output taps. Scalar reductions use FP64 because
the 8K tensors are large enough for an FP32 cosine reduction to exceed one from
accumulation error.

The generated 8192-token sequence, response text, SGLang rollout log
probabilities, and the full `[8191, 2, 8]` routed-expert replay tensor are
byte-identical across all four arms. The route tensor SHA-256 is
`5ba83c4a9ba4d3ff777cc62490eca49b7853f5ae2f9a097faa1757580a0ec044`, so the
metric deltas are not caused by different samples or expert selections.

The block/pre-MoE RMSNorm replacement makes the layer-0 input norm and Q-norm
input exact, but native Q/K RMSNorm remains the next mismatch and both terminal
metrics get worse. Adding Q/K RMSNorm makes the canonical Q/K values, post-RoPE
values, and attention input exact; the first mismatch moves to the different
attention kernels' output. This arm has the lowest log-probability absolute
difference, while its KL remains slightly above baseline. Adding final RMSNorm
improves its local final-norm output (relative L2 0.00834666 to 0.00819899 and
exact fraction 25.55% to 31.01%) but does not improve either terminal metric
over `rmsnorm_qk`. No arm improves both terminal metrics over baseline in this
single deterministic sample. The layer-1 MoE output remains the worst relative
L2 boundary, as expected for a declared non-goal.

### Operation-by-operation observed drift

The comparator also groups canonical taps into execution order. Each cell below
is the relative L2 at that operation's output. QKV uses the joint norm of its
canonical Q, K, and V outputs. For attribution, fused attention aggregates the
joint Q/K/V input and residual rows aggregate both operands. A dash means that
the fused baseline does not expose that boundary.

These are natural-forward cumulative measurements, not an additive error
budget. Once an operation receives different inputs, its output drift can grow
or shrink through amplification, rounding, or cancellation. Only an exact
input followed by a non-exact output proves that the observed implementation
boundary introduced a mismatch.

| Operation output | Baseline | RMSNorm | + Q/K RMSNorm | + final RMSNorm |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 input | 0 | 0 | 0 | 0 |
| Layer 0 input RMSNorm | - | 0 | 0 | 0 |
| Layer 0 QKV projection, joint Q/K/V | 3.195412e-7 | 0 | 0 | 0 |
| Layer 0 Q RMSNorm | 0.002319654 | 0.002319654 | 0 | 0 |
| Layer 0 K RMSNorm | 0.002579950 | 0.002579950 | 0 | 0 |
| Layer 0 Q RoPE + BF16 cast | 0.003225849 | 0.003225849 | 0 | 0 |
| Layer 0 K RoPE + BF16 cast | 0.003330338 | 0.003330338 | 0 | 0 |
| Layer 0 fused attention core | 0.002668378 | 0.002668378 | 0.001169901 | 0.001169901 |
| Layer 0 attention output projection | 0.002594350 | 0.002594352 | 0.001321609 | 0.001321609 |
| Layer 0 attention residual merge | 0.002592164 | 0.002592167 | 0.001843506 | 0.001843506 |
| Layer 0 pre-MoE RMSNorm | 0.003554548 | 0.003554576 | 0.003008270 | 0.003008270 |
| Layer 0 MoE output, control/non-goal | 0.008465747 | 0.008465757 | 0.008057719 | 0.008057719 |
| Layer 0 MoE residual merge | 0.005423944 | 0.005423987 | 0.005136664 | 0.005136664 |
| Layer 1 input | 0.005423944 | 0.005423987 | 0.005136664 | 0.005136664 |
| Layer 1 input RMSNorm | - | 0.005298427 | 0.005036407 | 0.005036407 |
| Layer 1 QKV projection, joint Q/K/V | 0.005035123 | 0.005035514 | 0.004599212 | 0.004599212 |
| Layer 1 Q RMSNorm | 0.005074052 | 0.005073934 | 0.004615262 | 0.004615262 |
| Layer 1 K RMSNorm | 0.003859255 | 0.003858314 | 0.003514436 | 0.003514436 |
| Layer 1 Q RoPE + BF16 cast | 0.005562515 | 0.005562488 | 0.004901619 | 0.004901619 |
| Layer 1 K RoPE + BF16 cast | 0.004574221 | 0.004573206 | 0.003732830 | 0.003732830 |
| Layer 1 fused attention core | 0.004936935 | 0.004936046 | 0.004154173 | 0.004154173 |
| Layer 1 attention output projection | 0.005048583 | 0.005046495 | 0.004317600 | 0.004317600 |
| Layer 1 attention residual merge | 0.005413086 | 0.005412723 | 0.005082244 | 0.005082244 |
| Layer 1 pre-MoE RMSNorm | 0.007113807 | 0.007113542 | 0.006773177 | 0.006773177 |
| Layer 1 MoE output, control/non-goal | 0.009882709 | 0.009867726 | 0.009285527 | 0.009285527 |
| Layer 1 MoE residual merge | 0.007009508 | 0.007002754 | 0.006625629 | 0.006625629 |
| Final RMSNorm | 0.008563357 | 0.008563956 | 0.008346661 | 0.008198990 |

The first attributable boundaries are:

- Native baseline: layer 0 starts exact, but TE fuses input RMSNorm into QKV.
  The first exposed Q/K/V outputs have joint relative L2 `3.195412e-7` and max
  absolute error `3.051758e-5`; the current dumps cannot assign that small
  mismatch to the norm or the projection separately.
- `rmsnorm`: layer-0 input RMSNorm and QKV are exact. Q RMSNorm then produces
  relative L2 `0.002319654` and K RMSNorm produces `0.002579950` from exact
  inputs, so both native Q/K norm boundaries independently introduce drift.
- `rmsnorm_qk` and `rmsnorm_qk_final`: layer-0 Q, K, and V remain exact through
  Q/K normalization and the post-RoPE BF16 attention inputs. The fused attention
  output is the first mismatch: relative L2 `0.001169901`, max absolute error
  `0.0004882812`, and `93.213%` exact elements.

Later rows are cumulative only. The largest increases occur around the opaque
MoE control boundary, but its inputs already differ and MoE is intentionally a
non-goal. The fused attention dumps likewise cannot split QK, softmax, and PV,
and the current post-RoPE taps cannot split RoPE from its BF16 cast. Deeper
attribution would require same-input operator replay or additional debug-only
kernel outputs, not reinterpretation of the natural-forward deltas.

### Runtime attempt ledger

- v1 used SGLang FA3 attention and stopped before training because that backend
  rejects SM100.
- v2 completed the first weight sync and the 8064-token generation prefill, then
  the exact 8192-token scoring prefill overflowed FlashInfer's default 2 GiB
  planner workspace.
- v3 (`20260806-infra-b200-v3`) used FlashInfer attention with a 4 GiB planner
  workspace and completed every arm, including generation, scoring, training,
  final weight sync, and intermediate comparison.

## Intermediate taps

Debugging is enabled only when `MILES_SGLANG_BF16_DEBUG_DIR` is set. Both
Megatron and the image-aligned SGLang branch install hooks instead of embedding
logging logic in forward implementations. Full dumps retain up to two calls
whose token dimension is at least 8064. This includes the generation prefill
and the longer scoring pass while excluding one-token decode calls; matching by
shape selects the training/scoring pair.

The matched sequence is:

1. Layer input.
2. Input RMSNorm input/output.
3. QKV boundary.
4. Q and K RMSNorm input/output.
5. Post-RoPE Q/K and canonical V.
6. Attention core input/output.
7. Attention output projection input/output.
8. Attention residual merge at the pre-MoE RMSNorm input.
9. Pre-MoE RMSNorm input/output.
10. MoE output as a non-goal boundary/control.
11. MoE residual merge at layer output.
12. Final RMSNorm input/output.

SGLang stores Q/K norm rows and post-RoPE Q/K/V in flattened attention
layouts, while Megatron exposes explicit head dimensions. Debug hooks reshape
only those known layouts to `[tokens, heads, head_dim]` before writing them;
the model tensors themselves are untouched. For block, pre-MoE, and final
RMSNorm, SGLang's raw FP32 norm output is retained under a `_raw` diagnostic
name and the canonical output tap records the effective BF16 input to the next
dense operation. This matches the isolated Megatron adapter's boundary without
claiming the native raw norm tensors are identical.

Native TE fuses the baseline attention-input RMSNorm into QKV and does not
expose its exact output. That tap is omitted for the baseline instead of being
mislabelled; the QKV output and canonical pre-norm Q/K taps are the first
available downstream signals. Packed raw QKV tensors are retained for manual
inspection but excluded from elementwise metrics because Megatron and SGLang
use backend-specific packed layouts. Unknown canonical shape differences are
reported as errors rather than flattened and reshaped.

Router replay fixes the selected expert IDs; it is not an assertion about MoE
numerical equality. Because native Megatron and SGLang still differ in MoE
probability placement, the first-layer pre-MoE tensors are the cleanest signal
for these non-MoE drop-ins, and a non-zero mismatch floor after the MoE boundary
is expected.

## Source revisions

| Component | Revision | Notes |
| --- | --- | --- |
| Megatron image base | `4716f75475c78e2fc2c6f0d3af095f1681b770b4` | Revision baked into the requested image |
| Megatron PR base | `50ac48e87b8a31da7330de4a03d8ae42b985d9d2` | `zianglih:megatron-miles`; descendant of the image checkout |
| Validated Megatron experiment | `ec9c206d1f5e40dee2aa4f27c338b4d3367dcd2c` | Exact clean `/root/Megatron-LM` runtime head; later reporting-only commits do not change model execution |
| SGLang image base | `d218d6c7835307da50373f81704e61338b4e4847` | Revision baked into `/sgl-workspace/sglang` |
| SGLang experiment | `71f9ddd6c6242cbc5d5f119e82ad8a8efa2f451f` | `agent/bf16-nonmoe-parity-debug`, synced as a patch over the image-base checkout |
| Miles image base | `43d38ada230a431845338ed913f6c3a1b5f8355d` | Revision baked into `/root/miles` |
| Miles validation integration | `7e19132a4240996e52cbfa8be213a0d5946b1209` | `agent/bf16-nonmoe-parity-integration`, synced as a patch over the image base; explicit input-norm live-sync alias only |

## Known limitations

- SGLang carries the block residual separately and performs the residual add in
  FP32 at the next RMSNorm. This isolated Megatron adapter deliberately keeps
  native residual materialization, so rounding before layer 1 remains a floor;
  the experiment tests whether norm and dense-consumer boundaries reduce drift,
  not the full residual contract from PR #30.
- The SGLang Triton RMSNorm API is forward-only, so backward is a recomputed
  plain BF16 surrogate.
- The Q/K arm changes both the fused norm kernel and the SGLang FP32-through-RoPE
  dtype boundary. The intermediate dumps separate Q/K norm output, post-RoPE
  Q/K, and attention input so that those effects remain diagnosable.
- This one-GPU training/one-GPU rollout harness cannot measure TP-invariant row
  linear or collective ordering. Those are separate future experiments.
- MoE and LM-head mismatch floors are deliberately not fixed in this PR.
