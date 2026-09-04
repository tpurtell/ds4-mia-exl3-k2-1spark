# Vision-Exp recursive DSpark investigation

Date: 2026-09-04

Status: **hypothesis rejected**. Keep the released Vision recipe at K3. The
experimental K3+K3 implementation remains in this repository only to make the
negative result reproducible; it is deliberately absent from the production
Dockerfile and launcher.

## Conclusion

There is no evidence that Vision-Exp's three draft layers form a recurrent
three-token unit which should be run twice to make K6. Both structurally
plausible K3+K3 constructions booted and generated valid text, but neither
recovered acceptance at positions four through six. Both were slower than one
flat K6 pass on every content type, and flat K6 itself lost to K3 on the
difficult prose/exposition cases.

The simplest explanation is that the experimental Vision DSpark run is weak
beyond its first three proposal positions. This is not solely an EXL3 defect:
[official-checkpoint measurements](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/discussions/11)
show the same workload-dependent K3/K5 behavior. Quantizing the Vision draft
experts to K2.2-D2 makes the boundary much more severe, but it amplifies an
existing model weakness rather than creating the whole phenomenon.

The checkpoint does contain a trained confidence head which this vLLM lineage
drops. That head can select or trim low-confidence proposals; it does not
generate a missing recurrent state and cannot improve the draft logits by
itself. A confidence-scheduled engine is the credible way to avoid paying to
verify this bad tail. Fixed K3 is the practical approximation available in this
recipe today.

## What the three and five mean

Two independent dimensions were conflated in the adjacent Mia recipe:

- `num_nextn_predict_layers=3` describes the three sequential DSpark decoder
  stages in the Vision checkpoint.
- `dspark_block_size=5` is the trained/default width of the parallel proposal
  block.
- The runtime `num_speculative_tokens` is how many proposal rows are requested
  and verified per target step.

The three decoder stages all process the same parallel proposal block. They are
not three one-token heads, nor three successive three-token blocks. The generic
vLLM check requiring a runtime K larger than three to be divisible by three is
therefore not evidence of a model recurrence. Mia's K6 choice was the smallest
number satisfying both that validation rule and the checkpoint's stated block
size of five; it was a compatibility deduction, not a demonstrated optimum.

The official reference inference code likewise performs one parallel block and
then applies the final head. It exposes no loop which feeds stage three back to
stage one. The checkpoint tensor inventory contains no separately named
recurrence projection.

## Official-checkpoint corroboration

The Hugging Face discussion reports these measurements from the official FP8
Vision checkpoint on 2x DGX Spark:

| Workload | K5 acceptance | K5 tokens/step | K3 acceptance | K3 tokens/step |
|---|---:|---:|---:|---:|
| Count to 300 | 97.4% | 5.88 | 99.7% | 4.00 |
| Code | 46.4% | 3.31 | 64.2% | 2.93 |
| Prose | 18.0% | 1.90 | 28.6% | 1.86 |

This is not a universal “tokens after three never work” result: deterministic
counting can use the full block extremely well, and K5 still gets a small
tokens-per-step benefit on code. On ordinary prose, however, K5 adds almost no
useful work over K3. The drafter is both content-sensitive and poorly trained
for a long horizon. The report also warns that an older missing shared-expert
loader fix can mimic this failure. This recipe's loader includes those shared
experts, so that known bug is not our explanation.

The discussion's report that SGLang became faster after overriding
`num_nextn_predict_layers` to one is compatible with this diagnosis: a much
cheaper drafter can win even with lower acceptance when the extra draft stages
do not buy enough accepted tokens. It is not evidence for K3 recursion, and an
engine/version-specific config override is not yet a portable recommendation.

## Recurrence hypotheses tested

For one K3 pass, each draft layer returns the delayed mHC representation

```text
Z_l: [request * 3, hc_mult=4, hidden_size=4096]
```

The target normally supplies DSpark with mean-pooled hidden states from target
layers 40, 41, and 42. The stronger experiment applied the same boundary to the
three draft stages:

```text
D_l = mean_hc(Z_l)                         # [R*3, 4096]
D   = concat(D_0, D_1, D_2)               # [R*3, 12288]
M   = main_norm(main_proj(D))              # [R*3, 4096]
```

`main_proj` and `main_norm` are existing checkpoint weights, so `M` has the
shape expected by every draft layer's context-KV projection. This proves only
shape compatibility; those weights were trained on target-layer taps, not on
draft-layer outputs.

The four arms were:

- **Stock K3:** one K3 block followed by target verification.
- **Flat K6:** the stock one-shot six-row proposal block.
- **Recursive KV:** K3, retain its tentative per-layer query KV, advance
  positions by three, anchor on proposal three, run a second K3, verify all six.
- **Recursive tap:** the same two passes, but replace first-pass context KV with
  KV projected from `M` before the second pass.

The recursive modes ran eagerly so the second pass could rebuild positions,
slot mappings, and attention metadata safely. That overhead makes absolute
speed worse, but it cannot explain their failed position-4+ acceptance.

## Acceptance result

Model: `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1`, one
Spark, identical released image/settings, eager mode. Decode is the median from
the acceptance probe, not the repeated-orchid workload.

| Arm | Decode tok/s | Overall draft acceptance | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock K3 | **34.6** | **90.3%** | 95.7% | 87.9% | 87.3% | — | — | — |
| Flat K6 | 30.0 | 47.9% | 96.1% | 92.5% | 85.2% | 9.3% | 2.4% | 1.8% |
| Recursive KV | 28.3 | 47.5% | 97.6% | 91.3% | 90.1% | 3.6% | 1.5% | 0.9% |
| Recursive tap | 27.2 | 46.7% | 97.0% | 89.9% | 89.3% | 2.7% | 1.5% | 0.0% |

The decisive observation is the recursive tail: neither pass reset proposal
four to a new “position one.” The reconstructed-tap version was even worse,
which is what we expect when shape-compatible but semantically different draft
states are fed into projections trained for target states.

For comparison, the older Vision K2 draft at flat K6 measured 34.5 tok/s,
44.0% overall, with `83.1 / 73.4 / 65.0 / 27.7 / 10.2 / 4.5%` per position.
K2.2-D2 greatly improves its first three positions but has a sharper tail. That
is evidence that quantization changes the severity and shape of the failure,
not evidence that the underlying model supports recurrence.

## Content-type result

Each regular case is the aggregate decode rate over five deterministic repeats.
Orchid has one warm-up and five timed cache-busted repeats. All values are
output tokens per second.

| Content type | Stock K3 | Flat K6 | Recursive KV | Recursive tap |
|---|---:|---:|---:|---:|
| Code | 31.65 | **32.49** | 29.50 | 25.86 |
| Math reasoning | 30.98 | **34.10** | 29.33 | 26.96 |
| Creative prose | **18.79** | 15.35 | 14.38 | 13.75 |
| Short response | 27.74 | **34.66** | 33.03 | 31.79 |
| Exposition | **22.27** | 19.96 | 18.42 | 17.38 |
| Structured JSON | 30.54 | **35.03** | 30.68 | 29.62 |
| Multilingual | 20.65 | **21.39** | 18.77 | 17.17 |
| Repeated orchid | 37.85 | **55.66** | 52.74 | 50.90 |

Flat K6 benefits on easy, constrained, or repetitive generations despite its
bad tail; K3 wins on the less predictable prose cases. Both recursive variants
lose to flat K6 on every row and usually lose to K3. Because the content suite
was the qualification gate, tool-call evaluation and production integration
were intentionally skipped.

Raw results:

- [`results/content-types-recursive-dspark-stock-k3-20260904.json`](results/content-types-recursive-dspark-stock-k3-20260904.json)
- [`results/content-types-recursive-dspark-one-shot-k6-20260904.json`](results/content-types-recursive-dspark-one-shot-k6-20260904.json)
- [`results/content-types-recursive-dspark-recursive-kv-20260904.json`](results/content-types-recursive-dspark-recursive-kv-20260904.json)
- [`results/content-types-recursive-dspark-recursive-tap-20260904.json`](results/content-types-recursive-dspark-recursive-tap-20260904.json)

## Quantization audit

The K2.2-D2 calibration did **not** optimize only three output positions. Its
evidence ledger records 327,680 anchors with five proposal rows per anchor, and
all five rows contribute jointly to each projection's Hessian. All three DSpark
stages and every routed expert were covered.

That rules out “the quantizer was configured for K3 instead of the trained K5”
as a literal cause. It does not rule out a quantization contribution: the GPTQ
objective pools rows into a position-agnostic `X^T X` reconstruction objective.
It protects average layer reconstruction, not acceptance at each proposal
position. A fragile late-position signal can therefore degrade even when the
aggregate reconstruction objective is healthy.

Direct comparison of the K2 and K2.2-D2 checkpoints found the same 9,316 MTP
tensor keys. Of those, 2,306 differ: all 2,304 routed-expert Trellis tensors
across three stages, plus two scale tensors. The context combiner, shared
experts, Markov head, and confidence head match. D2 was genuinely requantized;
it was not an unchanged draft copied into a new target checkpoint.

## The trained component vLLM omits

The official model includes `mtp.2.confidence_head.proj.weight`. Its reference
implementation computes a scalar confidence from the final draft hidden state
and the Markov embedding after sampling the proposal block. This vLLM lineage
explicitly discards that weight because the confidence head is not wired into
inference. Current SGLang source both loads the head and exposes confidence to
its ragged-verification planner:

- [official Vision reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/main/inference/model.py)
- [vLLM source dropping the confidence head](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/amd/dspark.py)
- [SGLang Vision DSpark implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_dspark.py)

This is a real integration gap, but it is not a missing checkpoint tensor and
does not support the recurrence hypothesis. The trained tensor is present; the
engine ignores it. Wiring confidence-scheduled, per-request proposal lengths
would be a separate scheduler project. Given this checkpoint's curve, fixed K3
is the low-risk production choice until such a scheduler is implemented and
qualified.

## Reproducing the rejected experiment

The experiment is isolated from the production build:

```bash
docker build -f Dockerfile.recursive-dspark-experiment \
  -t ds4-mia-recursive:exp .

# Inside a launch environment derived from this recipe:
DSPARK_TOKENS=6 DSPARK_RECURSIVE_MODE=kv  ...
DSPARK_TOKENS=6 DSPARK_RECURSIVE_MODE=tap ...
```

The source-locked patch is
[`patches/hotfix-dsv4-recursive-dspark.py`](patches/hotfix-dsv4-recursive-dspark.py).
It requires Vision-Exp, K6, the anchor-first DSpark layout, and eager execution.
It should not be incorporated into the released image.
