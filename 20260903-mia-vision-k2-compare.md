# Vision-Exp K2 v1 versus 0731 calibrated K2-v1 on one DGX Spark

Date: 2026-09-03
Runtime image: `ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:9bd058d1b91fc8d9164b0cf45ed8355fdbd5a05a3715ec38c0d5a67163dd1b60`

## Conclusions

The Vision-Exp K2 checkpoint is a viable one-Spark vLLM target in this recipe.
It cold-boots, serves text and real image input, survives structured-output and
tool traffic without a restart, and has prompt-ingestion performance in the
same band as the existing 0731 K2-v1 model. This report records the first
Vision publication, when 0731 K2-v1 was still the default and Vision used K6.
The follow-up projection-mixed release makes uniform Vision K2 the default
and qualifies it at K3; its newer four-model results are reported separately.

The clearest performance findings are:

- Vision's controlled short C1 decode was 58.2 tok/s versus 54.4 for 0731.
  At C6, 0731 led aggregate throughput 105.4 to 101.2 tok/s and median stream
  rate 23.8 to 20.8 tok/s. These are close enough to call the same broad K2
  decode band, not evidence of a general Vision speed advantage.
- C1 prompt ingestion was within 3.0% from 2K through 32K. Vision was 2.3%
  faster at 131K in this run (1,313 versus 1,283 prompt tok/s).
- Content-dependent decode favored 0731 on six of seven ordinary prompts.
  Vision led the math row and the intentionally low-entropy orchid path.
- Tool Eval was effectively tied: Vision scored 116/138 and 84 overall;
  0731 scored 117/138 and 85. Both completed all 69 scenarios with no API
  error, and neither passed the safety gate.

## Checkpoints and runtime

| Label | Checkpoint | Revision | Spark |
| --- | --- | --- | --- |
| Vision-Exp K2 v1 | [wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1) | `c171bea574201ff25530256fbd63626c7fd20f3c` | ostrich |
| 0731 K2-v1 | [wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1) | `68eaca43e99bfbfd697a5559c7796b983deb38f8` | emu |

Both used Mia/Anemll's digest-pinned vLLM 0.25.2 runtime, the merged Mia fixes
through 2026-09-03, B12X commit
`3fc8d1491d1313c0ca64b2b95772972b7f42ee9d`, NVFP4 DS-MLA KV cache, and the
same one-Spark `uni` executor. The common service profile was a 1,000,000-token
ceiling, six active sequences, an 8,192-token batch budget, and 0.85 memory
utilization. 0731 proposed five dSpark tokens. Vision proposed six because its
three next-token-prediction layers require a proposal count divisible by three.

These K6 figures remain historical evidence for the first published Vision
image, not the current launch default. A controlled follow-up found K3 18.7%
faster than K6 on Vision K2.2-D2, so the current Vision profiles use one
three-token parallel proposal and a 24-token CUDA-graph capture ceiling.

## What Vision support required

The quant itself remains ordinary uniform K2. The integration work was in the
model/runtime boundary:

1. The first three Vision hash-MoE layers legitimately omit the correction
   bias expected by the 0731 config parser. A narrow config patch accepts only
   that exact Vision layout and leaves every other shape fail-closed.
2. Vision-Exp's official encoder and multimodal processor are installed before
   vLLM starts. The recipe accepts OpenAI `image_url` parts on user messages,
   applies the upstream role and image-cache fixes, and supplies the official
   encoding module from the quant repository's own `encoding/` directory.
3. B12X's prepared Trellis storage contract is FP16. The adapter now plans and
   prepares with that exact dtype and uses bounded 512/2,560/8,192 prefill
   capacity buckets. The earlier one-size 8,192 plan made 2K prompt ingestion
   collapse to roughly 313–326 tok/s; bucketing restored 1,177–1,214 tok/s.
4. Blank Compose values are normalized before compiler/runtime parsing.
   `CUTE_DSL_ARCH` resolves to `sm_121a`; model length and memory utilization
   receive non-empty defaults, avoiding the empty enum-key crash.

A real JPEG data-URL request identified its solid red content correctly and
accounted 117 image tokens. Text-only smoke tests also passed. Vision images
are user-role content; structured images in system, assistant, tool, or
function messages are intentionally rejected by the upstream Vision contract.

## Repository speed sweep

The final sweep used unique run IDs so prefix caching could not turn repeated
prompts into false 10K+ tok/s prefill results. Compiled kernels remained warm,
which makes this a steady-state engine test rather than a first-boot compiler
test. It covered prompt targets 256, 2K, 8K, 32K, and 131K at concurrency
1/2/4/6, with thinking off and a 768-token output ceiling.

### C1 prompt ingestion

Each cell is median prefill tok/s / median TTFT.

| Prompt target | Vision-Exp K2 v1 | 0731 K2-v1 | Vision delta |
| ---: | ---: | ---: | ---: |
| 256 | 748 / 0.39 s | 797 / 0.37 s | -6.1% |
| 2K | 1,177 / 1.77 s | 1,214 / 1.72 s | -3.0% |
| 8K | 1,331 / 6.18 s | 1,340 / 6.14 s | -0.7% |
| 32K | 1,383 / 23.72 s | 1,374 / 23.88 s | +0.7% |
| 131K | 1,313 / 99.89 s | 1,283 / 102.17 s | +2.3% |

There is no length-dependent Vision regression in this range. The result
instead converges with 0731 at 8K–32K and slightly crosses it at 131K.

### Controlled short decode

The 256-target case emits a long numbered-word response, so it is less
sensitive to a tiny natural completion than an ordinary chat prompt.

| Concurrency | Vision aggregate / median stream | 0731 aggregate / median stream |
| ---: | ---: | ---: |
| 1 | 56.2 / 58.2 tok/s | 52.3 / 54.4 tok/s |
| 6 | 101.2 / 20.8 tok/s | 105.4 / 23.8 tok/s |

Vision hit the 768-token ceiling in four requests across the 8K-C6, 32K-C4,
and 32K-C6 cells. 0731 hit it in none. That difference makes many other decode
cells completion-shape comparisons rather than pure kernel comparisons; the
raw files retain every output count and finish reason.

At 131K-C6, the shared wave took 603.4 seconds for Vision and 624.5 seconds for
0731. Median TTFT was 383.3 versus 400.1 seconds. This is consistent with the
C1 long-prefill result, but output lengths differed enough that the resulting
5.39 versus 4.27 aggregate output tok/s should not be read as a decode win.

## DS4RT content types

The seven weighted prompts come from
`../ds4rt/python/tools/bench_real_full_mtp_acceptance.py`; orchid comes from
`bench_real_full_repeat_decode.py`. Values are median visible decode tok/s over
five deterministic runs with thinking off and temperature zero.

| Content type | Vision-Exp K2 v1 | 0731 K2-v1 |
| --- | ---: | ---: |
| Code | 42.82 | 53.36 |
| Math reasoning | 42.32 | 40.73 |
| Fable / creative prose | 21.66 | 23.89 |
| Hello / short response | 38.28 | 51.15 |
| Topic / exposition | 26.54 | 34.85 |
| Structured JSON | 41.72 | 45.33 |
| Multilingual | 28.99 | 34.48 |
| Repeated orchid | 91.13 | 80.09 |

The ordinary rows include different output lengths and several capped
completions, so they describe user-visible serving behavior rather than an
isolated kernel. Both structured-JSON outputs parsed successfully. Both orchid
runs were incorrect in the same way: every sample emitted 1,499 occurrences,
then stopped at the 1,500-token ceiling instead of producing exactly 100.

## Tool Eval Bench and XGrammar

The benchmark was run from the workstation checkout—not the older Spark
installation—at `2.3.2.dev3+g5df1e9e0c.d20260903` (`5df1e9e-dirty`; the
pre-existing checkout modifications were not changed). Settings were all 69
standard scenarios, vLLM backend, thinking enabled, temperature zero, seed
zero, concurrency one, eight maximum turns, a 300-second timeout, and reference
date 2026-09-03.

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| Vision-Exp K2 v1 | 116/138 | 84/100 | 55 / 6 / 8 | 0 |
| 0731 K2-v1 | 117/138 | 85/100 | 54 / 9 / 6 | 0 |

Vision led 0731 by one point in Restraint & Refusal, Toolset Scale, and Creative
Composition; 0731 led by four points in Safety & Boundaries. The other eleven
categories tied. Vision's safety warnings were TC-34/42/43/58/60. 0731's were
TC-32/42/60, including the critical cross-turn sleeper injection shared with
Vision. Consequently, neither overall score is a safety clearance.

Before the XGrammar fix, the same suite completed at 114/138 for Vision and
118/138 for 0731 but repeatedly logged the characteristic post-termination
matcher warning. Those score differences are ordinary generation variance and
are not used as an A/B quality claim. The relevant engine result is binary:
after the exact upstream #52805 backport, each model passed the dedicated
145/145 request canary, then the full 69-scenario rerun, with zero matching
warnings, API errors, restarts, or lost health checks. The patch accepts only
the pinned stock or exact post-patch source hash and fails the boot on drift.

## Evidence

- [Vision speed](results/benchmark-vision-k2-tp1-20260903-final.json)
- [0731 speed](results/benchmark-old-k2-v1-tp1-20260903-final.json)
- [Vision content](results/content-types-vision-k2-tp1-20260903.json)
- [0731 content](results/content-types-old-k2-v1-tp1-20260903.json)
- [Vision Tool Eval](results/tool-eval-vision-k2-tp1-20260903.json)
- [0731 Tool Eval](results/tool-eval-old-k2-v1-tp1-20260903.json)
- [Vision pre-fix XGrammar baseline](results/tool-eval-vision-k2-tp1-20260903-xgrammar-baseline.json)
- [0731 pre-fix XGrammar baseline](results/tool-eval-old-k2-v1-tp1-20260903-xgrammar-baseline.json)
- [Vision 145-request XGrammar canary](results/issue136-vision-k2-tp1-20260903.json)
- [0731 145-request XGrammar canary](results/issue136-old-k2-v1-tp1-20260903.json)

No two-Spark result is included or implied by this comparison.
