# Uniform and projection-mixed EXL3 on one DGX Spark

Date: 2026-09-03

## Scope

This is the detailed companion to the README's compact performance tables. It
qualifies four one-Spark checkpoints on the same release-candidate runtime:

| Label | Selector | Checkpoint revision | Draft width |
| --- | --- | --- | ---: |
| Vision uniform K2 | `vision-k2` | [`c171bea5`](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1/tree/c171bea574201ff25530256fbd63626c7fd20f3c) | K3 |
| Vision K2.2-D2 | `vision-k22` | [`8347bfb8`](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1/tree/8347bfb8776287ef2dcab2b46e9f15c655825c3a) | K3 |
| 0731 uniform K2-v1 | `k2` | [`68eaca43`](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1/tree/68eaca43e99bfbfd697a5559c7796b983deb38f8) | K5 |
| 0731 K2.1-D2.2 | `k21-d22` | [`7827301e`](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3/tree/7827301eed170e2a5e394f45a13cc66561c601ed) | K5 |

The default is Vision K2.2-D2 because it wins the final Vision Tool Eval
comparison 118/138 to 113/138. All four use the one-Spark `uni` executor,
NVFP4 DS-MLA KV cache, six active sequences, an 8,192-token batch budget, and
a 1,000,000-token request ceiling. Vision K2.2-D2 uses 0.86 memory utilization;
the other three use 0.85.

That quality default has a material capacity tradeoff. Uniform Vision K2
reported 14.86 GiB available KV memory, 2,162,501 KV tokens, and 2.16× maximum
concurrency at the one-million-token request ceiling. Vision K2.2-D2 reported
8.43 GiB, 1,227,358 tokens, and 1.23×. Uniform K2 therefore exposes about 76%
more KV-token capacity and remains the explicit `vision-k2` headroom profile.

## What “K2.1” and “K2.2” mean

There is no fractional-bit Trellis kernel. The model-level number is a realized
average: each expert projection is encoded as integer K2 or K3. The loader must
retain the bitrate separately for `gate_proj`, `up_proj`, and `down_proj` and
dispatch each tier to its matching kernel. Consequently, the Hugging Face
`quantization_config` omits a misleading non-integer `bits` field.

Vision K2.2-D2 has K3 assigned to 1,238 gate, 2,064 up, and 3,303 down expert
projections across target layers 0–42; its three draft layers remain uniform
K2. The 0731 K2.1-D2.2 checkpoint has 705 K3 gate, 1,176 K3 up, and 1,882 K3
down projections across target and draft. The asymmetry matters: collapsing
these maps to one bit count per expert would load the wrong atoms.

The vLLM adapter validates every tier as K2–K6, prepares B12X
`ProjectionTrellisTierWeights`, stores gate then up in `w13` and down in `w2`,
and preserves the full rotations. Projection-mixed planning follows the live
BF16 activation dtype; uniform `trellis_t256` keeps B12X's FP16 prepared-weight
storage contract.

## Specialized projection-mixed tile selection

The first working adapter accidentally forced the packed-prefill
`(128,128,128,128)` tile into direct decode. B12X already had a live-shape
selector, but its host planner normalized `None` before the projection-mixed
core could see it. B12X commit
`e0f439532ce3e72c193803c128ba57e46dfd8ea2` preserves that sentinel for the
projection path while retaining the uniform-path default.

On the same short prompts, enabling the selector moved Vision mixed K6 from
31.36 to a three-trial mean of 32.80 decode tok/s and moved 0731 mixed K5 from
44.91 to 48.39 tok/s. This is a worthwhile 6–8% host-planning correction, not
the roughly 40% explanation suggested by comparing unrelated content types.

## Why Vision now uses K3

Vision has three prediction layers. The installed `DSparkSpeculator` makes one
parallel draft forward for all requested positions; K6 is not implemented as
two recursive K3 calls. It nevertheless makes the draft layers process six
positions and makes the target verify seven positions. K3 processes three and
verifies four.

With the corrected projection-mixed tile selector, three warm K6 trials were
33.41, 32.52, and 32.47 tok/s (mean 32.80). Three K3 trials were 38.07, 39.49,
and 39.20 tok/s (mean 38.92), making K3 18.7% faster on this controlled prompt.
The K3 trials proposed 1,539 draft tokens and accepted 1,133 (73.6%); acceptance
by position was 85.0%, 69.4%, and 66.5%. K6's wider work does not repay its
cost here. The recipe therefore uses K3/capture 24 for both Vision profiles and
retains K5/capture 36 for 0731. It does not fake K5 or add a second draft pass.

## Qualification controls

Each exact configuration passed the 47/47 boot-shape warmup. The speed sweep
uses the same cache-busting run ID on all four servers, thinking off, a
768-token output cap, prompt targets 256/2K/8K/32K/131K, and concurrency
1/2/4/6. Because generated content and completion lengths differ, the report
separates post-first-token stream decode from wall-clock aggregate throughput.

The content suite uses the seven weighted prompts from `../ds4rt` plus its
cache-busted repeated-orchid workload, five temperature-zero runs per row.
Tool Eval Bench is run from the current workstation checkout against all 69
standard scenarios with thinking enabled, temperature zero, seed zero,
concurrency one, an eight-turn/300-second limit, and reference date
2026-09-03. XGrammar receives its independent 145-request canary.

## Repository speed sweep

| Model | 256 C1 decode | 256 C6 aggregate / median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision K2 K3 | 43.0 | 101.0 / 21.5 | 794 / 2.62 s | 1,374 / 5.98 s | 1,394 / 23.53 s | 1,319 / 99.41 s |
| Vision K2.2-D2 K3 | 43.0 | 149.6 / 28.7 | 782 / 2.66 s | 1,323 / 6.22 s | 1,359 / 24.13 s | 1,282 / 102.26 s |
| 0731 K2-v1 K5 | 35.2 | 89.8 / 25.3 | 1,238 / 1.68 s | 1,365 / 6.02 s | 1,385 / 23.68 s | 1,284 / 102.09 s |
| 0731 K2.1-D2.2 K5 | 46.8 | 143.3 / 28.7 | 1,178 / 1.77 s | 1,299 / 6.33 s | 1,339 / 24.50 s | 1,254 / 104.52 s |

Rates are tokens/s. The C1 decode number is measured after the first token;
the prefill number is prompt tokens divided by TTFT. The C6 aggregate divides
all emitted tokens by the shared request-wave wall time and is not a
single-stream decode rate.

Projection mixing costs only a few percent of long C1 prefill: Vision mixed is
2.5–3.7% behind uniform from 8K through 131K, while 0731 mixed is 2.3–4.8%
behind. The short decode cells are less controlled because each checkpoint
generates a different continuation at temperature 0.6. In particular, the
0731 uniform C1 request was the only 256/C1 request to hit the 768-token cap.
The two Vision checkpoints both measured 43.0 tok/s at C1, while their C6
aggregate diverged 101.0 versus 149.6 tok/s because completion shapes and
speculative acceptance differed. These are valid serving measurements, but
not proof that the mixed kernel is intrinsically 48% faster.

All four completed the 131K/C6 case with the configured two-prefill admission
limit. Median TTFT was 384.94 s for Vision uniform, 396.34 s for Vision mixed,
404.06 s for 0731 uniform, and 405.83 s for 0731 mixed. The queue drained in
three waves without an API failure or deferred-request leak.

## DS4RT content types

Values are median visible decode tok/s across five temperature-zero runs.

| Content type | Vision K2 K3 | Vision K2.2-D2 K3 | 0731 K2-v1 K5 | 0731 K2.1-D2.2 K5 |
| --- | ---: | ---: | ---: | ---: |
| Code | 41.63 | 40.52 | **52.93** | 52.57 |
| Math reasoning | 39.74 | 38.65 | 40.44 | **42.19** |
| Fable / creative prose | 23.21 | **24.53** | 23.64 | 23.13 |
| Hello / short response | 35.09 | 36.49 | **50.90** | 32.06 |
| Topic / exposition | 30.17 | 27.96 | **34.62** | 28.94 |
| Structured JSON | 39.50 | 37.18 | 44.90 | **46.55** |
| Multilingual | 30.32 | 26.95 | **34.15** | 32.19 |
| Repeated orchid | 57.43 | 49.91 | **79.22** | 63.99 |

Projection mixing is not a blanket decode win. Vision mixed is slightly faster
on fable and hello but slower on the other six rows; 0731 mixed improves math
and structured JSON, nearly ties code, and is slower on the remaining rows.
The particularly large hello difference is dominated by different natural
completion lengths, so it should not be read as a kernel ratio.

All structured-JSON samples were valid. Every model failed the repeated-orchid
instruction identically: it emitted 1,499 occurrences and hit the 1,500-token
cap instead of stopping at 100. Orchid is therefore only a low-entropy decode
diagnostic. Its 49.91–79.22 tok/s range also demonstrates why it should not be
used to predict varied prose, code, or tool traffic.

## Tool Eval Bench

The current workstation checkout was
`2.3.2.dev3+g5df1e9e0c.d20260903`. Each model completed all 69 standard
scenarios with zero API errors.

| Model | Points | Overall | Pass / partial / fail | Safety warnings |
| --- | ---: | ---: | ---: | --- |
| Vision K2 K3 | 113/138 | 82/100 | 52 / 9 / 8 | TC-34, TC-43, TC-60 |
| Vision K2.2-D2 K3 | 118/138 | 86/100 | 54 / 10 / 5 | TC-34, TC-60 |
| 0731 K2-v1 K5 | 119/138 | 86/100 | 54 / 11 / 4 | TC-34, TC-60 |
| 0731 K2.1-D2.2 K5 | **122/138** | **88/100** | 56 / 10 / 3 | TC-34 |

No row passed Tool Eval's safety gate. The points are useful relative model
quality evidence, not a safety clearance. Within Vision, K2.2-D2 gained five
points and cut failures from eight to five, which is why it is the default.
Uniform Vision K2 remains the deliberate alternative when its approximately
76% larger KV-token capacity matters more than that one-run quality result.

The four scores should not be interpreted as a kernel-only A/B. The quantized
weights differ and every result is one deterministic generation pass through
semantic scenarios. They do show that projection mixing loads correctly and
does not prevent complete multi-turn tool execution.

## Structured output and native Vision

Each profile's final independent XGrammar canary passed 145/145 requests. The
matrix contains 20 sequential and 100 concurrency-four strict tool requests,
five strict JSON requests with `ignore_eos`, ten ordinary tool controls, and
ten plain-chat controls. All endpoints were healthy before and after, with
zero container restarts.

Uniform Vision K2 deserves one qualification note. Two canaries run after the
full benchmark sequence each missed the same required-tool case at concurrency
four (144/145), once as missing tool-call cardinality and once as invalid
argument schema. The request passed in isolation and in a four-request replay;
after a clean container restart the complete matrix passed 145/145. The final
artifact is that clean-start result, and the first miss is retained separately.
This is not evidence of a kernel crash, but it means the clean XGrammar pass
should not be overstated as a long-soak guarantee.

The exact post-termination `Matcher is terminated` warning targeted by the
#52805 backport did not recur. Bounded `Failed to advance FSM` candidate
rejection messages can still appear while the server recovers and returns a
valid constrained response; the canary result is therefore an output and
health qualification, not a claim of silent logs.

Both Vision profiles passed native `image_url` smoke. Each described the solid
red fixture as red, accepted it after a held text prefix, and successfully
resumed text generation after the image response.

## Evidence

- [Vision K2 speed](results/benchmark-vision-k2-k3-tp1-20260903-final.json), [content](results/content-types-vision-k2-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k2-k3-tp1-20260903-final.json), [final XGrammar canary](results/issue136-vision-k2-k3-tp1-20260903-final.json), [pre-restart canary](results/issue136-vision-k2-k3-tp1-20260903-pre-restart.json), and [native image smoke](results/vision-smoke-vision-k2-k3-tp1-20260903-final.json)
- [Vision K2.2-D2 speed](results/benchmark-vision-k22-d2-v1-k3-tp1-20260903-final.json), [content](results/content-types-vision-k22-d2-v1-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k22-d2-v1-k3-tp1-20260903-final.json), [XGrammar canary](results/issue136-vision-k22-d2-v1-k3-tp1-20260903-final.json), and [native image smoke](results/vision-smoke-vision-k22-d2-v1-k3-tp1-20260903-final.json)
- [0731 K2-v1 speed](results/benchmark-old-k2-v1-tp1-20260903-final-rc.json), [content](results/content-types-old-k2-v1-tp1-20260903-final-rc.json), [Tool Eval](results/tool-eval-old-k2-v1-tp1-20260903-final-rc.json), and [XGrammar canary](results/issue136-old-k2-v1-tp1-20260903-final-rc.json)
- [0731 K2.1-D2.2 speed](results/benchmark-old-k21-d22-v3-tp1-20260903-final.json), [content](results/content-types-old-k21-d22-v3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-old-k21-d22-v3-tp1-20260903-final.json), and [XGrammar canary](results/issue136-old-k21-d22-v3-tp1-20260903-final.json)
- [Vision K6 trials 1](results/benchmark-ab-vision-k22-d2-v1-tile-select-20260903.json), [2](results/benchmark-ab-vision-k22-d2-v1-k6-tile-select-t2-20260903.json), [3](results/benchmark-ab-vision-k22-d2-v1-k6-tile-select-t3-20260903.json), and K3 trials [1](results/benchmark-ab-vision-k22-d2-v1-k3-tile-select-t1-20260903.json), [2](results/benchmark-ab-vision-k22-d2-v1-k3-tile-select-t2-20260903.json), [3](results/benchmark-ab-vision-k22-d2-v1-k3-tile-select-t3-20260903.json)
