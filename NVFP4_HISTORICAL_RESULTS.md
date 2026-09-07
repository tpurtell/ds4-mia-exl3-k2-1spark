# Historical `nvfp4_ds_mla` model comparisons

These results are retained for comparison only. They were collected with the
old one-Spark defaults: `MAX_MODEL_LEN=1000000`, `MAX_NUM_SEQS=6`,
`GPU_MEMORY_UTILIZATION=0.85` (0.86 for Vision K2.2-D2), and
`KV_CACHE_DTYPE=nvfp4_ds_mla`.

The main sparse-MLA cache uses the same 584-byte token envelope for both dtype
names, and the issue #22 patch routes both through the fast FP8 decode kernel.
That does **not** make their complete hybrid-cache allocations equivalent:
`nvfp4_ds_mla` also selects the runtime's packed NVFP4 quantization mode for
other cache groups. The old Vision K2.2-D2 boot reported 1,227,358 total KV
tokens, while the 2026-09-07 `fp8_ds_mla` boot reported 601,445 under the same
model and memory-utilization setting. These are therefore genuine historical
compressed-KV results, even though the main MLA row itself is not four-bit.

The current README contains only freshly reproduced `fp8_ds_mla` results for
the default Vision K2.2-D2 checkpoint.

## 2026-09-03 four-model qualification

The sweep used one Spark per checkpoint, one common cache-busting run ID,
thinking off, temperature 0.6, and at most 768 output tokens. Each
`Prefill / TTFT` cell is concurrency one.

| Model | 256 C1 decode | 256 C6 aggregate | 256 C6 median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Vision K2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1) K3 | 43.0 tok/s | 101.0 tok/s | 21.5 tok/s | 794 tok/s / 2.62 s | **1,374 tok/s / 5.98 s** | **1,394 tok/s / 23.53 s** | **1,319 tok/s / 99.41 s** |
| [Vision K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1) K3 | 43.0 tok/s | **149.6 tok/s** | **28.7 tok/s** | 782 tok/s / 2.66 s | 1,323 tok/s / 6.22 s | 1,359 tok/s / 24.13 s | 1,282 tok/s / 102.26 s |
| [0731 K2-v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1) K5 | 35.2 tok/s | 89.8 tok/s | 25.3 tok/s | **1,238 tok/s / 1.68 s** | 1,365 tok/s / 6.02 s | 1,385 tok/s / 23.68 s | 1,284 tok/s / 102.09 s |
| [0731 K2.1-D2.2 v3](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3) K5 | **46.8 tok/s** | 143.3 tok/s | **28.7 tok/s** | 1,178 tok/s / 1.77 s | 1,299 tok/s / 6.33 s | 1,339 tok/s / 24.50 s | 1,254 tok/s / 104.52 s |

The decode prompt requests numbered lowercase English words. Checkpoints can
produce different continuations, so these are served-workload results rather
than identical-token kernel timings.

### DS4RT content types

The seven DS4RT cases and repeated-orchid case were measured five times with
thinking off and temperature zero. Values are median visible decode tok/s.

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

All four produced valid structured JSON. Every orchid run emitted 1,499
occurrences and reached the 1,500-token cap instead of stopping at 100, so that
row is a low-entropy throughput diagnostic rather than a correctness pass.

### Tool Eval Bench

Tool Eval Bench `2.3.2.dev3+g5df1e9e0c.d20260903` ran all 69 standard
scenarios with thinking enabled, temperature zero, seed zero, concurrency one,
and reference date 2026-09-03.

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| Vision K2 K3 | 113/138 | 82/100 | 52 / 9 / 8 | 0 |
| Vision K2.2-D2 K3 | 118/138 | 86/100 | 54 / 10 / 5 | 0 |
| 0731 K2-v1 K5 | 119/138 | 86/100 | 54 / 11 / 4 | 0 |
| 0731 K2.1-D2.2 K5 | **122/138** | **88/100** | 56 / 10 / 3 | 0 |

None passed Tool Eval's safety gate. Uniform Vision warned on TC-34/43/60;
Vision K2.2-D2 and 0731 K2-v1 warned on TC-34/60; 0731 K2.1-D2.2 warned on
TC-34. These were model-behavior failures, not API errors.

All four final XGrammar canaries passed 145/145 requests with healthy endpoints
and zero restarts. Both Vision profiles also passed native `image_url` smoke,
including a held text prefix and generation resumed after the image response.

Raw evidence:

- [Vision K2 speed](results/benchmark-vision-k2-k3-tp1-20260903-final.json), [content](results/content-types-vision-k2-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k2-k3-tp1-20260903-final.json), [XGrammar](results/issue136-vision-k2-k3-tp1-20260903-final.json), and [image smoke](results/vision-smoke-vision-k2-k3-tp1-20260903-final.json)
- [Vision K2.2-D2 speed](results/benchmark-vision-k22-d2-v1-k3-tp1-20260903-final.json), [content](results/content-types-vision-k22-d2-v1-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k22-d2-v1-k3-tp1-20260903-final.json), [parallel-4 Tool Eval control](results/tool-eval-vision-k22-d2-v1-k3-tp1-p4-20260903.json), [XGrammar](results/issue136-vision-k22-d2-v1-k3-tp1-20260903-final.json), and [image smoke](results/vision-smoke-vision-k22-d2-v1-k3-tp1-20260903-final.json)
- [0731 K2-v1 speed](results/benchmark-old-k2-v1-tp1-20260903-final-rc.json), [content](results/content-types-old-k2-v1-tp1-20260903-final-rc.json), [Tool Eval](results/tool-eval-old-k2-v1-tp1-20260903-final-rc.json), and [XGrammar](results/issue136-old-k2-v1-tp1-20260903-final-rc.json)
- [0731 K2.1-D2.2 speed](results/benchmark-old-k21-d22-v3-tp1-20260903-final.json), [content](results/content-types-old-k21-d22-v3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-old-k21-d22-v3-tp1-20260903-final.json), and [XGrammar](results/issue136-old-k21-d22-v3-tp1-20260903-final.json)
- [Detailed four-model analysis](20260903-mia-all-four-compare.md)
- [First Vision release comparison](20260903-mia-vision-k2-compare.md)

## 2026-08-24 0731 quant comparison

These three checkpoints used K5, 0.85 memory utilization, and otherwise the
same historical one-Spark profile.

### Repository speed sweep

| Model | 256 C1 decode | 256 C6 aggregate | 256 C6 median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K2-v0 | 49.1 tok/s | 134.7 tok/s | 25.7 tok/s | 1,161 tok/s / 1.79 s | 1,359 tok/s / 6.05 s | 1,382 tok/s / 23.73 s | 1,287 tok/s / 101.83 s |
| K2-v1 | **58.1 tok/s** | 135.2 tok/s | 26.3 tok/s | **1,148 tok/s / 1.81 s** | **1,376 tok/s / 5.98 s** | **1,394 tok/s / 23.53 s** | **1,292 tok/s / 101.47 s** |
| K2.1-v2 | 56.8 tok/s | **167.9 tok/s** | **32.0 tok/s** | 732 tok/s / 2.84 s | 824 tok/s / 9.98 s | 825 tok/s / 39.75 s | 789 tok/s / 166.19 s |

### DS4RT content types

| Content type | K2-v0 | K2-v1 | K2.1-v2 |
| --- | ---: | ---: | ---: |
| Code | 50.09 | 53.37 | **60.62** |
| Math reasoning | **48.91** | 40.79 | 44.14 |
| Fable / creative prose | 22.74 | 23.91 | **23.97** |
| Hello / short response | 36.09 | **50.48** | 38.50 |
| Topic / exposition | 30.52 | **35.02** | 31.21 |
| Structured JSON | 47.05 | 45.35 | **55.29** |
| Multilingual | 32.84 | 34.49 | **37.82** |
| Repeated orchid | 79.08 | 80.29 | **84.26** |

### Tool Eval Bench

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| K2-v0 | 120/138 | 87/100 | 56 / 8 / 5 | 0 |
| K2-v1 | **123/138** | **89/100** | 57 / 9 / 3 | 0 |
| K2.1-v2 | 120/138 | 87/100 | 55 / 10 / 4 | 0 |

All three completed without API errors and failed the safety gate. See
[the detailed 2026-08-24 analysis](20260824-mia-kX-compare.md) for scenario-level
interpretation and the raw-evidence links.
