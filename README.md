# DeepSeek V4 Flash Vision-Exp, K2, and K2.1 on one DGX Spark

This recipe serves standard Hugging Face EXL3 checkpoints on a single DGX
Spark using MiaAI-Lab's DeepSeek V4 Flash runtime. It supports the Vision-Exp
K2 checkpoint, the 0731 uniform K2 checkpoints, and mixed per-expert K2/K3
checkpoints without rounding mixed weights to one checkpoint-wide bit count.

The default is
[Vision-Exp K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1),
the higher-scoring Vision checkpoint in the final Tool Eval run. It uses a
1,000,000-token request ceiling, six active sequences, 0.86 GPU memory
utilization, NVFP4 DS-MLA KV cache, and one three-token parallel dSpark
proposal. A plain `docker compose up -d` or `./launch.sh --nodes 1` keeps that
default. Choose `--model vision-k2` when KV-cache headroom matters more.

The recipe began as a fork of
[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
and includes its upstream fixes through 2026-09-03.

## Supported checkpoints

| Launch selector | Checkpoint | Layout | Tested here |
| --- | --- | --- | --- |
| `k2-v0` | [K2 calibrated v0](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0) | Uniform K2; top-6/legal calibration | Yes |
| `k2` / `k2-v1` | [K2 calibrated v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1) | Uniform K2; rare-expert fallback and math calibration | Yes |
| `vision-k2` / `vision` | [Vision-Exp K2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1) | Uniform K2 target and draft; 2.16M KV tokens | Yes; KV-headroom profile |
| `vision-k22` | [Vision-Exp K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1) | Projection-mixed K2/K3 target; uniform K2 draft; 1.23M KV tokens | Yes; quality default |
| `k21-d22` | [0731 K2.1-D2.2 calibrated v3](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3) | Projection-mixed K2/K3 target and draft | Yes |
| `k21-v1` | [K2.1 calibrated v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v1) | Mixed K2/K3 target and draft | Boot/tool-call smoke test |
| `k21` / `k21-v2` | [K2.1 calibrated v2](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2) | Mixed K2/K3 target; forced-K2 draft | Yes |

K2.1 is not a fractional-bit kernel. Each expert tensor still has an integer
`bits_per_weight` value—K2 or K3—and the loader dispatches each tier to its
matching kernel. The 2.1 figure is the checkpoint-wide realized average. The
embedded Hugging Face `quantization_config` therefore omits the invalid
non-integer `bits` field instead of rounding it.

### Choosing the Vision profile

| Profile | Tool Eval | Available KV memory | KV token capacity | 1M-token concurrency | Choose it for |
| --- | ---: | ---: | ---: | ---: | --- |
| `vision-k22` | **118/138**, overall 86 | 8.43 GiB | 1,227,358 | 1.23× | Measured tool quality; default |
| `vision-k2` | 113/138, overall 82 | **14.86 GiB** | **2,162,501** | **2.16×** | Long-context/concurrent KV headroom |

Uniform K2 exposes about 76% more KV-token capacity. K2.2-D2 won the controlled
Tool Eval comparison by five points, so the recipe defaults to quality and
keeps the larger-cache profile one selector away.

## Launch

The Spark should already have current NVIDIA drivers, Docker, the NVIDIA
Container Toolkit, Docker Compose, Git, and the Hugging Face CLI. Download the
default model, pull the image, and launch:

```bash
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1 \
  --revision 8347bfb8776287ef2dcab2b46e9f15c655825c3a

docker pull ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest
git clone https://github.com/tpurtell/ds4-mia-exl3-k2-1spark.git
cd ds4-mia-exl3-k2-1spark
cp .env.example .env
./launch.sh --nodes 1
```

Both Vision quant repositories carry the complete official `encoding/`
directory, so no separate base-model metadata download is needed. The prior
calibrated 0731 default remains available explicitly:

```bash
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1 \
  --revision 68eaca43e99bfbfd697a5559c7796b983deb38f8
./launch.sh --nodes 1 --model k2
```

The alternate Vision KV-headroom profile and projection-mixed 0731 checkpoint
use the same image. K2/K3 tier maps are read per expert and per projection; no
checkpoint-wide fractional bit value is passed to vLLM:

```bash
# Vision uniform-K2 target and draft: larger KV cache
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1 \
  --revision c171bea574201ff25530256fbd63626c7fd20f3c
./launch.sh --nodes 1 --model vision-k2

# 0731 target and draft: projection-mixed K2/K3
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3 \
  --revision 7827301eed170e2a5e394f45a13cc66561c601ed
./launch.sh --nodes 1 --model k21-d22
```

`vision-k22` resolves GPU memory utilization to 0.86 when the setting is
blank; the other profiles resolve to 0.85. An explicit valid value still wins.

The other two historically measured 0731 checkpoints use the same image:

```bash
# Original K2 calibration
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0 \
  --revision dff9afc6f5fe50a890590f7b6d5339ceaf5ba51e
./launch.sh --nodes 1 --model k2-v0

# Mixed K2.1 v2
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2 \
  --revision a2b066719ebdc0cbb0eacc752ffe7a2190c919aa
./launch.sh --nodes 1 --model k21
```

Follow startup with `docker logs -f ds4-mia-vision-k22-tp1`, or use Compose for
the default Vision K2.2-D2 service:

```bash
docker compose up -d
```

Once `/health` is ready, the server exposes the OpenAI-compatible API on port
8888:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp-exl3-k2.2-d2-v1",
    "messages": [{"role": "user", "content": "Why are tiny bits so charming?"}],
    "stream": true
  }'
```

Vision accepts `image_url` content on user messages. For example:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp-exl3-k2.2-d2-v1",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image briefly."},
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]}]
  }'
```

Stop it with `./stop.sh`.

## Defaults that matter

| Knob | Default | Purpose |
| --- | ---: | --- |
| `MODEL_KIND` | `vision-k22` | Tool-Eval-winning projection-mixed Vision target; uniform K2 draft |
| `DISTRIBUTED_EXECUTOR_BACKEND` | `uni` on one Spark | Unified-memory single-node execution |
| `MAX_MODEL_LEN` | `1000000` | Decimal one-million-token request ceiling |
| `MAX_NUM_SEQS` | `6` | Low-concurrency agent-serving profile |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Chunked-prefill budget |
| `GPU_MEMORY_UTILIZATION` | `0.85`; `0.86` for `vision-k22` | Model-aware KV-cache allocation target |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | Compact DeepSeek V4 hybrid cache |
| `DSPARK_TOKENS` | `5` for 0731; `3` for Vision-Exp | Model-aware speculative proposal width |
| `DSPARK_ENFORCE_EAGER` | `0` | Set to `1` only to isolate CUDA-graph behavior |
| `DEFAULT_THINKING` | `max` | Requests can override it in `chat_template_kwargs` |
| `PREFIX_CACHE` | `1` | Reuse real repeated agent prefixes |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | `1` | Source-locked fix for speculative tokens after grammar termination |

The launch path protects the CUTE kernel compiler from empty Compose values.
Blank `CUTE_DSL_ARCH`, memory-utilization, and model-length settings are
normalized to non-empty defaults; Spark resolves to `sm_121a`. This avoids the
empty-string enum lookup crash seen in the original recipe.

For target-only diagnostics, `DSPARK_TOKENS=0` cleanly omits speculative
decoding; it is not the recommended serving profile.

Vision's K3 default also matches DeepSeek's published Vision-Exp vLLM launch.
In the controlled fixed-output sweep, K3/K4/K5 measured 44.6/42.5/39.3 tok/s;
the fourth proposal's cumulative prefix acceptance fell to about 12--13%.
The [numerics analysis](NUMERICS_FOR_VISION_UPDATES.md) includes the phase-lock,
single-row indexer, and equal-width cycle-cost controls behind that choice.

## Performance

### 2026-09-03 final four-model qualification

All four checkpoints ran concurrently on separate single Sparks with the same
release-candidate runtime. The sweep used one common cache-busting run ID,
thinking off, temperature 0.6, and at most 768 output tokens. Each
`Prefill / TTFT` cell below is concurrency one.

| Model | 256 C1 decode | 256 C6 aggregate | 256 C6 median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Vision K2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1) K3 | 43.0 tok/s | 101.0 tok/s | 21.5 tok/s | 794 tok/s / 2.62 s | **1,374 tok/s / 5.98 s** | **1,394 tok/s / 23.53 s** | **1,319 tok/s / 99.41 s** |
| [Vision K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1) K3 | 43.0 tok/s | **149.6 tok/s** | **28.7 tok/s** | 782 tok/s / 2.66 s | 1,323 tok/s / 6.22 s | 1,359 tok/s / 24.13 s | 1,282 tok/s / 102.26 s |
| [0731 K2-v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1) K5 | 35.2 tok/s | 89.8 tok/s | 25.3 tok/s | **1,238 tok/s / 1.68 s** | 1,365 tok/s / 6.02 s | 1,385 tok/s / 23.68 s | 1,284 tok/s / 102.09 s |
| [0731 K2.1-D2.2 v3](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3) K5 | **46.8 tok/s** | 143.3 tok/s | **28.7 tok/s** | 1,178 tok/s / 1.77 s | 1,299 tok/s / 6.33 s | 1,339 tok/s / 24.50 s | 1,254 tok/s / 104.52 s |

The decode workload asks for numbered lowercase English words; it is not the
low-entropy orchid loop. Checkpoints still produce different continuations,
and the 0731 uniform C1 request alone reached the 768-token cap, so the rates
measure served behavior rather than identical-token kernel execution.

The seven DS4RT content prompts and repeated-orchid case were each measured
five times at temperature zero with thinking off. Values are median visible
decode tok/s.

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

All four produced valid structured JSON. None followed the orchid count
instruction: every timed run emitted 1,499 `orchid` occurrences and hit the
1,500-token cap instead of stopping at 100. That row is therefore a
low-entropy throughput diagnostic, not a correctness pass.

The current local Tool Eval Bench checkout
`2.3.2.dev3+g5df1e9e0c.d20260903` ran all 69 standard scenarios with thinking
enabled, temperature zero, seed zero, concurrency one, and reference date
2026-09-03.

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| Vision K2 K3 | 113/138 | 82/100 | 52 / 9 / 8 | 0 |
| Vision K2.2-D2 K3 | 118/138 | 86/100 | 54 / 10 / 5 | 0 |
| 0731 K2-v1 K5 | 119/138 | 86/100 | 54 / 11 / 4 | 0 |
| 0731 K2.1-D2.2 K5 | **122/138** | **88/100** | 56 / 10 / 3 | 0 |

None passed Tool Eval's safety gate. Uniform Vision warned on TC-34/43/60;
Vision K2.2-D2 and 0731 K2-v1 warned on TC-34/60; 0731 K2.1-D2.2 warned on
TC-34. These are model-behavior failures, not server errors. The source-locked
upstream #52805 backport is enabled by default and remains fail-closed against
an unexpected runtime source or dependency version.

For the two Vision profiles, the result is a user-selectable tradeoff rather
than a compatibility split: use the default `vision-k22` for the higher
measured Tool Eval score, or select `vision-k2` for about 76% more KV-token
capacity. Both use the same image, API, Vision encoding metadata, and K3 draft
width.

All four final XGrammar canaries passed 145/145 requests with healthy endpoints
and zero restarts. Both Vision profiles also passed native `image_url` smoke,
including a held text prefix and generation resumed after the image response.

Raw evidence and detailed interpretation:

- [Vision K2 speed](results/benchmark-vision-k2-k3-tp1-20260903-final.json), [content](results/content-types-vision-k2-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k2-k3-tp1-20260903-final.json), [XGrammar canary](results/issue136-vision-k2-k3-tp1-20260903-final.json), and [image smoke](results/vision-smoke-vision-k2-k3-tp1-20260903-final.json)
- [Vision K2.2-D2 speed](results/benchmark-vision-k22-d2-v1-k3-tp1-20260903-final.json), [content](results/content-types-vision-k22-d2-v1-k3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-vision-k22-d2-v1-k3-tp1-20260903-final.json), [parallel-4 Tool Eval control](results/tool-eval-vision-k22-d2-v1-k3-tp1-p4-20260903.json), [XGrammar canary](results/issue136-vision-k22-d2-v1-k3-tp1-20260903-final.json), and [image smoke](results/vision-smoke-vision-k22-d2-v1-k3-tp1-20260903-final.json)
- [0731 K2-v1 speed](results/benchmark-old-k2-v1-tp1-20260903-final-rc.json), [content](results/content-types-old-k2-v1-tp1-20260903-final-rc.json), [Tool Eval](results/tool-eval-old-k2-v1-tp1-20260903-final-rc.json), and [XGrammar canary](results/issue136-old-k2-v1-tp1-20260903-final-rc.json)
- [0731 K2.1-D2.2 speed](results/benchmark-old-k21-d22-v3-tp1-20260903-final.json), [content](results/content-types-old-k21-d22-v3-tp1-20260903-final.json), [Tool Eval](results/tool-eval-old-k21-d22-v3-tp1-20260903-final.json), and [XGrammar canary](results/issue136-old-k21-d22-v3-tp1-20260903-final.json)
- [Detailed four-model comparison](20260903-mia-all-four-compare.md), [first Vision release comparison](20260903-mia-vision-k2-compare.md), and [numerics analysis](NUMERICS_FOR_VISION_UPDATES.md)

### Historical 2026-08-24 0731 quant comparison

All figures below were measured on one DGX Spark per model with the same
published image and launch profile: 1,000,000 max model length, six sequences,
8,192 batched tokens, 0.85 memory utilization, NVFP4 DS-MLA cache, and five
dSpark proposal tokens. No two-Spark measurements are included.

### Repository speed sweep

The final sweep used fresh run IDs and restarted servers to clear prefix cache
state. It covered prompt targets 256, 2K, 8K, 32K, and 131K at concurrency
1/2/4/6, thinking off, with natural completion. `Prefill / TTFT` cells are C1.

| Model | 256 C1 decode | 256 C6 aggregate | 256 C6 median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K2-v0 | 49.1 tok/s | 134.7 tok/s | 25.7 tok/s | 1,161 tok/s / 1.79 s | 1,359 tok/s / 6.05 s | 1,382 tok/s / 23.73 s | 1,287 tok/s / 101.83 s |
| K2-v1 | **58.1 tok/s** | 135.2 tok/s | 26.3 tok/s | **1,148 tok/s / 1.81 s** | **1,376 tok/s / 5.98 s** | **1,394 tok/s / 23.53 s** | **1,292 tok/s / 101.47 s** |
| K2.1-v2 | 56.8 tok/s | **167.9 tok/s** | **32.0 tok/s** | 732 tok/s / 2.84 s | 824 tok/s / 9.98 s | 825 tok/s / 39.75 s | 789 tok/s / 166.19 s |

The 58.1 tok/s K2-v1 number is the repository's short synthetic decode case:
a 256-token structured numbered-word prompt that emits 513 tokens. It is a
useful controlled decode measurement, not a varied prose average. K2.1 decode
is essentially in the K2 band; its clear cost is prefill, about 39% below K2
over the longer C1 prompts.

### DS4RT content types

These are median visible-token decode rates over five repeats of the seven
weighted prompts from `../ds4rt`, plus its repeated-orchid workload. Thinking
was off and temperature was zero.

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

Values are tokens/s. Output lengths differ between models, so rows such as the
very short `hello` response are sensitive to small timing differences. Orchid
correctness failed for every model: all 15 timed trials produced 1,499
occurrences and hit the 1,500-token cap instead of stopping at exactly 100.

### Tool Eval Bench

Tool Eval Bench `2.3.2.dev3+g5df1e9e0c` ran from the workstation against each
Spark: all 69 standard scenarios, temperature zero, seed zero, thinking
enabled, sequential execution, and a 2026-08-24 reference date.

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| K2-v0 | 120/138 | 87/100 | 56 / 8 / 5 | 0 |
| K2-v1 | **123/138** | **89/100** | 57 / 9 / 3 | 0 |
| K2.1-v2 | 120/138 | 87/100 | 55 / 10 / 4 | 0 |

All three completed the entire suite without an API error. None passed the
safety gate: K2-v0 triggered warnings on TC-34/43/58, K2-v1 triggered the
critical cross-turn sleeper injection at TC-60, and K2.1-v2 triggered TC-34
plus TC-60. The numeric score is therefore not a safety-clearance claim.

Raw evidence and the fuller interpretation are retained in the repository:

- [K2-v0 speed](results/benchmark-k2-v0-tp1-20260824.json), [content](results/content-types-k2-v0-tp1-20260824.json), and [Tool Eval](results/tool-eval-k2-v0-tp1-20260824.json)
- [K2-v1 speed](results/benchmark-k2-v1-tp1-20260824.json), [content](results/content-types-k2-v1-tp1-20260824.json), and [Tool Eval](results/tool-eval-k2-v1-tp1-20260824.json)
- [K2.1-v2 speed](results/benchmark-k21-v2-tp1-20260824.json), [content](results/content-types-k21-v2-tp1-20260824.json), and [Tool Eval](results/tool-eval-k21-v2-tp1-20260824.json)
- [Detailed 2026-08-24 comparison](20260824-mia-kX-compare.md)

## What changed from Mia's recipe

| Piece | This fork |
| --- | --- |
| Runtime | Mia/Anemll vLLM `0.25.2.dev0+g752a3a504.d20260714`, pinned by digest |
| Upstream fixes | Mia's selected vLLM 0.27 backports, structured-output fix, and long-NVFP4-decode dispatch fix |
| Expert weights | Standard-HF calibrated EXL3 K2 or per-expert K2/K3 target and draft experts |
| Expert kernels | Current b12x/Trellis serving fork with mixed-tier kernels; Mia's native modules retained |
| Loader | Index-aware InstantTensor streaming with bounded CUDA allocator retention on unified memory |
| Topology tested here | One Spark using the `uni` executor |
| Draft | DeepSeek's own dSpark block, not a REAP-pruned or external draft model |

The EXL3 source image is used only as a Docker source stage for the quantizer
module. The final runtime inherits from Mia's pinned Anemll image.

## Build and lineage

```bash
docker build --progress=plain \
  -t ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest .
```

The projection-mixed release is tagged `2026-09-03-projection-mixed`,
`763f65b`, and `latest`:

```text
ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:41fa9e86768b48dd5fa6d6f29bfacbd3bec3b2bbc1711d80751ff37c2905dbf8
```

The first Vision K2 release remains tagged `2026-09-03-vision-k2` and
`7c7306f`:

```text
ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:9bd058d1b91fc8d9164b0cf45ed8355fdbd5a05a3715ec38c0d5a67163dd1b60
```

The historical 2026-08-24 image is also tagged `2026-08-24-k21`:

```text
ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:40c9fa96b23184c260ebf1213c747afe54b5ad0a8b8686292aca209397507548
```

- Mia base image: `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- b12x serving fork: `e0f439532ce3e72c193803c128ba57e46dfd8ea2`
- K2-v0 model revision: `dff9afc6f5fe50a890590f7b6d5339ceaf5ba51e`
- K2-v1 model revision: `68eaca43e99bfbfd697a5559c7796b983deb38f8`
- K2.1-v1 config revision: `73757f619a951d812fe8008a39dbade8df20e6c6`
- K2.1-v2 config revision: `a2b066719ebdc0cbb0eacc752ffe7a2190c919aa`
- Vision K2 model revision: `c171bea574201ff25530256fbd63626c7fd20f3c`
- Vision K2.2-D2 model revision: `8347bfb8776287ef2dcab2b46e9f15c655825c3a`
- 0731 K2.1-D2.2 model revision: `7827301eed170e2a5e394f45a13cc66561c601ed`

Mia's original field notes remain in the
[upstream recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).
The people and projects behind the stack are linked in [CREDITS.md](CREDITS.md).

## License

Recipe code and documentation follow this repository's MIT license. Models,
containers, CUDA components, vLLM, b12x, and other upstream components retain
their own licenses and terms.
