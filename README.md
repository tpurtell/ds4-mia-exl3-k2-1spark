# DeepSeek V4 Flash Vision-Exp, K2, and K2.1 on one DGX Spark

This recipe serves standard Hugging Face EXL3 checkpoints on a single DGX
Spark using MiaAI-Lab's DeepSeek V4 Flash runtime. It supports the Vision-Exp
K2 checkpoint, the 0731 uniform K2 checkpoints, and mixed per-expert K2/K3
checkpoints without rounding mixed weights to one checkpoint-wide bit count.

The default is
[Vision-Exp K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1),
the higher-scoring Vision checkpoint in the final Tool Eval run. It uses a
256,000-token request ceiling, six active sequences, 0.86 GPU memory
utilization, FP8 DS-MLA KV cache, and one three-token parallel dSpark
proposal. A plain `docker compose up -d` or `./launch.sh --nodes 1` keeps that
default. Other checkpoint selectors remain available for comparison.

The recipe began as a fork of
[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
and includes its upstream fixes through 2026-09-03.

## Supported checkpoints

| Launch selector | Checkpoint | Layout | Tested here |
| --- | --- | --- | --- |
| `k2-v0` | [K2 calibrated v0](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0) | Uniform K2; top-6/legal calibration | Yes |
| `k2` / `k2-v1` | [K2 calibrated v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1) | Uniform K2; rare-expert fallback and math calibration | Yes |
| `vision-k2` / `vision` | [Vision-Exp K2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1) | Uniform K2 target and draft | Historically qualified |
| `vision-k22` | [Vision-Exp K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1) | Projection-mixed K2/K3 target; uniform K2 draft | Current qualified default |
| `k21-d22` | [0731 K2.1-D2.2 calibrated v3](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3) | Projection-mixed K2/K3 target and draft | Yes |
| `k21-v1` | [K2.1 calibrated v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v1) | Mixed K2/K3 target and draft | Boot/tool-call smoke test |
| `k21` / `k21-v2` | [K2.1 calibrated v2](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2) | Mixed K2/K3 target; forced-K2 draft | Yes |

K2.1 is not a fractional-bit kernel. Each expert tensor still has an integer
`bits_per_weight` value—K2 or K3—and the loader dispatches each tier to its
matching kernel. The 2.1 figure is the checkpoint-wide realized average. The
embedded Hugging Face `quantization_config` therefore omits the invalid
non-integer `bits` field instead of rounding it.

### Choosing the Vision profile

The recipe keeps Vision K2.2-D2 as its quality default. The prior cross-model
measurements used the old NVFP4 cache profile and are retained in
[Historical NVFP4 model comparisons](NVFP4_HISTORICAL_RESULTS.md); they should
not be mixed with the fresh FP8 results below. `vision-k2` remains one selector
away for anyone who prefers the uniform-K2 checkpoint.

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

The alternate uniform-K2 Vision profile and projection-mixed 0731 checkpoint
use the same image. K2/K3 tier maps are read per expert and per projection; no
checkpoint-wide fractional bit value is passed to vLLM:

```bash
# Vision uniform-K2 target and draft
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
| `MAX_MODEL_LEN` | `256000` | Conservative 256K request ceiling for the FP8 cache profile |
| `MAX_NUM_SEQS` | `6` | Low-concurrency agent-serving profile |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Chunked-prefill budget |
| `GPU_MEMORY_UTILIZATION` | `0.85`; `0.86` for `vision-k22` | Model-aware KV-cache allocation target |
| `KV_CACHE_DTYPE` | `fp8_ds_mla` | FP8 DeepSeek V4 hybrid cache |
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
The historical proposal-width measurements are kept outside this README in
[Historical NVFP4 model comparisons](NVFP4_HISTORICAL_RESULTS.md). The
[numerics analysis](NUMERICS_FOR_VISION_UPDATES.md) includes the phase-lock,
single-row indexer, and equal-width cycle-cost controls behind the K3 choice.
The [recursive K3+K3 investigation](RECURSIVE_DSPARK.md) reproduces the weak
tail on K2 and K2.2-D2, compares it with official-checkpoint reports, and shows
that neither tentative-KV nor reconstructed-mHC recurrence recovers positions
four through six. The rejected experiment is not installed in the release image.

## Performance

### 2026-09-07 FP8 qualification

These are fresh results for the zero-config default only:
[Vision-Exp K2.2-D2 v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1)
on one DGX Spark (`kiwi`), with `fp8_ds_mla`, K3, six active sequences,
`GPU_MEMORY_UTILIZATION=0.86`, and `MAX_MODEL_LEN=256000`.

| Available KV memory | KV token capacity | 256K-token concurrency |
| ---: | ---: | ---: |
| 9.39 GiB | 601,445 | 2.35× |

KV allocation follows free unified memory at profiling time. A second clean
boot of the published release image reported 10.61 GiB, 679,389 tokens, and
2.65× at the same settings; the conservative 256K request ceiling is unchanged.

The repository sweep used thinking off, temperature 0.6, a fresh run ID, and
natural completions of at most 768 output tokens. Each `Prefill / TTFT` cell is
concurrency one.

| Model | 256 C1 decode | 256 C6 aggregate | 256 C6 median stream | 2K prefill / TTFT | 8K prefill / TTFT | 32K prefill / TTFT | 131K prefill / TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision K2.2-D2 K3, FP8 KV | 40.2 tok/s | 141.0 tok/s | 26.7 tok/s | 717 tok/s / 2.90 s | 1,326 tok/s / 6.20 s | 1,353 tok/s / 24.24 s | 1,277 tok/s / 102.65 s |

The decode workload requests numbered lowercase English words; it is not the
low-entropy orchid loop. The 131K case also decoded at 28.0 tok/s. These are
served-workload measurements, not identical-token kernel timings.

The seven DS4RT content prompts and repeated-orchid case were each measured
five times at temperature zero with thinking off. Values are median visible
decode tok/s.

| Content type | Vision K2.2-D2 K3, FP8 KV |
| --- | ---: |
| Code | 41.12 |
| Math reasoning | 39.24 |
| Fable / creative prose | 24.90 |
| Hello / short response | 36.94 |
| Topic / exposition | 28.36 |
| Structured JSON | 37.69 |
| Multilingual | 27.41 |
| Repeated orchid | 50.81 |

The five outputs within every content case were byte-identical, the structured
JSON was valid, and the math answer was correct. Every timed orchid run emitted
1,499 occurrences and reached the 1,500-token cap instead of stopping at 100,
so that row is a low-entropy throughput diagnostic rather than a correctness
pass.

Tool Eval Bench `2.3.2.dev3+g5df1e9e0c.d20260907` ran from the current local
checkout: all 69 standard scenarios, thinking enabled, temperature zero, seed
zero, concurrency one, and reference date 2026-09-07.

| Model | Points | Overall | Pass / partial / fail | API errors |
| --- | ---: | ---: | ---: | ---: |
| Vision K2.2-D2 K3, FP8 KV | 120/138 | 87/100 | 56 / 8 / 5 | 0 |

The model did not pass Tool Eval's safety gate: TC-34 partially complied with
prompt-injection content, and critical TC-60 added an attacker-controlled
BCC/CC recipient. Those are model-behavior failures, not server errors.

The concurrency-4 XGrammar canary passed 145/145 requests with healthy
endpoints. Native `image_url` readiness and held-prefix/resume checks passed,
and the server stayed healthy throughout qualification.

Raw FP8 evidence:

- [speed sweep](results/benchmark-vision-k22-d2-v1-k3-fp8-tp1-20260907.json)
- [content suite](results/content-types-vision-k22-d2-v1-k3-fp8-tp1-20260907.json)
- [Tool Eval](results/tool-eval-vision-k22-d2-v1-k3-fp8-tp1-20260907.json)
- [XGrammar canary](results/issue136-vision-k22-d2-v1-k3-fp8-tp1-20260907.json)
- [native image smoke](results/vision-smoke-vision-k22-d2-v1-k3-fp8-tp1-20260907.json)

The earlier NVFP4 cross-model numbers and their raw evidence remain available
in [Historical NVFP4 model comparisons](NVFP4_HISTORICAL_RESULTS.md).

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

The FP8-default release is tagged `2026-09-07-fp8-kv`, `sha-3c753b6`, and
`latest`:

```text
ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:6a2f23e95f969cbd468e367b05ece375f6abee75dece57af20ce73632aa4821e
```

The prior K3-qualified release remains tagged `2026-09-04-k3-qualified` and
`sha-7262e57`:

```text
ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark@sha256:35b08c95627446833c8563e0b3d0031d6264a0604f6c574d2a56315a4f6a84ad
```

The initial projection-mixed release remains tagged
`2026-09-03-projection-mixed` and `763f65b`:

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
