# DeepSeek V4 Flash EXL3 K2 and K2.1 on DGX Spark

Mia built the Spark race car. This fork gives it a much smaller fuel tank.

This recipe starts from
[MiaAI-Lab's DeepSeek V4 Flash DSpark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
and keeps her serving stack intact: the Anemll GX10 vLLM runtime, native
dSpark, NVFP4 DS-MLA cache, async scheduling, RoCE transport, regular CUDA
graphs, and her selected DeepSeek V4 fixes from vLLM 0.27. This fork adds
standard Hugging Face EXL3 checkpoints that fit on **one DGX Spark**, including
uniform K2 and mixed K2/K3 (the 2.1-bit K2.1 variants).

The default launch serves
[`wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1`](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1)
with a **1,000,000-token** request ceiling, six active sequences, 0.85 GPU
memory utilization, and the checkpoint's own five-token dSpark draft. Nothing
is rank-sliced on disk; TP2 slices the ordinary checkpoint while loading.

The same image also serves the other calibrated variants:

| Launch name | Checkpoint | Expert layout |
| --- | --- | --- |
| `k2-v0` | `DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0` | Uniform K2, original top-6/legal calibration |
| `k2` / `k2-v1` | `DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1` | Uniform K2, rare-expert fallback and math calibration |
| `k21-v1` | `DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v1` | Per-expert K2/K3 mix in the target and draft |
| `k21` / `k21-v2` | `DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2` | Per-expert K2/K3 target with a forced-K2 dSpark draft |

K2.1 is not rounded to K2 or K3. The loader reads each expert's integer
`bits_per_weight` metadata and dispatches the K2 and K3 tiers separately. The
checkpoint-wide realized average remains descriptive metadata, so the embedded
Hugging Face `quantization_config` intentionally omits its non-integer `bits`
field.

| Configuration | C1 TTFT | C1 stream | C4 aggregate | C4 stream | 131K prefill | 131K TTFT | Tool Eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1× Spark, calibrated K2 v1 | 346 ms | 37.7 tok/s | 76.2 tok/s | 20.2 tok/s | 1,354 tok/s | 96.8 s | 87/100 · 120/138 |
| 2× Spark TP2, calibrated K2 v1 | **247 ms** | 53.7 tok/s | **125.2 tok/s** | **36.8 tok/s** | **1,674 tok/s** | **78.3 s** | 88/100 · 122/138 |
| 2× Spark TP2, native MXFP4 | 287 ms | **79.5 tok/s** | 122.2 tok/s | 36.5 tok/s | 1,665 tok/s | 78.7 s | **91/100 · 126/138** |

The decode columns come from the capture-era
[sparkDash decode benchmark](https://github.com/MiaAI-Lab/sparkDash/tree/ca5d55b663a674ec9c6df3beed698bdb4b7f7bbf),
on warmed servers with Mia's 1M-context launch profile. The 131K columns are
the C1 natural-completion prefill sweep over 131,096 actual prompt tokens.
Tool Eval used the complete 69-scenario Tool Eval Bench 2.3.2.dev3 suite at
temperature zero, thinking enabled, sequential execution, and no decode
assistance; every run completed with zero API errors. Raw results:
[`K2 TP1`](results/tool-eval-k2-v1-tp1.json),
[`K2 TP2`](results/tool-eval-k2-v1-tp2.json), and
[`native TP2`](results/tool-eval-native-tp2.json).
Mia's original native TP2 capture reported 82.4 tok/s at C1 and 120.4 tok/s
aggregate at C4; our 79.5/122.2 control puts this comparison in the same
neighborhood.

## The short route

The Spark should already have current NVIDIA drivers, Docker, the NVIDIA
Container Toolkit, and Docker Compose. Fetch the model into the normal HF
cache, pull the prebuilt runtime, and launch:

```bash
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1 \
  --revision 68eaca43e99bfbfd697a5559c7796b983deb38f8

docker pull ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest
git clone https://github.com/tpurtell/ds4-mia-exl3-k2-1spark.git
cd ds4-mia-exl3-k2-1spark
cp .env.example .env
./launch.sh --nodes 1 --model k2
```

Follow startup with:

```bash
docker logs -f ds4-mia-k2-tp1
```

Then say hello:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731-exl3-k2-calibrated-v1",
    "messages": [{"role": "user", "content": "Why are tiny bits so charming?"}],
    "stream": true
  }'
```

Docker Compose is also ready for the one-Spark default:

```bash
docker compose up -d
```

To try the recommended mixed checkpoint instead, cache it and select `k21`:

```bash
hf download \
  wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2 \
  --revision a2b066719ebdc0cbb0eacc752ffe7a2190c919aa

./launch.sh --nodes 1 --model k21
```

## Two Sparks, if they are sitting there looking bored

Put the same HF snapshot and container image on both machines. Set the host,
fabric IPs, and interface names in `.env`, then run from the head or any
machine that can SSH to both nodes:

```bash
./launch.sh --nodes 2 --model k2
```

The launcher resolves the RoCEv2 GID independently on both nodes, exposes
`/dev/infiniband`, and starts Mia's direct multi-node MP/TP2 path. If you want
the original DeepSeek MXFP4 checkpoint instead, cache
`deepseek-ai/DeepSeek-V4-Flash-0731` on both nodes and use:

```bash
./launch.sh --nodes 2 --model native
```

The native checkpoint does not fit on one Spark. The K2 checkpoint does.

Stop either layout with:

```bash
./stop.sh
```

## Defaults that matter

| Knob | Default | Why |
| --- | ---: | --- |
| `MODEL_KIND` | `k2` | Calibrated EXL3 K2 target plus its native dSpark draft |
| `DISTRIBUTED_EXECUTOR_BACKEND` | automatic | `uni` on one Spark; `mp` for multi-Spark launches |
| `MAX_MODEL_LEN` | `1000000` | Decimal one-million-token request ceiling |
| `MAX_NUM_SEQS` | `6` | Mia's low-concurrency agent-serving shape |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Chunked-prefill budget |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Plenty of KV without making graph capture exciting |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | Compact DeepSeek V4 hybrid cache |
| `DSPARK_TOKENS` | `5` | Full checkpoint draft block |
| `DEFAULT_THINKING` | `max` | Request-level `chat_template_kwargs` can override it |
| `PREFIX_CACHE` | `1` | Useful for real repeated agent prefixes |

The KV cache is a shared pool, not six eagerly allocated one-million-token
slots. Requests consume cache when admitted; excess work waits for room.

The launch path also protects the kernel compiler from empty Compose values:
`CUTE_DSL_ARCH` resolves to `sm_121a`, and memory/model-length knobs use
non-empty defaults even if their environment entries are blank.

## Performance

Three views are kept on purpose:

- The sparkDash decode benchmark is the headline apples-to-apples Spark
  comparison and matches the tool behind Mia's published screenshot.
- The fixed 2,048-prompt/128-output matrix forces identical work and is the
  cleaner engineering comparison with non-Spark engines.
- The natural-completion sweep exercises long prompt lengths and preserves the
  model's own stopping behavior.

### Mia/sparkDash decode comparison

This is sparkDash's original decode workload: randomly shuffled, distinct
structured JSON/HTML prompts; `max_tokens=2048`; greedy streaming; thinking
enabled; EOS ignored; and sequential C1–C4 waves. Decode begins at the first
visible token. Aggregate throughput is total decode tokens divided by the
shared first-to-last-token window, while stream throughput is the mean of the
individual streams.

All three configurations used this recipe's same image and Mia's captured
runtime profile: 1,048,576 max model length, four sequences, 8,192 batched
tokens, 0.835 memory utilization, NVFP4 DS-MLA cache, and six dSpark proposal
tokens. Each server was warmed before its recorded ladder.

| Configuration | Load | TTFT | Streams | Aggregate tok/s | Stream tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1× Spark, calibrated K2 v1 | C1 | 346 ms | 1/1 | 37.7 | 37.7 |
| 1× Spark, calibrated K2 v1 | C2 | 591 ms | 2/2 | 52.1 | 26.4 |
| 1× Spark, calibrated K2 v1 | C3 | 663 ms | 3/3 | 77.3 | 29.6 |
| 1× Spark, calibrated K2 v1 | C4 | 603 ms | 4/4 | 76.2 | 20.2 |
| 2× Spark TP2, calibrated K2 v1 | C1 | 247 ms | 1/1 | 53.7 | 53.7 |
| 2× Spark TP2, calibrated K2 v1 | C2 | 417 ms | 2/2 | 97.9 | 55.6 |
| 2× Spark TP2, calibrated K2 v1 | C3 | 472 ms | 3/3 | 105.9 | 37.5 |
| 2× Spark TP2, calibrated K2 v1 | C4 | 541 ms | 4/4 | 125.2 | 36.8 |
| 2× Spark TP2, native MXFP4 | C1 | 287 ms | 1/1 | 79.5 | 79.5 |
| 2× Spark TP2, native MXFP4 | C2 | 375 ms | 2/2 | 77.6 | 44.9 |
| 2× Spark TP2, native MXFP4 | C3 | 396 ms | 3/3 | 106.6 | 43.0 |
| 2× Spark TP2, native MXFP4 | C4 | 462 ms | 4/4 | 122.2 | 36.5 |

The collector intentionally randomizes prompt assignment, so modest run-to-run
movement is normal. The raw histories retain stream-level token counts and
timings:

- [`results/sparkdash-k2-v1-tp1-mia-profile.json`](results/sparkdash-k2-v1-tp1-mia-profile.json)
- [`results/sparkdash-k2-v1-tp2-mia-profile.json`](results/sparkdash-k2-v1-tp2-mia-profile.json)
- [`results/sparkdash-native-tp2-mia-profile.json`](results/sparkdash-native-tp2-mia-profile.json)

### Fixed-work engineering matrix

Fixed-work results use llama-benchy 0.4.0, 2,048 new prompt tokens, 128 forced
output tokens, depths 0/4,096/8,192, concurrency 1/2/4, three warm measured
runs, no prefix cache hits, greedy decoding, and thinking disabled.

| Depth | C | 1× Spark K2 | 2× Spark K2 | 2× Spark native |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 29.2 | 41.0 | **42.5** |
| 0 | 2 | 40.1 | **58.3** | 47.5 |
| 0 | 4 | 61.3 | **99.4** | 67.3 |
| 4,096 | 1 | 29.5 | 41.2 | **42.8** |
| 4,096 | 2 | 26.9 | **38.5** | 33.8 |
| 4,096 | 4 | 26.3 | **37.9** | 32.1 |
| 8,192 | 1 | 33.3 | **43.8** | 41.9 |
| 8,192 | 2 | 23.0 | **33.7** | 32.8 |
| 8,192 | 4 | 19.8 | 22.2 | **27.4** |

Values are aggregate generated tokens per second. The same run produced these
TTFT values:

| Depth | C | 1× Spark K2 | 2× Spark K2 | 2× Spark native |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1.60 s | **1.09 s** | 1.12 s |
| 0 | 4 | 5.60 s | **5.41 s** | 6.67 s |
| 4,096 | 1 | 4.32 s | 5.47 s | **3.20 s** |
| 4,096 | 4 | 12.20 s | **8.44 s** | 9.83 s |
| 8,192 | 1 | 7.23 s | 6.81 s | **5.26 s** |
| 8,192 | 4 | 19.46 s | 16.01 s | **15.49 s** |

The strict K5 acceptance counters over the fixed K2 runs were 35.32% on one
Spark and 34.76% on two. That is the same acceptance neighborhood—the TP2
throughput win comes from runtime scheduling and placement, not different
weights or a friendlier draft.

### Natural-completion and long-context sweep

The included benchmark client can explicitly disable thinking so a
template-default change does not quietly turn an apples-to-wax-apples
comparison into fruit salad:

```bash
python3 scripts/benchmark-0731.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731-exl3-k2-calibrated-v1 \
  --thinking off \
  --output results/mia-k2-v1-1spark-1m.json
```

DeepSeek sometimes ends these prompts early. The raw files therefore retain
every observed output length; aggregate throughput from a short completion is
not dressed up as a fixed decode result.

The recipe's natural-completion ladder tells a complementary story. Each cell
below is `median prefill tok/s / median TTFT` at C1; unlike the fixed matrix,
the model is allowed to stop when it is done.

| Configuration | 2K prompt | 8K prompt | 32K prompt | 131K prompt |
| --- | ---: | ---: | ---: | ---: |
| 1× Spark K2 | 1,267 / 1.64 s | 1,431 / 5.74 s | 1,461 / 22.45 s | 1,354 / 96.79 s |
| 2× Spark K2 | 1,549 / 1.34 s | 1,000 / 8.22 s | 1,452 / 22.59 s | **1,674 / 78.32 s** |
| 2× Spark native | **1,992 / 1.04 s** | **2,155 / 3.81 s** | 942 / 34.83 s | 1,665 / —¹ |

For one high-concurrency calibration point, observed C6 aggregate output was
76.3/54.9/18.3 tok/s on one-Spark K2, 138.7/64.4/16.1 tok/s on two-Spark K2,
and 143.1/76.7/27.6 tok/s on native TP2 at 2K/8K/32K prompts. Those rates are
not forced-work scores: output lengths vary, and the raw request records are
the source of truth.

Raw benchmark evidence:

- [`results/mia-k2-v1-spark-tp1-mia-runtime.json`](results/mia-k2-v1-spark-tp1-mia-runtime.json)
- [`results/mia-k2-v1-spark-tp2-mia-runtime.json`](results/mia-k2-v1-spark-tp2-mia-runtime.json)
- [`results/mia-native-spark-tp2-mia-runtime.json`](results/mia-native-spark-tp2-mia-runtime.json)
- [`results/mia-k2-v1-1spark-1m.json`](results/mia-k2-v1-1spark-1m.json)
- [`results/mia-k2-v1-1spark-1m-long.json`](results/mia-k2-v1-1spark-1m-long.json)
- [`results/mia-k2-v1-2spark-1m.json`](results/mia-k2-v1-2spark-1m.json)
- [`results/mia-k2-v1-2spark-1m-long.json`](results/mia-k2-v1-2spark-1m-long.json)
- [`results/mia-native-2spark-1m.json`](results/mia-native-2spark-1m.json)

## What changed from Mia's recipe

| Piece | This fork |
| --- | --- |
| Runtime | Mia/Anemll vLLM `0.25.2.dev0+g752a3a504.d20260714`, pinned by digest |
| DeepSeek fixes | Mia's selected vLLM 0.27 backports, including the structured-output boundary fix, baked into the image |
| Long NVFP4 decode | Mia Issue #22 fast-dispatch fix, baked and asserted at build time |
| Expert weights | Standard-HF calibrated EXL3 K2 or per-expert K2/K3 target and dSpark experts |
| Expert kernels | Current b12x/Trellis serving fork with mixed-tier kernels; Mia's native MXFP4 modules retained |
| Loader | Index-aware InstantTensor streaming with bounded CUDA allocator retention on unified memory |
| Topology | K2/K2.1 default to one Spark; EXL3 TP2 and native TP2 are launch options |
| Draft | DeepSeek's own dSpark block—not a REAP-pruned or external draft model |

The EXL3 source image is used only as a Docker source stage for the quantizer
module. The final runtime inherits from Mia's pinned Anemll image, so swapping
in K2 does not swap out the serving engine we came here to test.

## Build it yourself

The public image is the quick path, but the build is reproducible:

```bash
docker build --progress=plain \
  -t ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest .
```

Both parent images, the b12x source revision, the model revisions, and the
runtime checks are pinned in the recipe. Build-time assertions verify the
EXL3 registry, native MXFP4 imports, and the NVFP4 fast decode dispatch.

## Runtime lineage

- Mia base image:
  `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- b12x serving fork commit:
  `28e083482fd18ca3ce0e2553cd533102be85552f`
- calibrated K2 revision:
  `68eaca43e99bfbfd697a5559c7796b983deb38f8`
- calibrated K2.1 v1 config revision:
  `73757f619a951d812fe8008a39dbade8df20e6c6`
- calibrated K2.1 v2 config revision:
  `a2b066719ebdc0cbb0eacc752ffe7a2190c919aa`
- native checkpoint revision:
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

Mia's wonderfully detailed original field notes remain in the
[upstream recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).
The people and projects that made this stack possible are linked in
[`CREDITS.md`](CREDITS.md).

## License

Recipe code and documentation follow this repository's MIT license. Models,
containers, CUDA components, vLLM, b12x, and every other upstream component
retain their own licenses and terms.
