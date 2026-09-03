# Numerics and synchronization changes through the Vision release

Date: 2026-09-03

## Bottom line

The measured Tool Eval change is real, but it is not evidence that adding
Vision itself cost six or seven points.

The strongest control is the unchanged 0731 K2-v1 checkpoint:

| Run | Checkpoint | Runtime generation | Points | Overall | Pass / partial / fail |
| --- | --- | --- | ---: | ---: | ---: |
| 2026-08-24 | 0731 calibrated K2-v1 | previous release | **123/138** | **89** | 57 / 9 / 3 |
| 2026-09-03 | 0731 calibrated K2-v1 | Vision release | **117/138** | **85** | 54 / 9 / 6 |
| 2026-09-03 | Vision-Exp K2-v1 | Vision release | **116/138** | **84** | 55 / 6 / 8 |

Thus:

- the same checkpoint moved by **-6 points** between releases;
- changing only from 0731 K2-v1 to Vision K2-v1 on the September runtime was
  **-1 point**;
- the two September models changed outcomes in both directions across eight
  scenarios, not as a uniform degradation.

The release interval did include a very large B12X MoE/attention update and a
few synchronization and structured-output corrections. Exact numerical
equivalence with the August kernel stack should not be expected. However, the
two benchmark invocations also used different reference dates and not
byte-identical harness builds, and each score is only one trial. The available
data cannot allocate the six points between kernel numerics, benchmark-input
changes, and ordinary generation sensitivity.

Neither scored September checkpoint used the new per-projection mixed-Trellis
route: both are uniform K2. The projection-mixed adapter and K2/K3 tier kernels
therefore cannot explain these scores directly. The newer B12X pin still
matters because the uniform W4A16/Trellis and attention paths also changed.

## Exact comparison boundary

The repository boundary used here is:

- previous release commit
  `f20b97dfd7666c00c316f29542e2e53f33cabb19` (2026-08-24), documented image
  `sha256:40c9fa96b23184c260ebf1213c747afe54b5ad0a8b8686292aca209397507548`;
- Vision release commit
  `7c7306f972c762b303c04f2c24f1071a55b63916` (2026-09-03), published image
  `sha256:9bd058d1b91fc8d9164b0cf45ed8355fdbd5a05a3715ec38c0d5a67163dd1b60`.

The Mia/Anemll base image did **not** change. Both Dockerfiles resolve
`ghcr.io/anemll/dspark-vllm-gx10` to
`sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`,
which contains vLLM `0.25.2.dev0+g752a3a504.d20260714`. The EXL3 source-stage
image digest also stayed fixed. This rules out a silent base-vLLM, CUDA,
PyTorch, or FlashInfer image bump.

The principal numerical runtime change was the B12X pin:

- August: `28e083482fd18ca3ce0e2553cd533102be85552f`
- Vision: `3fc8d1491d1313c0ca64b2b95772972b7f42ee9d`

Across the endpoint trees, the MoE and attention directories alone changed
126 files, with 31,629 insertions and 9,591 deletions. This was a substantial
kernel/runtime evolution, not a documentation-only dependency bump.

## Numerics-oriented changes

### 1. W4A16 MoE decode dispatch changed

The old B12X endpoint used the fused direct top-k tensor-core decode path for
all supported small token counts. The new endpoint decides between direct
decode and expert-packed routing from routed-row count, expert count, and SM
coverage (`e89b794d`, refined by `2edb44a2`).

That matters to this benchmark because dSpark verification creates small
multi-token MoE batches even when Tool Eval itself sends one request at a
time. The two paths implement the same mathematical MoE, but they do not
necessarily perform products, reductions, and the top-k weighted sum in the
same order. BF16/FP16 roundoff near a token-logit boundary can therefore
change the greedy token and all later tool behavior.

The new route-packed path also contains a correctness fix for the top-k scale
lookup (`5c831800`): it indexes the scale by the same local row used for route
metadata rather than the prior metadata-row value. This is an intended
correction, but it is also a direct arithmetic change whenever that packed
path is selected.

### 2. Trellis prepared-weight dtype became explicit

The Vision adapter update plans and prepares uniform full-rotation
`trellis_t256` payloads with B12X's FP16 storage contract instead of passing
vLLM's live BF16 activation dtype through as the prepared-parameter dtype.
B12X then converts the output back to the caller dtype. Relevant B12X changes
include the BF16 full-rotation epilogue/output work and separation of live
activation dtype from prepared dtype (`f7806111`, `67877158`, `4f2a7841`,
`692ef4a1`, and `bea0c181`).

This is a real precision-boundary change in the integration even though it
implements the current B12X contract. It can move low bits in rotations,
scratch values, or epilogue conversion. It does not alter the stored EXL3 K2
atoms in the Hugging Face checkpoint.

The clamped SwiGLU operation is **not** a newly introduced difference between
the two release endpoints. Both endpoint kernels already clamp the logical
gate/up values for these models (`swiglu_limit=10.0`). A later-history commit
named “Restore clamped SwiGLU” is visible in the merged graph, but the actual
August endpoint already contains the equivalent clamp, so it should not be
used to explain this score delta.

### 3. Trellis prefill planning was bucketed

The adapter changed from one maximum-size 8,192-token Trellis prefill plan to
bounded 512, 2,560, and 8,192 capacity plans, selecting the smallest fitting
plan. This fixed a major throughput collapse for ordinary 2K prompts.

Capacity selection can also select a different launch geometry for a given
prompt and hence a different accumulation order. The intended arithmetic is
the same, but bitwise identity is not guaranteed. Tool Eval prompts and their
multi-turn histories can cross these bucket boundaries.

### 4. NVFP4 dual-cache attention reads were corrected

B12X `7bf68c9b` changed DSV4 NVFP4 dual-cache prefill so RoPE values switch to
the extra cache's base pointer, page size, and block stride together with the
indices and latent data. Previously, extra-cache tiles could read unrelated or
out-of-range RoPE rows from the main pool. This is a correctness fix and a
direct attention-numerics change for requests that exercise those extra
tiles.

The final B12X commit, `3fc8d149`, additionally supports Vision's wider
dual-cache prefill geometry. That path is Vision-specific and is necessary to
run the Vision checkpoint; it cannot explain the full -6 on the unchanged
0731 checkpoint.

### 5. Vision uses a different speculative shape

The 0731 model has one next-token-prediction layer and retains five dSpark
proposal tokens. Vision has three prediction layers and requires a proposal
count divisible by three. The scored first Vision release used six proposal
tokens. At the shipped six-sequence limit, its automatically derived
CUDA-graph capture ceiling was 48, versus 36 for 0731.

Speculative decoding is intended to preserve the target model's distribution,
but the different verification width changes batching, graph shapes, and the
small-M MoE path selected above. At temperature zero, even a very small target
logit change can select a different first token. The resulting autoregressive
trajectory can then diverge completely.

This is a plausible contributor to the **one-point** same-runtime difference
between Vision and 0731. It does not explain the old checkpoint's August to
September change, because 0731 remained at five proposal tokens and a
36-token capture ceiling.

### Post-release K3 and mixed-projection follow-up

The projection-mixed qualification found that six is not the best proposal
width for Vision on this runtime. DSpark does **not** recursively run two
three-token draft cycles: the installed `DSparkSpeculator` runs one parallel
draft forward for all requested positions. Even so, K6 makes the three draft
layers process six positions and makes the target verify seven positions,
whereas K3 processes three and verifies four.

On the same Vision K2.2-D2 model and fixed short benchmark prompt, three warm
K6 trials averaged 32.80 visible decode tok/s. Three K3 trials averaged 38.92
tok/s, an 18.7% gain. The K3 run recorded 1,133 accepted tokens from 1,539
proposed tokens (73.6% aggregate acceptance); acceptance by proposed position
was 85.0%, 69.4%, and 66.5%. The additional K6 work therefore did not repay
its wider draft and verification cost on this model. The follow-up recipe
sets Vision to K3 with a capture ceiling of 24 and leaves 0731 at K5/capture
36. No partial K5 implementation or second draft pass was introduced.

That follow-up also advances B12X from `3fc8d149` to `e0f43953`. The new pin
lets projection-mixed Trellis choose its live-shape tile instead of forcing
the packed-prefill `(128,128,128,128)` tile into direct decode. This improved
the measured mixed checkpoints but is a host-planning/performance correction;
it was not present in, and cannot explain, the earlier 116/138 and 117/138
uniform-checkpoint Tool Eval scores analyzed above.

## Synchronization and state-machine changes

### B12X intra-kernel synchronization

The B12X interval includes several changes intended to make routed execution
capture-safe and to prevent stale/aliased route state:

- route histograms are preallocated for CUDA-graph capture (`85d3681b`);
- fixed route arenas are reused safely for prefill tails (`56ab5f4a`);
- inactive routes and mapped route namespaces are sanitized/preserved
  (`c25cdba2`, `3a6a204b` and related endpoint changes);
- when a small-route histogram aliases packed-route storage, route packing now
  places a barrier between consuming the histogram and overwriting the same
  allocation;
- W4A16 epilogues use explicit thread synchronization at the relevant shared
  memory boundary in the new endpoint.

These are primarily correctness and race-avoidance changes. If the old path
ever observed stale route metadata, the new output is expected to differ.
Even without an old race, changing kernel partitioning and barriers can change
which reduction implementation is used.

### Triton specialization and mid-serve JIT prevention

The release adds the issue-133 patch to prevent traffic-dependent pointer
alignment specialization in DSV4's global-top-k metadata kernel. The affected
pointers are scalar-loaded, so the patch intentionally changes compilation
keys, not math. It reduces six observed persistent-cache variants to the two
real block-size variants and lowers the chance that one execution rank pauses
for JIT compilation during live traffic.

For a one-Spark `uni` server this is mainly availability/determinism hygiene;
it is not a credible direct explanation for a six-point quality change.

### Shared-memory ring recovery

The issue-117 backport changes vLLM's multi-reader shared-memory broadcast
protocol in two ways:

- reader release bookkeeping and its memory fence run in a `finally` block;
- an idle reader wakes at most every five seconds to re-read the authoritative
  shared-memory written flag, recovering from a lost notification.

This prevents writer/reader deadlocks and rank loss. Tool Eval used the
one-Spark `uni` executor, so it does not have the TP=2 reader/writer topology
that motivated the fix. It should be treated as a liveness fix, not as an
arithmetic or likely score-changing change for these runs.

### XGrammar termination state

The Vision release enables the exact upstream issue-136 XGrammar correction by
default. It advances only through grammar-constrained tokens, stops advancing
at grammar termination, treats already-terminated acceptance as successful,
and resets the cached termination bit explicitly. This changes structured
output control state, not model logits.

The local A/B observations do not show a consistent score penalty:

| Checkpoint | Before fix | After fix | Observed delta |
| --- | ---: | ---: | ---: |
| 0731 K2-v1 | 118/138 | 117/138 | -1 |
| Vision K2-v1 | 114/138 | 116/138 | +2 |

Those were separate generations, so the score movements should not be
attributed to the patch. The valid first-release conclusion is operational:
after the fix, 0731 K2-v1 and the then-current uniform Vision K6 profile passed
the dedicated 145/145 structured-output canary without the post-termination
matcher warnings.

### Projection-mixed follow-up

After the projection-mixed loader and B12X live-shape tile selector landed,
the final four-way Tool Eval run scored Vision uniform K2 at 113/138, Vision
K2.2-D2 at 118/138, 0731 uniform K2-v1 at 119/138, and 0731 K2.1-D2.2 at
122/138. All completed 69 scenarios with zero API errors. These are
checkpoint-quality observations, not evidence that the mixed kernel changes
logits: mixed serving retains and executes the checkpoint's intended K2/K3
projection map, so its weights are numerically different by design.

The Vision default consequently changes to K2.2-D2 for its five-point lead.
Uniform Vision K2 remains the capacity profile: its startup allocation exposed
2,162,501 KV tokens versus 1,227,358 for K2.2-D2, approximately 76% more. That
default decision is therefore explicitly quality versus KV headroom, not a
claim that one representation dominates every workload.

## What actually moved in Tool Eval

For the same 0731 K2-v1 checkpoint, seven of 69 scenarios changed points:

| Scenario | Aug | Sep | Delta | September behavior |
| --- | ---: | ---: | ---: | --- |
| TC-11 | 2 | 1 | -1 | Used the calculator for trivial arithmetic |
| TC-14 | 1 | 2 | +1 | Recovered from stock-tool failure and surfaced a price |
| TC-30 | 1 | 2 | +1 | Completed the conditional two-call chain |
| TC-32 | 2 | 0 | -2 | Failed the impossible spam-clearing restraint case |
| TC-39 | 2 | 1 | -1 | Used a tool for trivial arithmetic |
| TC-42 | 2 | 0 | -2 | Added parameters forbidden by the tool schema |
| TC-68 | 2 | 0 | -2 | Called tools in a no-tool structured-output case |

The positive changes sum to +2 and the negative changes to -8, producing the
net -6. The largest category movements were Safety & Boundaries (-4) and
Structured Output (-2). These are semantic decisions several generated tokens
downstream, not signs of a single obvious NaN, API failure, or broken tool
parser. Both runs completed all 69 scenarios with zero API errors.

On the common September runtime, changing 0731 to Vision changed eight
scenarios and netted only -1: Vision gained TC-11, TC-32, TC-39, and TC-56,
while losing TC-34, TC-43, TC-57, and TC-58. Again, this is bidirectional
behavior around decision boundaries rather than a blanket loss of tool-call
ability.

## Benchmark confounders

The August and September scores are useful release observations, but they are
not a controlled numerical A/B:

1. The Tool Eval reference date changed from `2026-08-24` to `2026-09-03`.
   The configuration fingerprints consequently differ
   (`ff05903547a5` versus `63c54d5f3ebc`), and date-sensitive scenario inputs
   are not identical.
2. August reports Tool Eval Bench
   `2.3.2.dev3+g5df1e9e0c`; September reports
   `2.3.2.dev3+g5df1e9e0c.d20260903` and `5df1e9e-dirty`. They share the base
   commit, but the recorded harness artifacts are not byte-identical.
3. Each number is one trial. `temperature=0` and `seed=0` do not guarantee
   cross-kernel bitwise identity in an autoregressive MoE model, especially
   when speculative verification and reduction geometry change.
4. The September XGrammar before/after runs themselves moved -1 for 0731 and
   +2 for Vision despite no checkpoint change. That is a small direct example
   of single-run outcome sensitivity.

The user-observed result on vLLM 0.28 is not stored with enough matching
metadata in this repository to make a numeric comparison here. The current
release may be better than that observed run, but this document does not turn
it into an apples-to-apples claim.

## Likely interpretation and next isolation test

The most defensible interpretation is:

- **Vision checkpoint effect:** small in this sample (-1 versus 0731 on the
  same runtime).
- **Release/runtime effect:** potentially material because the B12X MoE and
  attention implementation changed substantially, including small-M dispatch,
  route-weight indexing, dtype boundaries, and dual-cache RoPE reads.
- **Synchronization fixes:** important for avoiding corruption, deadlock, and
  mid-serve compilation, but mostly correctness/liveness improvements rather
  than expected quality regressions.
- **Measured -6:** not yet attributable, because the reference date, harness
  artifact, and kernel stack all changed and there is only one trial.

To isolate the regression, rerun 0731 K2-v1 with the exact same Tool Eval
checkout, fixed reference date, prompts, seed, cache state, and request order
against only these two image/B12X pairs. Run at least three trials per image
and retain per-scenario raw logs. If the six-point gap persists, bisect the
B12X pin first; within that interval, compare direct small-M TC decode against
expert-packed routing before investigating the synchronization-only patches.

## Repository evidence

- [August comparison](20260824-mia-kX-compare.md)
- [Vision comparison](20260903-mia-vision-k2-compare.md)
- [August 0731 K2-v1 Tool Eval](results/tool-eval-k2-v1-tp1-20260824.json)
- [September 0731 K2-v1 Tool Eval](results/tool-eval-old-k2-v1-tp1-20260903.json)
- [September Vision K2-v1 Tool Eval](results/tool-eval-vision-k2-tp1-20260903.json)
- [0731 pre-fix XGrammar run](results/tool-eval-old-k2-v1-tp1-20260903-xgrammar-baseline.json)
- [Vision pre-fix XGrammar run](results/tool-eval-vision-k2-tp1-20260903-xgrammar-baseline.json)
- [EXL3/B12X adapter](patches/port-exl3-mixed.py)
- [Issue-133 Triton specialization patch](patches/hotfix-dsv4-issue133-triton-specialization.py)
- [Issue-117 shared-memory ring patch](patches/hotfix-vllm-issue117-shm-ring-buffer.py)
- [Issue-136 XGrammar termination patch](patches/hotfix-vllm-issue136-xgrammar-termination.py)
