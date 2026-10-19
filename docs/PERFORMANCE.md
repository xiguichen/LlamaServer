# LlamaServer performance baseline

**Why this file exists.** Performance statements about this server must be
*measured on device*, not inferred by reading the code. On 2026-10-18 an
in-session coding agent read `LlamaInference.swift` and reported several
architecture limits that the device's own logs directly contradict (see
[Claims the logs disprove](#claims-the-logs-disprove)). Numbers below are from
that device's logs, with the reproducing query for each.

Anything not measured here is labelled **open question**. Add rows as they are
measured; do not replace a measurement with a theory.

## Rig

| | |
|---|---|
| Device | A19 Pro GPU (`ggml_metal_init: picking default device: Apple A19 Pro GPU`) |
| Build | `llamaserver-470957a` (every response carries `system_fingerprint`) |
| Model | `Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf` (4B, Q4_K_M) |
| Model source | [Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF](https://huggingface.co/Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF), base [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) — see [`STACK.md`](STACK.md) |
| Context | 131,072 tokens (hybrid SSM+attention, `can_shift=false`) |
| KV cache | q8_0, **17,408 B/token**, 8 of 32 layers, head_dim K=256 V=256 |
| Long context | YaRN x4.0 (`orig 32768`) |
| MTP | enabled, two-context speculative, output slots 2, `maxDraft` 1 |
| Load | 13.05 s (`backend init 9.60 + model load 2.34 + context 0.92 + mtp 0.19`) |

Traffic: pi (17 tools), 2026-10-18 12:22–12:57, single client, prompt growing
23.7k → 48.5k tokens across 15 turns.

## How to measure

```bash
curl 'http://<device>:8443/logs?name=inference&tail=6000&backups=1'   # engine internals
curl 'http://<device>:8443/logs?name=server&tail=4000&backups=1'      # request lifecycle
```

Then read, per turn:

- `generated N tokens (prefill=…, gen=…, …t/min, finish=…, cache now N)` — throughput.
- `KV snapshot restored (R tokens, prompt P); decoding P-R new` — how much was reused.
- `MTP stats: … ; draft Xms/rd + verify Yms/rd + … = wall Zms/rd` — where the round time went.

Rules for a usable number:

1. **One client at a time.** The inference queue is serial; concurrent requests contaminate the measurement.
2. **Compare like with like.** Generation speed depends on context length (below), so quote the context size with every t/s figure.
3. **Thermals matter more than you think.** Back-to-back turns at ~47k context measured `verify-gpu 148.81 ms/rd`; the same 48k context after a 28-minute idle measured `103.49 ms/rd` — a 30% swing with no code change.
4. **A/B must be paired.** For configuration comparisons (MTP on/off, KV quant), run the *same* prompt shapes in both configs, alternating, and state the order. Two unpaired runs prove nothing.
5. **The `prefill=` rate is over the decoded tokens**, not the prompt length (fixed 2026-10-18 — it used to divide the full prompt by the delta time and printed `15431.54t/s` for a 113-token decode). Read the `(X of Y, Z reused)` part before quoting it.

## Measured baseline

### Prompt phase ("prefill")

Decode of new prompt tokens; `prefill=` seconds enclose the prompt decode (and
the snapshot **save**, see below), not the snapshot restore.

| case | prompt | decoded | seconds | t/s |
|---|---|---|---|---|
| cold conversation, no usable snapshot | 23,699 | 23,699 | 78.48 | **302** |
| short prompt | 464 | 464 | 1.34 | 347 |
| short prompt | 56 | 56 | 0.33 | 171 |
| reuse, large turn | 26,049 | 2,357 | 11.32 | 208 |
| reuse, large turn | 35,074 | 3,307 | 20.70 | 160 |
| reuse, small turn | 47,557 | 532 | 4.42 | 120 |

Roughly **170–350 t/s** for genuinely new tokens, i.e. a cold 24k-token agent
prompt costs ~80 s. That is the single most expensive thing the server does.

The rate also falls with context: 344 t/s for 8,577 decoded tokens at an 8.9k
context → 271 at 17.7k → 205 at 26.6k → 184 at 35.4k (the A/B runs below, which
also show MTP-on paying roughly 9% more here than MTP-off).

### Generation

MTP on, `gen=` measures from the first sampled token (after the prompt GPU sync).

| context | tokens | seconds | t/s |
|---|---|---|---|
| 23.7k | 69 | 4.06 | 17.0 |
| 35.1k | 70 | 4.75 | 14.7 |
| 47.0k | 499 | 50.78 | 9.8 |
| 48.3k (cool) | 41 | 3.35 | 12.2 |
| 48.5k (cool) | 204 | 16.06 | 12.7 |

Roughly **10–17 t/s**, falling with context. Long generations dominate the
wall clock of an agent turn (the 499-token turn took 51 s to generate).

### The real cost: the verify pass, and it scales with context

`MTP stats` per round (one draft decode + one 2-token verify decode):

| approx. context | draft-gpu | **verify-gpu** | sample | emit | host | **wall/round** | emitted/round |
|---|---|---|---|---|---|---|---|
| 0.4k | 9.75 | 51.49 | 0.79 | 0.06 | 0.11 | 64.45 | 2.00 |
| 26k | 13.80 | 96.29 | 1.48 | 0.11 | 0.06 | 113.59 | 1.97 |
| 32k | 15.61 | 113.68 | 1.51 | 0.11 | 0.03 | 132.83 | 2.00 |
| 44k | 11.51 | 141.21 | 1.43 | 0.15 | 0.11 | 156.50 | 1.89 |
| 47k | 11.25 | 148.81 | 1.23 | 0.29 | 0.30 | 163.98 | 1.61 |
| 48k (cool) | 11.88 | 103.49 | 1.01 | 0.22 | 0.31 | 118.88 | 1.51 |

(all figures ms/round)

- **`verify-gpu` is 80–90% of every round** and grows with context
  (51 → 149 ms/round as context grew 0.4k → 47k). At 48k the verify pass
  attends over a KV of 48,450 × 17,408 B ≈ **843 MB**, plus ~2.5 GB of weights
  per pass. Reducing context or KV bytes is the only lever on this number.
- Host/CPU work is **negligible**: `draft + verify + sample` CPU is ~3 ms/round,
  `host` ~0.3 ms/round. Micro-optimising the Swift decode loop cannot help.
- MTP emits 1.5–2.0 tokens/round; acceptance (`hits`) tracked 36–100% and fell
  as context grew.

### MTP depth 1: paired on/off A/B (2026-10-18, build `46c215d`)

Identical prompt sequences in both configs (`temperature: 0`, so speculative
decoding must be exact — **all 7 comparable turns emitted byte-identical text**,
as it must), one engine restart between the runs. `t/s` is the engine's own
`gen=` figure; `ms/tok` is the per-round wall divided by tokens emitted per
round (on) versus the per-step wall (off).

| context | MTP on t/s | off t/s | on/off | on ms/tok | off ms/tok | on/off |
|---|---|---|---|---|---|---|
| 43 | 27.84 | 22.19 | 1.25 | 35.3 | 44.6 | 0.79 |
| 164 | 23.64 | 22.03 | 1.07 | 42.3 | 45.0 | 0.94 |
| 294 | 22.60 | 21.90 | 1.03 | 43.6 | 45.1 | 0.97 |
| 8,864 | 21.41 | 19.75 | 1.08 | 46.7 | 49.8 | 0.94 |
| 17,711 | 18.10 | 16.33 | 1.11 | 53.8 | 60.2 | 0.89 |
| 26,558 | 15.23 | 14.41 | 1.06 | 63.8 | 68.2 | 0.94 |
| 35,405 | 13.91 | 12.90 | 1.08 | 70.0 | 76.2 | 0.92 |

- **Generation: MTP depth 1 wins, ~+6–11% at 9k–35k context** (and up to +25%
  when the whole context is tiny). Acceptance settles at a flat **2.03
  tokens/round (100% hits)** from ~9k up, and the draft pass is a constant
  ~11–13 ms/round; the win is the 2-row verify sharing one weight read.
- **Prefill: MTP loses ~9%** (344 vs 376 t/s at 8.9k, 184 vs 204 at 35.4k). The
  MTP path also decodes each prompt chunk into the draft context
  (`mirrorPromptChunkIntoDraft`, no logits — MTP-only work). This is *not*
  thermal drift: the same run was *faster* per generated token while slower per
  prefill token, which throttling cannot produce.
- Net for the pi agent loop (2–3k-token delta prefill + 200–500 generated tokens
  at ~35k context) the two roughly cancel: ~1.7 s lost on prefill, ~1.7 s
  gained on generation. MTP helps long generations; it hurts cold prompts.
- The MTP-off run was the only one to reach the long end: 44,252 → 11.47 t/s,
  52,961 → 10.32, and the top turns at ~53k → 9.82 / 10.41 / 9.69 t/s
  (`gpu-wait` 80–97 ms/step).

**Thermal bracket (second MTP-on run, 2026-10-18 14:56).** The first pair ran
14:06 vs 14:28, so a re-run of the identical first five turns checks how much of
the difference was drift. Both runs then sat right after a fresh app start:

| context | MTP on t/s | off t/s | on/off | MTP on prefill t/s | off prefill t/s | on/off |
|---|---|---|---|---|---|---|
| 43 | 28.36 | 22.19 | 1.28 | — | — | — |
| 164 | 24.49 | 22.03 | 1.11 | 368.1 | 364.3 | 1.01 |
| 294 | 24.25 | 21.90 | 1.11 | 259.4 | 261.0 | 0.99 |
| 8,864 | 22.64 | 19.75 | 1.15 | 363.0 | 375.8 | **0.97** |
| 17,711 | 18.99 | 16.33 | 1.16 | 283.4 | 295.5 | **0.96** |

The bracket moves *both* numbers: the first MTP-on run was ~5% down on
everything (it followed a long session), so the honest read is **generation
+11–16%, prefill −4%**. The prefill penalty is real but about half what the
first pair suggested, and the generation win is larger. Same byte-identical
text and the same flat 2.03 tokens/round.

The paired numbers supersede the earlier unpaired note in the repo
(`MTP-on 18.3/17.5` vs `off 19.9/19.7` t/s, context unrecorded), which had
suggested MTP was a loss.

### MTP depth 2: re-test (pending, 2026-10-19)

The configured draft depth is now **2** (`mtpMaxDraft`, build `3e56c47`), so the
depth-1 figures above are the baseline, not the current configuration.

The earlier depth-2 run (2026-10-18) measured ~9–13% **slower** than plain
decoding, but that run billed both GPU waits to `sample` (the draft's and the
verify's) — `MtpPhaseStats` now splits `draft-gpu` and `verify-gpu`, and a
deeper chain adds exactly that kind of work (one more chained draft decode per
round). The comparison therefore has to be redone.

Re-run the same paired protocol as depth 1:

- identical prompt sequences, `temperature: 0`, byte-identical output required
- one engine restart between configs; same model, context, KV type
- read `accepted`/`depth` from the `MTP stats:` line — depth 2 should raise
  accepted tokens/round; if acceptance stays ~2.0, the second draft is wasted
  work and the round is slower by construction
- quote context length and pair the runs (thermals swing ~30%)

Rows go here once measured.

### KV cache and prefix reuse

Reuse works, including across the periodic context recreate:

| time | prompt | restored | **new decoded** |
|---|---|---|---|
| 12:23:39 | 26,049 | 23,692 | 2,357 |
| 12:25:01 | 38,239 | 35,067 | 3,172 |
| 12:26:20 | 47,032 | 44,354 | 2,678 |
| 12:55:44 | 48,131 | 48,065 | **66** |
| 12:57:28 (after recreate at 12:55:57) | 48,450 | 48,337 | **113** |

The delta is exactly the new turn's content (a 3,307-token delta *is* a 250-line
file read). Without recreate survival, the 12:57:28 turn would have re-decoded
48,450 tokens — ~2.7 minutes at the measured 302 t/s — instead of 3.1 s.

Mechanism: Qwen3.5 is hybrid, `can_shift=false`, so `llama_memory_seq_rm` cannot
trim a suffix; reuse goes through a serialized sequence-state blob
(`savePromptSnapshot`/`restorePromptSnapshot` →
`llama_state_seq_save_file`/`load_file`), held
across the recreate by `ReuseSurvival`.

### Snapshot cost (per turn)

Every turn saves a prompt-state snapshot and the next turn loads it back:
**664 MB at 31.8k tokens → 897 MB at 48.4k tokens** (~1.2 s/turn fixed cost,
visible on small turns where 66 new tokens still took 1.45 s). It is written to
`NSTemporaryDirectory()` and only needed (a) across a recreate and (b) because
the MTP path wipes `cachedTokenIds`/both caches at the end of `generate`
(`LlamaInference.swift:2164`) instead of leaving the live KV reusable.

### 262,144 context: what it costs (derived, not yet measured)

The ceiling is no longer 131,072. This model's own GGUF window is 262,144
(llama.cpp logs `n_ctx_train = 262144` at load) and the YaRN factor is now
derived from the context being built (`KVCacheSpec.yarnFactor(forContext:)`:
×4 at 131,072, ×8 at 262,144). The model card documents exactly those two
recipes (`--rope-scale 4 … 131072`, `--rope-scale 8 … 256K`, both
`--yarn-orig-ctx 32768`). The old fixed ×4 *halved* the native window:
`llama-context.cpp` re-sets `n_ctx_train = yarn_orig_ctx / rope_freq_scale`, so
`32768 / 0.25 = 131072`, and the ×8 recipe lands it back on 262,144.

The KV figures below are **arithmetic on measured inputs** (17,408 B/token for
q8_0 measured on device; 5,346 MB KV budget after the memory fix) — nothing
here has run on the device:

| config at 262,144 | KV | fits the 5,346 MB KV budget? |
|---|---|---|
| q8_0, MTP **off** | 4,352 MB | yes — 781 MB slack |
| q8_0, MTP **on** | 8,704 MB (two caches) | **no** — clamped to 160,768 |
| f16, MTP off | 8,704 MB | no |

So the full window is reachable only with MTP off; with MTP on the (now
cache-count-aware) clamp reduces it to ~160,768 and reports it as a memory cap.
The dominant unmeasured risk is the snapshot: on the measured slope
(664 MB at 31.8k → 897 MB at 48.4k, i.e. ~14–19 KB/token) the per-turn blob at
262,144 extrapolates to **~3.9–4.9 GB of writes per turn**, on a process whose
commitments (model 2.78 GB + KV 4.35 GB + compute ~0.32 GB ≈ 7.45 GB) already
sit under a ~9.26 GB limit. That is the argument for doing the snapshot
round-trip removal (open question 3) before the long end is used in anger.

## Stability: a GPU reset invalidates a run

Both A/B runs hit `kIOGPUCommandBufferCallbackErrorInnocentVictim` (the GPU was
reset by something outside the process). `DecodeRetry` retried once; one run
recovered, the other did not — so the MTP-on run stopped at 44k while the off
run completed to 53k. The fault is environmental, not config-specific. Memory
was tight throughout: rss 5.3–5.6 GB against ~2.8–3.1 GB available on an 11.7 GB
device, climbing across the run.

Recognise it in a log by `llama_decode failed — recreating context to recover`
followed by `ggml_metal_graph_compute: backend is in error state … recreate the
backend to recover` and `llama_decode: failed to decode, ret = -3`. Two
consequences: a run containing one is not a measurement, and after a
*persistent* fault the engine answers 500 to every request until the app is
restarted — the recovery rebuilds the `llama_context` while llama.cpp asks for
the **backend** (`llama_backend_free`/`llama_backend_init` are exported by the
vendored framework but unused).

Note also that a failed non-streaming generation logs nothing in
`llamaserver.log` (only `generation OK` prints on success), so the request line
just stops; the evidence lives in the inference file's error lines.

## Claims the logs disprove

These were stated by an agent that read the code but did not measure. Each is
contradicted by the device's own logs — treat this section as the corrective
record.

| Claim | Reality |
|---|---|
| "Hybrid models **cannot reuse KV** — `llama_memory_clear` wipes the cache, forcing full prompt re-decode" | False. Reuse is the normal path: `KV snapshot restored (48337 tokens, prompt 48450); decoding 113 new`. What's unavailable is the *cheap* `seq_rm` suffix trim; the snapshot replaces it. `llama_memory_clear(_, false)` runs on the fallback/recreate paths, not per request. |
| "Token limit **wastes 1 decode** per step — the last token is decoded but never sampled" | False, backwards. `LlamaInference.swift:2109` breaks *before* the useless decode (`if generated >= limit { break }`). |
| "MTP is **slower** (2 decode cycles per output token)" | False, measured. A paired A/B (see above) puts depth 1 at **+6–11% generation** at 9k–35k context with byte-identical output; the draft pass is ~11 ms/round against a flat 2.03 tokens/round. |
| "**Cannot be improved** without changing the serial decode+sample model" | False. The measured bottlenecks — KV bytes per token, snapshot round-trip, delta re-decode — are all addressable without touching the decode loop. |
| "Serial queue / async GPU + CPU sample per token" are the bottleneck | True as facts, but not the bottleneck here: CPU time is ~3 ms/round vs 95–150 ms of GPU verify wait. |

## Open questions / next measurements

1. ~~**MTP on/off at long context**~~ — **answered** 2026-10-18: MTP depth 1 is
   +6–11% on generation at 9k–35k context but ~9% slower on prefill (see the A/B
   above). The 44k–53k end is still one-sided (the MTP-on run died of a GPU reset
   there); a bracket run with MTP back on would close it, as would a thermal
   bracket to confirm the ~9% prefill cost is structural.
   **Re-opened 2026-10-19 at depth 2** — re-measure with the depth-2 section
   above.
2. **KV q4_0** — this model qualifies (`head_dim 256 % 32 == 0`, flash-attn
   forced). Halves the KV (843 MB → ~466 MB at 48k) and the snapshot; measure
   the t/s gain and the quality cost.
3. **Snapshot round-trip removal** — keep the live KV when the cached state is
   still a strict prefix (needs the MTP path to stop wiping `cachedTokenIds`),
   or save only when `RecreateTracker` says a recreate is due. Expected ~1.2 s
   and ~700 MB of writes per turn; measure before/after.
4. **Is the verify pass bandwidth-bound or compute-bound?** ~3.3 GB of traffic
   per 164 ms round ≈ 20 GB/s — far below the SoC's peak, so attention compute
   or kernel overhead is likely involved. Needs a profiler.
5. **Does 262,144 actually work and is it coherent?** Load it on device
   (MTP off, q8_0): the load line should report `YaRN long-context on (x8.0,
   orig 32768)` and a 262,144-token context, and `n_ctx_train adjusted to 262144`
   should appear in the llama.cpp log. Then measure prefill/gen at the top of the
   range, the snapshot size and time, and — most importantly — compare output
   against the ×4/131,072 config on the same long prompt. The ×8 recipe is
   documented by the model author, but this fine-tune's *behaviour* at 256K is
   unobserved, and the snapshot extrapolation above predicts the first
   operational problem.

## Log-line semantics

- `KV snapshot restored (R tokens, prompt P); decoding P-R new` — prefix reuse applied.
- `generated N tokens (prefill=<s>s <t/s>t/s (X of Y, Z reused), gen=…, …t/min, finish=…, cache now N) preview='…'`
  - `prefill` seconds = `[genStart, prefillEnd]`: the prompt decode (including the MTP draft-context mirror) **plus the snapshot save**, excluding the snapshot restore.
  - `prefill` t/s = **decoded** tokens (`X`) / seconds. Before 2026-10-18 this divided the full prompt length and printed impossible rates (e.g. `33116.58t/s`); any log captured before that build must be re-derived by hand.
  - `gen` = first sampled token → generation end, so it excludes the prompt tail.
- `MTP stats: … ; draft <ms>/rd + verify <ms>/rd + sample <ms>/rd; draft-gpu … + verify-gpu … + emit … + host … = wall …` — per-round phase averages. Keep the part before `;` unchanged (parsers match it).
- `plain stats: emitted=N; gpu-wait … + sample … + host … = wall …` — the non-MTP path.
