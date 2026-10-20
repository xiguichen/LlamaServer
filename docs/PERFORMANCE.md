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

Baseline rig for the tables below unless a section says otherwise (2026-10-18):

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
- Depth 1 emits 1.5–2.0 tokens/round; acceptance (`hits`) tracked 36–100% and
  fell as context grew.

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

### MTP depth 2: re-test (2026-10-20)

The draft depth was raised to **2** (`mtpMaxDraft`, build `3e56c47`) and then
measured against depth 1 on the same harness — result below. The depth-1
figures earlier in this file are the baseline, not the raised configuration.

The earlier depth-2 run (2026-10-18) measured ~9–13% **slower** than plain
decoding, but that run billed both GPU waits to `sample` (the draft's and the
verify's) — `MtpPhaseStats` now splits `draft-gpu` and `verify-gpu`, and a
deeper chain adds exactly that kind of work (one more chained draft decode per
round). It also compared against a different build and day, so the 2026-10-20
paired run below supersedes it.

Protocol: identical prompts, `temperature: 0`, one install per config, context
quoted. The reused-prefix repeat below shows why a single run cannot settle a
±5% question.

**Measured 2026-10-20, paired** (builds `e192d3e` = depth 2 and `7980625` =
depth 1 — the same code, only the constant differs; ctx 160,512, KV q8_0,
`temperature: 0`, 256-token generations, a distinct header per size so each
case prefills from scratch, same case order on both builds):

| prompt | d1 t/s | d2 t/s | Δ | d1 emit/rd | d2 emit/rd | d1 draft-gpu | d2 draft-gpu | d1 verify-gpu | d2 verify-gpu | d1 wall/rd | d2 wall/rd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 26.16 | 26.59 | +1.6% | 1.69 | 2.04 | 9.7 ms | 19.2 ms | 52.1 ms | 55.6 ms | 64.8 ms | 77.0 ms |
| 6,883 | 20.89 | 21.90 | +4.8% | 1.73 | 2.16 | 10.9 ms | 21.2 ms | 69.8 ms | 75.8 ms | 83.3 ms | 99.0 ms |
| 13,957 | 16.55 | 17.16 | +3.7% | 1.74 | 2.12 | 12.2 ms | 22.2 ms | 90.4 ms | 98.5 ms | 105.2 ms | 123.2 ms |
| 21,028 | 14.80 | 14.43 | −2.5% | 1.73 | 2.08 | 13.5 ms | 24.4 ms | 101.9 ms | 117.1 ms | 117.7 ms | 144.2 ms |

Depth 1 accepts 0.69–0.74 drafts/round (its single draft lands in ~70% of
rounds); depth 2 accepts 1.04–1.16, so the **second link lands in ~35–43% of
rounds**. That buys ~+20% emitted tokens per round, paid for by ~+19% wall per
round: +10 ms of `draft-gpu` (the chained draft, roughly constant) and
+3.5–15 ms of `verify-gpu` (the third verify row, growing with the KV the
verify attends over).

So depth 2 is ahead at three of the four sizes and only −2.5% at 21k, where the
wider verify has grown to +15 ms/round. Those gaps are near single-run thermal
noise: a repeat of the 21k case with a reused prefix (21,021 of 21,028 tokens
reused) measured `verify-gpu` 74–84 ms/round instead of 102 ms — up to ~20% from
clock state alone.

**Verdict: keep depth 2.** It emits ~20% more tokens per round for ~19% more
wall, which is a net +1.6–4.8% generation at 48–14k prompt tokens, and a −2.5%
wash at 21k (inside the run-to-run spread above, so not evidence that depth 2
loses there — just that its advantage fades as the third verify row grows). A
workload dominated by ≥20k contexts sees no benefit; below that depth 2 is the
better setting.

Prefill is unaffected by depth (296–400 t/s at 6.9k–21k) — the prompt is
mirrored into the draft context once either way.

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
the MTP path wipes `cachedTokenIds`/both caches at the end of `generate` (the
`usedMtp` branch) instead of leaving the live KV reusable.

### 262,144 context: what it costs

The model's GGUF window is 262,144, and the YaRN factor is derived from the
context being built (×4 at 131,072, ×8 at 262,144) — see `docs/MEMORY.md`.

Measured on device:

| config | result |
|---|---|
| q8_0, MTP on (build `e192d3e`) | clamped to **160,512** by the memory budget |
| q4_0, MTP off (build `f6793a4`) | **loads at 262,144**, YaRN off |

KV at the full window: **4,352 MiB** for q8_0 (17,408 B/token) and **2,304 MiB**
for q4_0 (9,216 B/token charged). Coherence at 256K is unmeasured (open
question 5), and the per-turn snapshot extrapolates to ~4 GB of writes at the
full window (open question 3) — measure both before using the long end in
anger.

## Stability: a GPU reset invalidates a run

Both A/B runs hit `kIOGPUCommandBufferCallbackErrorInnocentVictim` (the GPU was
reset by something outside the process). `DecodeRetry` retried once; one run
recovered, the other did not — so the MTP-on run stopped at 44k while the off
run completed to 53k. The fault is environmental, not config-specific. Memory
was tight throughout: rss 5.3–5.6 GB against the 6 GiB per-process ceiling
(`docs/MEMORY.md`), climbing across the run.

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
| "Token limit **wastes 1 decode** per step — the last token is decoded but never sampled" | False, backwards. The generation loop breaks *before* the useless decode (`if generated >= limit { break }`). |
| "MTP is **slower** (2 decode cycles per output token)" | False, measured. A paired A/B (see above) puts depth 1 at **+6–11% generation** at 9k–35k context with byte-identical output; the draft pass is ~11 ms/round against a flat 2.03 tokens/round. |
| "**Cannot be improved** without changing the serial decode+sample model" | False. The measured bottlenecks — KV bytes per token, snapshot round-trip, delta re-decode — are all addressable without touching the decode loop. |
| "Serial queue / async GPU + CPU sample per token" are the bottleneck | True as facts, but not the bottleneck here: CPU time is ~3 ms/round vs 95–150 ms of GPU verify wait. |

## Open questions / next measurements

1. ~~**MTP on/off at long context**~~ — **answered** 2026-10-18 (depth 1:
   +6–11% generation at 9k–35k, ~9% slower prefill) and **2026-10-20** (depth 2:
   +1.6–4.8% at 48–14k prompt tokens, −2.5% at 21k). The 44k–53k end is still
   one-sided: the depth-1 MTP-on run died of a GPU reset there.
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
5. **Is 262,144 coherent?** Loading it is answered: q4_0 reaches 262,144
   (measured). Still open: compare output against the ×4/131,072 config on the
   same long prompt, and measure snapshot size/time at the top of the range.

## Log-line semantics

- `KV snapshot restored (R tokens, prompt P); decoding P-R new` — prefix reuse applied.
- `generated N tokens (prefill=<s>s <t/s>t/s (X of Y, Z reused), gen=…, …t/min, finish=…, cache now N) preview='…'`
  - `prefill` seconds = `[genStart, prefillEnd]`: the prompt decode (including the MTP draft-context mirror) **plus the snapshot save**, excluding the snapshot restore.
  - `prefill` t/s = **decoded** tokens (`X`) / seconds. Before 2026-10-18 this divided the full prompt length and printed impossible rates (e.g. `33116.58t/s`); any log captured before that build must be re-derived by hand.
  - `gen` = first sampled token → generation end, so it excludes the prompt tail.
- `MTP stats: … ; draft <ms>/rd + verify <ms>/rd + sample <ms>/rd; draft-gpu … + verify-gpu … + emit … + host … = wall …` — per-round phase averages. Keep the part before `;` unchanged (parsers match it).
- `plain stats: emitted=N; gpu-wait … + sample … + host … = wall …` — the non-MTP path.
