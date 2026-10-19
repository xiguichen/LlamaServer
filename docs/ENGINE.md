# Engine internals (`LlamaInference.swift`)

Read this when touching model loading, MTP speculative decoding, sampling, the
generation loop, or context lifecycle. Loaded on demand — do not repeat this
detail in `AGENTS.md`.

## Context lifecycle

- **Recreate every 10 generations** (`contextRecreateThreshold`) to flush Metal
  GPU/allocator state that can cause hangs.
- The recreate runs **off the response path** (`RecreateTracker`, called after
  the finish chunk in streaming), and the prefix-reuse snapshot survives it
  (`ReuseSurvival`) — a dense model's `cachedTokenIds` and the on-disk sequence
  blob are preserved, so the next turn restores from disk instead of re-decoding
  the whole prompt.
- `recreateContext()` frees the old context **before** creating the new one and
  sets `contextValid = false` immediately after the free, so a subsequent
  failure cannot use the dangling pointer.
- Prefix reuse: dense caches trim with `seq_rm`. Hybrid/SSM caches have
  `can_shift = false`, so `llama_memory_seq_rm` can never drop a suffix; instead
  `generate` restores the previous prompt's serialized sequence state
  (`restorePromptSnapshot` → `llama_state_seq_load_file`, saved by
  `savePromptSnapshot` → `llama_state_seq_save_file` into `NSTemporaryDirectory()`)
  and decodes only the suffix. This is real reuse:
  `KV snapshot restored (48337 tokens, prompt 48450); decoding 113 new`.
- The snapshot is not free: ~664 MB at 32k tokens → ~897 MB at 48k
  (**~1.2 s and ~700 MB of writes per turn**, measured). The next turn re-decodes
  the previous turn's generation (the snapshot is the *stable* prefix, taken
  before generation).
- The per-turn save is only needed because the MTP path ends by wiping
  `cachedTokenIds` and both caches (the `usedMtp` branch at the end of
  `generate`) instead of leaving the live KV reusable. Removing that wipe is the
  open optimisation in `docs/PERFORMANCE.md`.

## MTP (Multi-Token Prediction)

### Output slots

`n_outputs_max` is resolved by `MtpSpec.resolveOutputSlots`: the larger of the
model's GGUF-derived head count (`nextn_predict_layers`, or fallback
`n_predict_layers`, `+1` for the main head, fallback 3 when absent, overridable
by an explicit count) and the verify batch `maxDraft + 1`.

- **Never sample from more slots than the model supports** — it `ggml_abort`s in
  `llama_sampler_sample`. The runtime guard `llama_get_logits_ith(ctx, i) != nil`
  prevents this (the plain path probes via `MtpProbe`).
- **A non-negative index is a BATCH token index, not an output slot**: a row with
  `logits = 0` returns NULL and `llama_sampler_sample` aborts. Read the draft
  context's logits with index `-1` (last output row), never a batch index whose
  row has no logits.
- The target's unmasked nextn hidden states are indexed by token offset within
  the last `llama_decode` batch, so after the prompt the last token's `h` is row
  `lastChunkCount − 1`.

### Depth

`mtpMaxDraft = 2` (hard-coded, no UI toggle; raised from 1 on 2026-10-19 for a
fresh A/B). The round machinery is depth-N capable (chained drafts, N-row verify,
draft seed rows). The earlier depth-2 run (2026-10-18) measured **~9–13%
slower** than plain decoding, but its "sample" phase billed both GPU waits (the
phases were split afterwards), so that number is being re-measured with the
current instrumentation — method in `docs/PERFORMANCE.md`.

The depth-1 result (the previous configuration) is a measured net win on
*generation*, not on prefill — paired A/B, same prompts, temperature 0,
byte-identical output, 2026-10-18:

- first pair: **+6–11% generation** at 9k–35k context, **~9% slower prefill**
- thermal-bracket re-run: **+11–16% / −4%** (the first pair was taken straight off
  a long session)

The prefill cost is that the MTP path also mirrors each prompt chunk into the
draft context; the win is the 2-row verify sharing one weight read (acceptance
flat at 2.03 tokens/round). For a pi-style turn (small delta prefill + a few
hundred generated tokens at ~35k) the two roughly cancel. See
`docs/PERFORMANCE.md`.

### One round

1. Draft up to `mtpMaxDraft` tokens with the `draft` context
   (`LLAMA_CONTEXT_TYPE_MTP`). The draft token is a greedy `MtpSpec.argmax`
   (vDSP) over `llama_get_logits_ith(dft, -1)`.
2. The confidence gate (`MtpSpec.draftConfidence >= MtpSpec.pMin`) is **off**:
   `pMin` is 0.0 (`MtpSpec.gateEnabled` false), so a computed draft is always
   verified.
3. The target verifies `[idLast] + drafts` in one batch. **Every emitted token is
   sampled from the target**, so a bad draft costs a round but never corrupts
   output.
4. Verify rows are sampled lazily: sampling stops at the first real that misses
   its draft or is eog, because later rows are never emitted.
5. `MtpSpeculator.decide` is the pure acceptance rule (`accepted`, `emitCount`,
   `rollbackPos`, `carryHFromRow = emitCount − 1`).
6. On a full accept the draft context has a one-position gap (the last accepted
   token was never drafted into it). `MtpSpeculator.draftSeedIndices` lists the
   missing rows; they are decoded as **leading seed rows of the next round's
   draft decode** (one weight pass), not as a separate catch-up decode. Partial
   accepts stay contiguous after the two-context `seq_rm` rollback.
7. `llama_sampler_sample` accepts the token it returns, so the driver never calls
   `llama_sampler_accept` itself.

### The draft context must see the prompt

llama.cpp drops `ctx_other` for non-Gemma4 archs, so the draft has its own
memory. `mirrorPromptChunkIntoDraft` decodes every target prompt chunk into it
(row `k` = token `x_k` with target `h_{k−1}`, logits off), like upstream
draft-mtp `process()`. It is skipped when the draft shares the target's memory
(`is_mem_shared`).

- The draft KV is kept across generations and trimmed to the target's reused
  prefix when `draftPromptTokens` shows it holds the same tokens; otherwise it
  starts empty.
- A failed mirror leaves the draft **empty**, never with a position gap, because
  a gap makes every later draft decode fail.

### Timings and the stats line

Metal decode is async and the first logits read waits for it, so the round calls
`llama_synchronize` on the draft right after `decodeDraft` and on the target
right after `decodeVerify` (free — the read would wait anyway):

- `draft` / `verify` = CPU enqueue only
- `draft-gpu` / `verify-gpu` = the GPU waits
- `sample` = CPU sampling only
- `host` = the rest of the wall clock

The `MTP stats:` line reports `accepted`/`depth` (avg accepted drafts per round)
plus per-phase ms per round (`MtpPhaseStats`):
`draft + verify + sample (N rounds); draft-gpu + verify-gpu + emit + host = wall`.
**Keep the part before `;` unchanged** (log parsers match it); new fields go
after it. Logs from before this split bill both GPU waits to `sample`.

The plain path logs
`plain stats: emitted=N; gpu-wait + sample + host = wall (N steps)`
(`PlainStepStats`), syncing the previous decode at the top of each step.

## Generation loop

### Gen clock

`generate` calls `llama_synchronize` on the target (and on the draft when the
prompt was mirrored) right before `prefillEnd`, and the gen clock
(`firstTokenTime`) starts there for **both** paths. Do not move it back into a
path: the prompt's GPU tail would then land in plain `gen` but not MTP `gen`, and
the MTP/plain ratio stops being comparable.

The `generated N tokens` line prints seconds and t/s with 2 decimals, and its
`prefill` segment comes from `PrefillStats.segment` — the rate is over the
**decoded** tokens (`X of Y, Z reused`), never the full prompt length. Before
2026-10-18 it divided the full prompt by the delta decode time and printed rates
like `33116.58t/s` after cache reuse; re-derive such numbers by hand.

### Greedy fast path

When the chain is pure greedy (`MtpSpec.isPureGreedy`: temperature <= 0 and no
penalty sampler; the chain and the check share one `MtpSpec.hasPenalties`
result), `sampleTarget` takes `MtpSpec.argmax` over `llama_get_logits_ith`
instead of `llama_sampler_sample`. That gives the same token (first max on a tie;
greedy `accept` is a no-op) and applies to the plain step and the MTP
first/fallback/verify rows.

On the greedy path only, a NULL logits row yields -1 (`sampler_error`) instead of
a sample; the sampler path keeps llama.cpp's behaviour. Any stateful sampler
stage (penalties) keeps the sampler path.

### Token limit

The plain loop breaks **before** decoding the token that reaches the limit
(nobody reads its logits), so `decodedGenerated` lags by that token and
`cachedTokenIds` excludes it — the same rule as a stop-string break.

### KV clearing

Use `llama_memory_clear(mem, false)` everywhere. `true` also memsets the buffers
(~39 ms per call for a 2 GiB q8_0 KV on device) and buys nothing: cleared KV
cells are masked and fresh recurrent cells are zeroed in-graph (`rs_z`), so stale
bytes are never read. Decode-failure recovery recreates the context and does not
rely on a clear.

## Failure recovery

### Decode failure (automatic)

A `llama_decode` failure (`ret = -3`) comes from the GPU discarding our Metal
command buffers (`kIOGPUCommandBufferCallbackErrorInnocentVictim` — a GPU reset
caused outside this process; llama.cpp then says "recreate the backend to
recover"). `recoverFromDecodeFailure()` rebuilds the context and dumps the tail
of llama.cpp's own log (`NativeLogRing`, last 16 lines) at error level, because
those native lines are routed at `.debug` and are otherwise absent from a
default-level device log — without the dump a device log shows only the return
code.

`generate` is a thin retry wrapper around `generateOnce`: `DecodeRetry` allows
one retry only when the failure is `.decode` and nothing was delivered yet (a
delivered token cannot be unsent). A prompt larger than the context throws
`.contextTooFull` (400), never `.decode`.

### Unloading

`unload()` must be called before discarding `LlamaInference` — otherwise ARC
runs `deinit` and the (multi-GB) Metal/CPU allocation stays resident, causing the
next `llama_model_load_from_file` to return NULL (OOM).

### Vendored llama.cpp

`LLAMA_TAG` in `scripts/fetch-llama-xcframework.sh` must point to a llama.cpp
release that publishes an `xcframework.zip` asset. Changing it may require
adapting `Sources/App/LlamaInference.swift` for API drift.

## Startup

`StartupTimer` records named phases and durations and formats the
`load phases:` / `Startup gap:` breakdown line.
