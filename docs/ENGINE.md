# Engine internals (`LlamaInference.swift`)

Read this when touching model loading, MTP speculative decoding, sampling, the
generation loop, or context lifecycle. Loaded on demand — do not repeat this
detail in `AGENTS.md`.

## Context lifecycle

- **Recreate every 10 generations** (`contextRecreateThreshold`) to flush Metal
  GPU/allocator state that can cause hangs.
- The recreate runs **off the response path** (`RecreateTracker`, called after
  the finish chunk in streaming), and the prefix-reuse snapshot survives it
  (`ReuseSurvival`). Before freeing a dense model's context, `recreateContext()`
  serializes its live KV state to the same on-disk blob the hybrid path uses,
  then clears `cachedTokenIds`; the next turn restores from disk instead of
  re-decoding the whole prompt. Hybrid snapshots were already saved per turn.
  A recreate triggered by a mid-generation decode failure skips this capture:
  `cachedTokenIds` is only updated after a successful generation, so the live
  KV would no longer match it.
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
- The snapshot is not free (size and per-turn write cost are measured in
  `docs/PERFORMANCE.md`). The next turn re-decodes the previous turn's
  generation (the snapshot is the *stable* prefix, taken before generation).
- The MTP path **publishes its resident target prefix** after a normal
  generation (`MtpKVPlan` + the live `seq_pos_min`/`seq_pos_max` guard), so the
  next turn reuses the live KV instead of restoring the snapshot. The per-turn
  snapshot is still written and remains the fallback: clients that
  replay transformed assistant text (reasoning stripped, tool envelopes
  re-rendered) may not match the raw generated tokens, and any abnormal exit or
  guard mismatch clears both caches as before. Invariant, trim rule,
  draft-context consequences, and the required paired A/B are in
  [`docs/MTP-KV-REUSE.md`](MTP-KV-REUSE.md).

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

`mtpMaxDraft = 2` (hard-coded, no UI toggle; raised from 1 on 2026-10-19). The
round machinery is depth-N capable (chained drafts, N-row verify, draft seed
rows). A deeper chain adds a chained draft decode and another verify row, so its
marginal gain fades as context grows. The measured depth-1/depth-2 comparison
and the rationale for keeping 2 are in `docs/PERFORMANCE.md` — do not restate
the numbers here.

The MTP path also pays a prefill cost: it mirrors each prompt chunk into the
draft context (`mirrorPromptChunkIntoDraft`).

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

### Target KV after a generation

The speculative loop used to wipe the target and draft caches at the end of
every MTP generation, so the next turn could only recover a prefix from the
snapshot. It now reconstructs what is resident and keeps the target when the
bookkeeping agrees with the live sequence:

- A normal verify round leaves `nextPos + actualEmitted` rows resident. The last
  emitted token is a prediction, not a decoded row, so it is excluded; a
  fallback round that stops on eog keeps the decoded `idLast` row
  (`MtpKVPlan.committedResident` / `fallbackStopResident`).
- A stop string, token limit, or client cancel trims both contexts to
  `nextPos + actualEmitted` even when `rollbackPos` is nil, because
  `rollbackPos` only drops whole unaccepted rows.
- Before publishing, `MtpKVPlan.boundsMatch` requires `seq_pos_min == 0` and
  `seq_pos_max + 1 == resident`. A sampler error is never publishable (the
  decision was not committed). Any failure falls back to
  `llama_memory_clear(mem, false)` and `cachedTokenIds = []`.
- The draft is cleared only when the target is cleared (shared-memory case);
  with its own memory it is retained and trimmed at the next prompt start.

### The draft context must see the prompt

llama.cpp drops `ctx_other` for non-Gemma4 archs, so the draft has its own
memory. `mirrorPromptChunkIntoDraft` decodes every target prompt chunk into it
(row `k` = token `x_k` with target `h_{k−1}`, logits off), like upstream
draft-mtp `process()`. It is skipped when the draft shares the target's memory
(`is_mem_shared`).

- The draft KV is kept across generations and trimmed to the target's reused
  prefix when `draftResidentTokens` (which includes the generated rows target-KV
  preservation keeps) shows it holds the same tokens; otherwise it starts empty.
  Publishing those ids is gated on a gap-free sequence: a failed draft decode or
  rollback is not visible in `seq_pos_min`/`seq_pos_max`, so a private flag
  disqualifies it.
- A failed mirror leaves the draft **empty**, never with a position gap, because
  a gap makes every later draft decode fail.
- The draft context is built with a much smaller `n_batch`/`n_ubatch`
  (`draftBatchSize`, 64) than the target. Its compute workspace is dominated by
  the attention mask over `n_ctx` and scales with `n_ubatch`, not with the
  one-layer MTP weights, so mirroring a target-sized 512/1024-token chunk in one
  decode reserves a target-sized buffer. `MtpSpec.mirrorChunks` splits the
  chunk; draft generation itself decodes only `mtpMaxDraft` rows. Measured at
  262,144 (q4_0, build `2b6dd82`): the draft still reserved 1,079.56 MiB MTL0 +
  33.52 MiB CPU at batch 64, down from 1,304.46 + 256.78 MiB at batch 512. The
  cap buys ≈448 MiB but ≈1.1 GiB of fixed MTP-graph workspace remains, which
  is why `KVCacheSpec.contextPlan` reserves the draft compute for MTP.

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
rely on a clear. The one intentional exception is the MTP preservation path
above: a normal generation keeps the live target KV instead of clearing it.

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
