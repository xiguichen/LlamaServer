# MTP target-KV reuse (design + test plan)

Status: **target preservation implemented 2026-11-12; hybrid full-attention
bounds fix implemented and confirmed on device 2026-11-14; paired A/B measured
2026-11-14 (see `docs/PERFORMANCE.md`, "Snapshot cost").** The pure bookkeeping
is `Sources/App/MtpKVPlan.swift` (tests in
`Tests/LlamaServerTests/MtpKVPlanTests.swift`). The stable-prefix snapshot is
**kept** as the fallback: a client that replays transformed assistant text
(reasoning stripped, tool envelopes re-rendered) may not match the raw generated
tokens, and the snapshot is what preserves reuse across that divergence. It is
written only when a recreate is due or the live cache was not reusable
(`ReuseSurvival.hybridSnapshotSaveNeeded`). Note that a chat replay *always*
takes the save path: the stable-prefix snapshot ends before the assistant primer,
and the next prompt replaces that primer, so the live cache is never a whole
prefix. The skip therefore helps append-only clients (`/v1/completions`) only;
the per-turn chat save is still the cheaper policy (measured). Numbers live in
`docs/PERFORMANCE.md`; this file only defines the invariant and the tests.

## Problem

The MTP path used to end `generateOnce` by clearing the target KV and resetting
`cachedTokenIds`:

```swift
if let mem = llama_get_memory(context) { llama_memory_clear(mem, false) }
if draftSharesTargetMemory, let dmem = llama_get_memory(dft) {
    llama_memory_clear(dmem, false)
}
decodedGenerated = 0
```

The clear existed because the speculative loop manages positions explicitly and
the cross-call reuse bookkeeping was not reconstructed after it. The cost was
large: on a dense model with MTP on, the next turn could not use the shared
system/history prefix at all and re-prefilled the whole prompt. Hybrid models
still recovered the *stable* prefix from the per-turn snapshot, but paid a
multi-hundred-MB write + load every turn (`docs/PERFORMANCE.md`, “Snapshot
cost”). The loop now reconstructs the resident prefix (`MtpKVPlan`) and keeps the
target cache when the live bounds guard agrees; the clear remains the fallback.

## What is actually resident after a round

Two contexts write the same rows:

- target: `decodeVerify([idLast] + drafts, nextPos)` (or
  `decodeSingleTarget(idLast, nextPos)` on a no-draft fallback);
- draft: `decodeDraft` writes `idLast` and (sometimes) the first draft.

`reals[k]` is sampled from row `k`; it is a *prediction*, not a decoded row.
Rows are `idLast, drafts[0], drafts[1], …`; after the round's
`rollbackPos = nextPos + emitCount` trim, the resident sequence is:

```
prompt … | idLast | reals[0] … reals[emitCount-2]
```

The **last emitted token** (`reals[emitCount-1]`, or the “bonus” real on a full
accept) is not resident: it becomes the next round's `idLast` and would be
decoded at the new `nextPos`.

So, for a round that ends normally:

```
resident tokens = promptTokens.count + generated - 1
resident ids    = promptTokens + generatedTokenIds.dropLast()
```

Two exceptions matter:

1. **No generation at all** (`idLast == -1` or `eog_immediate`): the prompt rows
   are resident, `generated == 0`, and the rule is
   `resident = promptTokens.count` (not `prompt - 1`).
2. **The final fallback round sampled eog / sampler_error**: the fallback
   decodes `idLast` at `nextPos` before sampling its successor, then breaks
   without emitting. The last emitted token *is* resident:
   `resident = promptTokens.count + generated`.

A verify round that samples eog is **not** an exception: the eog row is trimmed
and the last emitted real is not resident, so the normal rule holds.

## Trim rule for a round that stops mid-emit

`step.rollbackPos` is computed from `step.emitCount` (the decision), but a stop
string, the token limit, or a client cancel can stop the emit loop earlier.
`actualEmitted` may therefore be smaller. Before preserving, trim both contexts
to:

```
nextPos + actualEmitted
```

even when `rollbackPos == nil` (a full accept leaves every decoded row
resident, but some of those rows were never emitted). The current comment at
the emit loop already records this requirement.

## Proposed bookkeeping (implemented as `MtpKVPlan`)

Track `residentTokenCount` explicitly rather than deriving it from `nextPos`,
because `nextPos` is not advanced on the eog path:

- start: `residentTokenCount = promptTokens.count`
- after a verify round: `residentTokenCount = nextPos + actualEmitted`
  (trim to that position first)
- fallback, successor emitted: `residentTokenCount = nextPos_after += 1`
- fallback, successor eog/sampler error: `residentTokenCount = nextPos + 1`

At the end of generation, if the loop did not abort on a sampler error or a
decoder failure:

```swift
let residentIds = promptTokens + generatedTokenIds.prefix(residentTokenCount - promptTokens.count)
cachedTokenCount = residentTokenCount
cachedTokenIds = residentIds
```

and do **not** clear the target. Keep the draft clear only when the target is
cleared (abnormal exit).

### Runtime guard (required)

Never trust the computed count blindly. Before publishing it, compare with the
live sequence bounds:

```swift
let posMin = Int(llama_memory_seq_pos_min(memory, 0))
let posMax = Int(llama_memory_seq_pos_max(memory, 0))
guard MtpKVPlan.boundsMatch(posMin: posMin, posMax: posMax,
                            resident: residentTokenCount,
                            hybridFullAttention: isHybrid && nSwa == 0) else {
    clear; return
}
```

For a dense cache `boundsMatch` still requires `posMin == 0` and
`posMax + 1 == resident`. A hybrid cache reports
`posMin = max(attention_min, recurrent_min)` and
`posMax = min(attention_max, recurrent_max)`; its recurrent component only
keeps the tail, so `posMin > 0` is normal even when every attention row from 0
is resident. When the model's attention layers are full (`n_swa == 0`, logged
as `swa_window 0`), the attention component cannot evict the front within the
context, so the guard checks only `posMax + 1 == resident`. A hybrid model with
a sliding window keeps the strict check and falls back to the old clear path —
correctness is preserved, only the optimisation is lost. Log any mismatch at
`.warn` so a device run shows which cache types cannot use it.

The same guard also protects **dense reuse in general**: when computing `lcp`
from `cachedTokenIds`, the code clamps it to `MtpKVPlan.liveResidentCount`
(`posMax + 1`, or 0 when `posMin > 0`) and always runs `seq_rm(lcp)` — even when
`lcp == cachedTokenIds.count` — to drop any rows beyond the tracked prefix.
Dense reuse used to assume the tracking was exact; MTP preservation makes that
assumption load-bearing, so the clamp is part of this change.

### Abnormal exits

Do not publish reuse after:

- `sampler_error` (the verify/fallback decode ran but no decision was committed;
  rows may extend past the emitted set);
- `decode` failure (`recoverFromDecodeFailure` recreates the context; the
  generation defer already invalidates tracking);
- any future early `return` that skips the bookkeeping.

## Draft-context consequences (why this is not free)

The draft is kept across generations and trimmed at the next prompt start when
`draftResidentTokens` proves it holds the reused prefix. Preserving the target KV
makes `reuseLen` extend into the generated assistant text, so the tracked ids are
the draft's **resident** rows (prompt + the generated rows it kept), not just the
last prompt. When the draft's rows match, it is trimmed to `reuseLen` and only
the rest is mirrored — including a `reuseLen` inside generated text. Two residual
cases remain:

1. The draft holds the same rows, but `draftPromptBoundaryH` only has hidden
   states at target prompt-chunk boundaries. A `reuseLen` inside the generated
   text has no boundary `h`, so the first newly mirrored row pairs with a zero
   hidden state. That degrades draft acceptance for one row, never output.
2. The draft does not cover the prefix (it holds fewer rows, or its live bounds
   disagree — e.g. a failed draft decode left a hole, or the previous generation
   took the plain path). `MtpKVPlan.draftReuseLen` returns nil, the draft is
   cleared, and mirroring only `[reuseLen, total)` leaves a position gap. MTP
   then degrades to no-draft rounds until a generation with `reuseLen == 0`
   rebuilds the draft from position 0.

Neither case corrupts output (every emitted token is sampled from the target),
and the common case — the previous generation used the speculative path and
preserved its target KV — now keeps drafting on the next turn. The paired
on-device A/B is complete: append-only prompts reuse the live cache without a
snapshot round-trip; a chat replay still restores the stable-prefix snapshot
every turn, and that per-turn save is cheaper than letting the tail grow
(`docs/PERFORMANCE.md`, question 3).

## Test plan

Status: the pure tests below are implemented in
`Tests/LlamaServerTests/MtpKVPlanTests.swift`; the on-device A/B ran
2026-11-14.

### Pure tests (CI, no model)

Add `Sources/App/MtpKVPlan.swift` (pure, `Int32` tokens, no `llama` import) and
`Tests/LlamaServerTests/MtpKVPlanTests.swift`:

- `testNormalRoundLeavesLastEmittedTokenUndecoded` — `generated > 0` →
  `resident == prompt + generated - 1`, ids end at `generatedTokenIds.dropLast()`.
- `testIdleGenerationKeepsWholePrompt` — `generated == 0` →
  `resident == promptTokens.count`.
- `testFinalFallbackEogKeepsLastEmittedToken` — fallback eog → `resident ==
  prompt + generated`.
- `testVerifyEogDropsLastEmittedToken` — verify eog → normal rule.
- `testStoppedRoundTrimsToActualEmitted` — `actualEmitted < step.emitCount` →
  trim position `nextPos + actualEmitted`.
- `testSamplerErrorIsNotPublishable` — abnormal flag → no reuse.
- `testResidentIdsArePrefixOfGeneratedIds` — never record more ids than were
  emitted, and never fewer than the resident count.

Existing `MtpSpeculatorTests` already pin `emitCount`, `rollbackPos`,
`draftSeedIndices`, and `carryHFromRow`; add a property-style test that the
plan's resident count matches the hard-coded rules for random
`(accepted, stopRow, fallback)` combinations.

### CI integration tests

- All existing tests must pass unchanged.
- The bounds guard must be exercised with a fake `posMin/posMax` (pure helper),
  not a live context.

### On-device A/B (completed 2026-11-14)

Use the paired protocol in `docs/PERFORMANCE.md`:

1. Same model (4B Q4_K_M), MTP on, same prompt shapes, alternate
   baseline (wipe) vs preserve, at least 3 pairs.
2. Record per turn: `prefill=` `(X of Y, Z reused)`, `gen=`, total wall, and the
   `MTP stats` `no-draft`/`depth` fields.
3. Acceptance:
   - for an **append-only** prompt, reused tokens on turn N+1 equal the previous
     whole prompt and no `KV snapshot` write/load runs;
   - for a **chat replay**, reuse equals the stable-prefix snapshot length, not
     system + history: the primer is replaced, so expecting more is wrong (the
     live cache diverges there by design);
   - no output regressions on a fixed tool-call + long-context prompt set;
   - generation t/s must not drop by more than noise on turns where the draft
     lost its boundary `h` (`no-draft` rate is the diagnostic).
4. If the draft regression dominates, land the target-KV preservation and the
   draft boundary `h` follow-up together, or keep MTP-off for long turns.

The code is in place; only after those numbers exist should
`docs/PERFORMANCE.md` get a measured row. This document's status already records
the implementation and explicitly leaves the A/B open.
