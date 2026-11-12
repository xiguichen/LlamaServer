# MTP target-KV reuse (design + test plan)

Status: **target preservation implemented 2026-11-12; paired on-device A/B still
required before quoting a gain.** The pure bookkeeping is `Sources/App/MtpKVPlan.swift`
(tests in `Tests/LlamaServerTests/MtpKVPlanTests.swift`). The per-turn
stable-prefix snapshot is **kept** as the fallback: a client that replays
transformed assistant text (reasoning stripped, tool envelopes re-rendered) may
not match the raw generated tokens, and the snapshot is what preserves reuse
across that divergence. Numbers live in `docs/PERFORMANCE.md`; this file only
defines the invariant and the tests.

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
guard posMin == 0, posMax + 1 == residentTokenCount else { clear; return }
```

If a model/cache type reports bounds differently (recurrent/hybrid), the guard
fails and the old clear path runs — correctness is preserved, only the
optimisation is lost. Log the mismatch at `.warn` so a device run shows which
cache types cannot use it.

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
`draftPromptTokens` proves it holds the reused prefix. Preserving the target KV
makes `reuseLen` extend into the generated assistant text. Two cases:

1. The draft holds the same rows (it was rolled back with the target). It can be
   trimmed and reused, but `draftPromptBoundaryH` only has hidden states at
   target prompt-chunk boundaries. A `reuseLen` inside the generated text has no
   boundary `h`, so the first new mirrored row would pair with a zero/stale
   hidden state. That degrades draft acceptance, never output.
2. The draft does not hold the prefix (`draftPromptTokens` is only the previous
   prompt). Current code clears it but still mirrors only `[reuseLen, total)`,
   which leaves a position gap. MTP then degrades to no-draft rounds for that
   generation.

Neither case corrupts output (every emitted token is sampled from the target),
but both can erase the generation-speed benefit of MTP on the turn after a
reused turn. This is exactly what the paired A/B must measure. A follow-up can
feed the target's `h` at the new reuse boundary into `draftPromptBoundaryH` or
re-mirror the draft from 0 when it was cleared; do not make that change in the
same step.

## Test plan

Status: the pure tests below are implemented in
`Tests/LlamaServerTests/MtpKVPlanTests.swift`; the on-device A/B is still open.

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

### On-device A/B (required before enabling)

Use the paired protocol in `docs/PERFORMANCE.md`:

1. Same model (4B Q4_K_M), MTP on, same prompt shapes, alternate
   baseline (wipe) vs preserve, at least 3 pairs.
2. Record per turn: `prefill=` `(X of Y, Z reused)`, `gen=`, total wall, and the
   `MTP stats` `no-draft`/`depth` fields.
3. Acceptance:
   - reused tokens on turn N+1 must be at least the shared prefix length
     (system + history), not just the stable snapshot prefix;
   - no output regressions on a fixed tool-call + long-context prompt set;
   - generation t/s must not drop by more than noise on turns where the draft
     lost its boundary `h` (`no-draft` rate is the diagnostic);
   - no `KV snapshot` write/load on turns where the live cache was reused.
4. If the draft regression dominates, land the target-KV preservation and the
   draft boundary `h` follow-up together, or keep MTP-off for long turns.

The code is in place; only after those numbers exist should
`docs/PERFORMANCE.md` get a measured row. This document's status already records
the implementation and explicitly leaves the A/B open.
