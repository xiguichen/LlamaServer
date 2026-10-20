# MTP: Fuse the Draft Catch-up into the Next Draft Decode — Implementation Plan

> **Historical (2026-10-14).** Completed. Current facts live in `docs/ENGINE.md`; do not use this plan as reference.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the one remaining full draft-model weight pass per full-acceptance MTP round by decoding the accepted tokens into the draft context as one multi-row batch instead of a separate one-row "catch-up" decode plus the next round's one-row draft decode.

**Architecture:** The MTP path runs two llama.cpp contexts: `context` (target, source of truth) and `mtpContext` (draft, `LLAMA_CONTEXT_TYPE_MTP`, `ctx_other = target`). Each round decodes one row into the draft context at `nextPos`, samples a draft token, verifies `[idLast] + draft` on the target, then (on a full accept) runs a second one-row draft decode — the catch-up — to fill the position the draft context is missing. Both draft decodes read the whole draft model (MTP block + tied Q6_K head, 568.5 MiB), so the round pays two weight passes to draft. This plan defers the catch-up row and prepends it to the next round's draft decode, so the two rows go through the model in one batch (one weight pass). The verify batch is unchanged, so every emitted token is still sampled from the target.

**Tech Stack:** Swift (iOS), llama.cpp staging C API via `llama` framework + `LlamaExtShim.cpp`, XcodeGen project, GitHub Actions CI, on-device measurement over HTTP.

**Spec:** This plan (root-cause analysis performed 2026-10-14 from the on-device `MTP stats:` log `llamaserver 3.log`, cross-checked against `Sources/App/LlamaInference.swift:1487-1654` and the weight arithmetic below).

## Evidence (why this change)

Device log, `a32c0bc`, 256-token generation, `Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf`:

```
MTP stats: rounds=143 hits=112 (78%) accepted=112 depth=0.78 no-draft=0 attempts=143 emitted=256;
  draft 8.95ms/rd + verify 0.89ms/rd + sample 65.16ms/rd + catchup 0.42ms/rd (143 rounds)
```

- `no-draft=0` — the `pMin=0` fix is confirmed working: no round falls back to a plain decode.
- Round total ≈ 75.4 ms; 143 rounds → 256 tokens (1.79 tokens/round) ⇒ ~42 ms/token (phase sum) / ~45 ms/token (wall).
- Weight arithmetic from the GGUF dump: target 4B Q4_K_M ≈ 2651 MiB ÷ 52.6 MiB/ms ≈ 50.4 ms/row (matches the known plain 50.3 ms/token); draft (block 71.2 + tied head 497.3) = 568.5 MiB ≈ 10.8 ms/row; a 2-row target verify ≈ 50.4 + ~7 ≈ 57 ms. Round ≈ 50.4 + 6 + 10.8 (draft) + 0.78 × 10.8 (catch-up) ≈ 75.6 ms — matches the observed 75.4 ms.
- **The `catchup 0.42ms/rd` in the log is only the CPU-side enqueue.** `decodeDraft` calls `llama_decode` asynchronously; the ~10.8 ms GPU cost is drained by the *next* round's draft sampler synchronize, so it is booked under `sample`. The catch-up is therefore a real ~0.78 × 10.8 ≈ 8.4 ms/round (≈11% of the round) that the per-phase line understates.
- Fusing it removes one of the two draft weight passes on ~78% of rounds: 75.4 → ~67 ms/round ⇒ ~1.2–1.3× over the current build, with the ceiling (100% acceptance, no catch-up) at ~69 ms/2 = 34.5 ms/token.

Ruled out by this data (do not revisit): a hidden multi-decade-millisecond cost elsewhere; the draft being "2× slow" (it is 568.5 MiB at full bandwidth); the gate (already fixed).

## Global Constraints

- One problem per commit; commit and push to `main`. (User rule.)
- CI-only builds (no local xcodeproj/xcodebuild). `.xcodeproj` is never committed; edit `project.yml` only if a source file is added. This plan adds no source file, so `project.yml` is untouched.
- No envelope-lookalike (tag-shaped) text in diffs, logs, comments, or commit messages. (User rule.)
- Tests are pure Swift and run on the simulator in CI (no model, no inference) — `xcodegen generate` → `xcodebuild test`, destination `iPhone 17`.
- Device work is split: the user installs the build and toggles MTP; the assistant measures over HTTP at `http://192.168.3.3:8443`. Use non-streaming requests (streaming is buffered on this build). Do not fabricate measurements.
- Keep `mtpMaxDraft = 1` (no UI toggle; depth 2 measured slower).

## Review Focus

1. **Multi-row draft batch output mapping** — the draft context is built with `n_outputs_max = mtpOutputCount` (2 at depth 1). A 2-row batch that requests logits on only the last row must still return that row at output slot 0 (`llama_sampler_sample(draftSampler, dft, 0)` / `llama_get_logits_ith(dft, 0)`). If the slot mapping shifts, the draft is sampled from the wrong row and acceptance silently collapses. Owned by Task 3 (acceptance must not drop below the historical 68–78%).
2. **Draft KV contiguity** — the seed rows must land at exactly the positions the next verify assumes (`nextPos + i + 1`), or the draft conditions on a wrong prefix and acceptance falls. Owned by Task 3 (`depth` in the stats line).
3. **Stopping rounds must not seed** — a round that stops (token limit, stop string, `eog`) must leave the draft context untouched, exactly as the old catch-up was guarded by `!stopped, firstStopRow == nil`. Owned by Task 1 (predicate) + Task 2 (guard placement).
4. **Depth-2 self-conditioning row** — the dormant chain path reads the draft's own `h` via `copyH(dft, 0)`; with a multi-row batch the correct row is the last one. Owned by Task 2.
5. **Rollback vs seed ordering** — a round with `rollbackPos` must trim both contexts *before* the next round's seed rows are written; at depth 1 the two are mutually exclusive (partial accepts produce no seed rows), but the plan makes the ordering explicit. Owned by Task 2.

---

### Task 1: Pure seed plan and drop the dead catch-up phase

**Files:**
- Modify: `Sources/App/MtpSpeculator.swift` (replace `catchupNeeded`, lines ~72-95)
- Modify: `Sources/App/MtpPhaseStats.swift` (drop `catchup`)
- Test: `Tests/LlamaServerTests/MtpSpeculatorTests.swift`
- Test: `Tests/LlamaServerTests/MtpPhaseStatsTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `static func MtpSpeculator.draftSeedIndices(emitCount: Int, draftEntries: Int) -> Range<Int>` — indices into the round's `reals` whose draft-context rows are missing and must be decoded as a prefix of the next round's draft decode. Token `i` sits at position `nextPos + i + 1` and pairs with target hidden-state row `i`.
  - `mutating func MtpPhaseStats.record(draft:verify:sample:)` (the `catchup:` parameter and the `catchup` term in `summary()` are removed).

- [ ] **Step 1: Write the failing tests**

In `MtpSpeculatorTests.swift`, replace the `catchupNeeded` tests with:

```swift
func testDraftSeedCoversTheGapLeftByAFullAccept() {
    // Depth 1 full accept: one draft decoded (position nextPos), two tokens
    // emitted; the draft is missing only reals[0] at nextPos + 1.
    XCTAssertEqual(MtpSpeculator.draftSeedIndices(emitCount: 2, draftEntries: 1), 0..<1)
}

func testDraftSeedIsEmptyOnARejectedOrPartialAccept() {
    XCTAssertEqual(MtpSpeculator.draftSeedIndices(emitCount: 1, draftEntries: 1), 0..<0)
    XCTAssertEqual(MtpSpeculator.draftSeedIndices(emitCount: 2, draftEntries: 2), 0..<1 - 1)
}

func testDraftSeedSkipsPositionsAChainCutAlreadyWrote() {
    // A chained link was decoded (draftEntries == 2) but cut, then the first
    // draft was accepted: position nextPos + 1 is already populated.
    XCTAssertEqual(MtpSpeculator.draftSeedIndices(emitCount: 2, draftEntries: 2), 0..<1 - 1)
}

func testDraftSeedAtDepthTwoCoversOnlyTheUnwrittenTail() {
    XCTAssertEqual(MtpSpeculator.draftSeedIndices(emitCount: 3, draftEntries: 2), 1..<2)
}
```

In `MtpPhaseStatsTests.swift`, update every `record(...)` call to the three-argument form and every `summary()` expectation to the string without the `catchup` term.

- [ ] **Step 2: Run tests to verify they fail**

Run: `xcodebuild test -project LlamaServer.xcodeproj -scheme LlamaServerTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17' -only-testing:LlamaServerTests/MtpSpeculatorTests -only-testing:LlamaServerTests/MtpPhaseStatsTests CODE_SIGNING_ALLOWED=NO`

Expected: FAIL — `draftSeedIndices` does not exist; `record` does not accept three arguments.

(CI is the real gate; a local run is optional.)

- [ ] **Step 3: Implement `draftSeedIndices` in `Sources/App/MtpSpeculator.swift`**

```swift
/// Indices into a round's `reals` whose draft-context rows are missing after
/// the round and must be decoded before the next round's own input row, so the
/// draft KV stays contiguous with the emitted sequence.
static func draftSeedIndices(emitCount: Int, draftEntries: Int) -> Range<Int> {
    let start = max(0, draftEntries - 1)
    let end = max(start, emitCount - 1)
    return start..<end
}
```

Delete `catchupNeeded` and its doc comment. Replace the class doc note about a chained gated link with the same statement rephrased against `draftEntries` (a decoded-then-cut link has already written its position, so `start` skips it).

- [ ] **Step 4: Drop `catchup` from `MtpPhaseStats`**

Remove `catchupTotal`, the `catchup:` parameter, and its `max(0, ...)` line; change `summary()` to `String(format: "draft %.2fms/rd + verify %.2fms/rd + sample %.2fms/rd (%d rounds)", ...)`. Update the doc examples that mention catch-up.

- [ ] **Step 5: Run tests to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Sources/App/MtpSpeculator.swift Sources/App/MtpPhaseStats.swift Tests/LlamaServerTests/MtpSpeculatorTests.swift Tests/LlamaServerTests/MtpPhaseStatsTests.swift
git commit -m "refactor(mtp): pure draft-seed plan replaces the catch-up predicate"
```

---

### Task 2: One multi-row draft decode per round

**Files:**
- Modify: `Sources/App/LlamaInference.swift` (`decodeDraft` at ~1413-1437; the MTP loop at ~1497-1653)

**Interfaces:**
- Consumes: `MtpSpeculator.draftSeedIndices(emitCount:draftEntries:) -> Range<Int>` (Task 1); existing `copyH(_:_:)`, `decodeVerify`, `MtpSpeculator.decide`, `decodeSingleTarget`.
- Produces: no public API change; `decodeDraft` becomes `decodeDraft(_ rows: [(token: llama_token, pos: llama_pos, h: [Float])]) -> Int32`.

- [ ] **Step 1: Generalize `decodeDraft` to N rows**

Replace the single-row `decodeDraft(_ token:_ pos:_ h:)` with a rows-array version:
- `llama_batch_init(Int32(rows.count), Int32(nEmbd), 1)`; allocate `batch.token` when `llama_batch_init` didn't (same `ownTok` dance as today).
- For row `i`: `batch.token[i] = rows[i].token`; copy `rows[i].h` into `batch.embd + i * nEmbd`; `batch.pos[i] = rows[i].pos`; `batch.n_seq_id[i] = 1`, `batch.seq_id[i]![0] = 0`; `batch.logits[i] = (i == rows.count - 1) ? 1 : 0` so the decoded row is output slot 0 for any row count.
- Decode on `dft`; free `ownTok` and the batch as today.

Update the depth-2 chain call site to `decodeDraft([(cand0, nextPos + 1, copyH(dft, Int32(rows.count - 1)) ?? pendingH)])`-style (one row there, so the self-conditioning row is `copyH(dft, 0)`; if the chain's own batch ever grows, it is the last row — see Review Focus 4).

- [ ] **Step 2: Defer the seed instead of decoding the catch-up**

Replace the catch-up block (`let carryH = copyH(...)` through the `phaseCatchup` record) with:
- Keep `let carryH = copyH(context, step.carryHFromRow) ?? pendingH`.
- Declare, before the loop, `var draftSeedRows: [(token: llama_token, pos: llama_pos, h: [Float])] = []`.
- On a round that does not stop (`!stopped`, `firstStopRow == nil`), set `draftSeedRows` from `MtpSpeculator.draftSeedIndices(emitCount: step.emitCount, draftEntries: draftEntries)`: for each `i`, append `(reals[i], nextPos + llama_pos(i + 1), copyH(context, Int32(i)) ?? pendingH)`. Otherwise leave it empty.
- Remove `phaseCatchup` and pass only `draft:`, `verify:`, `sample:` to `phaseStats.record(...)` (both the fallback and the main path).

- [ ] **Step 3: Prepend the seed rows to the next round's draft decode**

At the top of the round, replace `let drc0 = decodeDraft(idLast, nextPos, pendingH)` with a rows list:
```swift
var draftRows = draftSeedRows
draftSeedRows = []
draftRows.append((idLast, nextPos, pendingH))
let drc0 = decodeDraft(draftRows)
```
`drc0 == 0` still gates the round, and `llama_sampler_sample(draftSampler, dft, 0)` still samples the draft (the last row, i.e. `idLast`'s successor) — the seed rows only warm the KV.
Set `draftEntries = Int32(draftRows.count)` semantics clearly: `draftEntries` remains "positions written into the draft context this round counting from `nextPos`", so it counts only `idLast` plus any successful chain link (the seed rows belong to *previous* positions and must not enter this round's `draftSeedIndices`). Rename the local to make that explicit if it reads ambiguously.

- [ ] **Step 4: Compile in CI**

Push and watch the run:
```bash
git push origin main && gh run watch "$(gh run list --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
```
Expected: build job (compile) and test job both green.

- [ ] **Step 5: Commit**

```bash
git add Sources/App/LlamaInference.swift
git commit -m "perf(mtp): decode the accepted tokens into the draft context in one batch"
```

---

### Task 3: On-device verification

**Files:** none (measurement only).

**Interfaces:**
- Consumes: the `MTP stats:` line and the client wall-clock from tasks 1–2's build.

- [ ] **Step 1: Ask the user to install and toggle**

User installs `main` (MTP on), then runs one generation over the app. Assistant then runs the same non-streaming timed requests used for the baseline (128 `max_tokens`, `temperature 0`, spaced ~15 s, 5 runs) against `http://192.168.3.3:8443`.

- [ ] **Step 2: Read the acceptance**

Obtain the `MTP stats:` line for one run from the user's log. Expected: acceptance within the historical band (`hits/rounds` ≈ 68–78%, `depth` ≈ 0.68–0.78) and `no-draft=0`. A collapse means Review Focus 1 or 2 — revert Task 2 and report.

- [ ] **Step 3: Read the throughput**

Expected: ≈ 40 ms/token (~24–25 t/s) versus the pre-change ~45 ms/token, i.e. ~1.15–1.25× over the current build and ~1.3× over plain. Report the raw runs; do not round an improvement up, and if it is within noise say so.

- [ ] **Step 4: Keep or revert**

If acceptance held and throughput improved, keep and note the result on the commit. If not, revert Task 2's commit (`git revert`) and report the measured numbers with the reverted state, then re-plan from the new evidence.

---

## Self-Review

- **Spec coverage:** the evidence section names one change (fuse the catch-up) — Tasks 1–2 implement it; Task 3 verifies it. The `no-draft=0` confirmation and the phase-attribution finding are recorded, not acted on.
- **Step scan:** Task 1 steps each produce one testable thing; Task 2's steps 1–3 are code moves whose correctness is verified by CI compile (Step 4) and Task 3 (no pure logic to unit-test — `draftSeedIndices` is the pure part and is tested in Task 1); Task 3 steps are measurements with explicit expected outputs.
- **Type consistency:** `draftSeedIndices(emitCount:draftEntries:) -> Range<Int>` is used with the same labels in both tasks; `decodeDraft` takes `[(token:pos:h:)]` everywhere it is called; `phaseStats.record(draft:verify:sample:)` matches the Task 1 signature.
- **Review Focus:** each of the five lines is owned by a named task (1, 2, 3).
- **Proportion:** three tasks, no code transcripts; the only full function body is `draftSeedIndices` (four lines).
