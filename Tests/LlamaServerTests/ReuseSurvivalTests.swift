import XCTest

/// Tests for whether a rebuild's sequence-state snapshot can be carried across a
/// context recreate.
///
/// Why this exists: `recreateContext()` builds a brand-new llama_context and used
/// to always drop `cachedTokenIds`/`hasSnapshot`, so the next turn could never
/// reuse the shared prompt prefix. For a dense model that re-decodes thousands of
/// prompt tokens it never needed to — the on-disk sequence-state blob (`seq_0`)
/// written before the old context was freed is fully restorable into the fresh
/// context, exactly like the hybrid path already relies on. The decision of
/// whether that carry-over is safe is pure, so it is pinned here.
final class ReuseSurvivalTests: XCTestCase {

    func testRecreateSnapshotSurvivesForReusableView() {
        XCTAssertTrue(ReuseSurvival.snapshotSurvivesRecreate(supportsPartialReuse: true),
                      "dense model: the on-disk seq_0 blob is restorable into a fresh context, so the snapshot must carry")
    }

    func testRecreateSnapshotSurvivesForHybridModel() {
        XCTAssertTrue(ReuseSurvival.snapshotSurvivesRecreate(supportsPartialReuse: false),
                      "hybrid: already relied on the snapshot; the recreate must not break that")
    }

    func testDenseCaptureNeededWhenCacheHasTokens() {
        XCTAssertTrue(ReuseSurvival.denseSnapshotCaptureNeeded(supportsPartialReuse: true,
                                                               hasCachedTokens: true),
                      "dense + live cache: the recreate would otherwise destroy the only reusable prefix")
    }

    func testDenseCaptureSkippedWithoutCachedTokens() {
        XCTAssertFalse(ReuseSurvival.denseSnapshotCaptureNeeded(supportsPartialReuse: true,
                                                                hasCachedTokens: false),
                       "nothing to capture after an MTP turn already cleared the cache")
    }

    func testHybridCaptureNotNeededAtRecreate() {
        XCTAssertFalse(ReuseSurvival.denseSnapshotCaptureNeeded(supportsPartialReuse: false,
                                                                hasCachedTokens: true),
                       "hybrid already saves its snapshot during generation")
    }

    // MARK: - Per-turn hybrid snapshot save

    func testHybridSnapshotSkippedWhenLiveCacheReusedAndNoRecreate() {
        XCTAssertFalse(ReuseSurvival.hybridSnapshotSaveNeeded(recreateDue: false,
                                                              liveCacheReused: true),
                       "the live cache already carries the prefix; the snapshot is only a fallback")
    }

    func testHybridSnapshotSavedWhenRecreateIsDue() {
        XCTAssertTrue(ReuseSurvival.hybridSnapshotSaveNeeded(recreateDue: true,
                                                             liveCacheReused: true),
                      "the recreate destroys the live cache; the snapshot is the only survivor")
    }

    func testHybridSnapshotSavedWhenLiveCacheWasNotReused() {
        XCTAssertTrue(ReuseSurvival.hybridSnapshotSaveNeeded(recreateDue: false,
                                                             liveCacheReused: false),
                      "the snapshot is the active reuse path (diverged prompt, cleared cache, or cold start), so refresh it")
    }

    // MARK: - Live hybrid prefix reuse

    func testHybridLiveCacheReusedWhenItIsAStrictPromptPrefix() {
        let cached: [Int32] = [1, 2, 3]
        let prompt: [Int32] = [1, 2, 3, 4, 5]
        XCTAssertEqual(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                        promptTokens: prompt,
                                                        seqPosMin: 0,
                                                        seqPosMax: 2),
                       3,
                       "the whole live cache is a prefix and nothing needs trimming")
    }

    func testHybridLiveCacheRejectedWhenItDiverges() {
        let cached: [Int32] = [1, 2, 9]
        let prompt: [Int32] = [1, 2, 3, 4]
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: prompt,
                                                      seqPosMin: 0,
                                                      seqPosMax: 2),
                     "a hybrid cache cannot trim the divergent suffix, so it is unusable whole")
    }

    func testHybridLiveCacheRejectedWhenItEqualsThePrompt() {
        let cached: [Int32] = [1, 2, 3]
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: cached,
                                                      seqPosMin: 0,
                                                      seqPosMax: 2),
                     "at least one prompt token must be decoded to produce fresh logits")
    }

    func testHybridLiveCacheRejectedWhenPrefixWasEvicted() {
        let cached: [Int32] = [1, 2, 3]
        let prompt: [Int32] = [1, 2, 3, 4]
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: prompt,
                                                      seqPosMin: 1,
                                                      seqPosMax: 2),
                     "seq_pos_min > 0 means the reusable prefix is no longer resident")
    }

    func testHybridLiveCacheRejectedWhenBoundsDisagreeWithTracking() {
        let cached: [Int32] = [1, 2, 3]
        let prompt: [Int32] = [1, 2, 3, 4]
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: prompt,
                                                      seqPosMin: 0,
                                                      seqPosMax: 1),
                     "a shorter live sequence than cachedTokenIds describes an inconsistent cache")
    }

    func testHybridFullAttentionAcceptsRecurrentTailAsPosMin() {
        // seq_pos_min is max(attention_min, recurrent_min): the recurrent part
        // only keeps its tail, so posMin = posMax is normal for a full-attention
        // hybrid model. The end bound still has to match.
        let cached: [Int32] = [1, 2, 3]
        let prompt: [Int32] = [1, 2, 3, 4]
        XCTAssertEqual(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                        promptTokens: prompt,
                                                        seqPosMin: 2,
                                                        seqPosMax: 2,
                                                        hybridFullAttention: true),
                       3)
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: prompt,
                                                      seqPosMin: 2,
                                                      seqPosMax: 1,
                                                      hybridFullAttention: true),
                     "the end bound must still match trackedTokenIds")
    }

    func testHybridSlidingWindowStillRequiresAnUnEvictedFront() {
        let cached: [Int32] = [1, 2, 3]
        let prompt: [Int32] = [1, 2, 3, 4]
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: cached,
                                                      promptTokens: prompt,
                                                      seqPosMin: 2,
                                                      seqPosMax: 2,
                                                      hybridFullAttention: false),
                     "a hybrid model with SWA can evict the front; posMin must be 0")
    }

    func testHybridLiveCacheRejectedWhenEmpty() {
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: [],
                                                      promptTokens: [1],
                                                      seqPosMin: 0,
                                                      seqPosMax: 0))
    }

    // MARK: - Snapshot restore guard (whole-prompt blobs are never reusable)

    /// The 2026-11-14 device log, replayed as pure logic. A 15628-token prompt was
    /// snapshotted whole; the next prompt shared its first 15627 tokens, so the
    /// old guard rejected the blob with `lcp 15627/15628; full decode` and spent
    /// 65.25 s re-prefilling 15653 tokens.
    func testWholePromptSnapshotIsRejectedWhenTheTrailingTokenDrifts() {
        let snapshotted: [Int32] = [10, 11, 12, 13, 14]
        let nextPrompt: [Int32] = [10, 11, 12, 13, 99, 20, 21]
        let lcp = commonPrefix(snapshotted, nextPrompt)

        XCTAssertEqual(lcp, 4, "the reply replaced the primer's last token")
        XCTAssertNil(ReuseSurvival.snapshotRestorablePrefix(lcp: lcp,
                                                            snapshotCount: snapshotted.count,
                                                            promptCount: nextPrompt.count),
                     "this is the '+65.25 s full decode' branch")
    }

    func testHeldBackSnapshotIsRestoredForTheSameTrailingTokenDrift() {
        let snapshotted: [Int32] = [10, 11, 12, 13]     // PromptBoundary holds 14 back
        let nextPrompt: [Int32] = [10, 11, 12, 13, 99, 20, 21]
        let lcp = commonPrefix(snapshotted, nextPrompt)

        XCTAssertEqual(lcp, 4)
        XCTAssertEqual(ReuseSurvival.snapshotRestorablePrefix(lcp: lcp,
                                                              snapshotCount: snapshotted.count,
                                                              promptCount: nextPrompt.count),
                       4,
                       "same drift, same lcp — the blob is now a strict prefix and restores")
    }

    /// The repeated-identical-prompt case from the same log
    /// (`lcp 15653/15653, prompt 15653; full decode`).
    func testIdenticalPromptRestoresFromAHeldBackSnapshot() {
        let whole: [Int32] = [10, 11, 12, 13, 14]
        XCTAssertNil(ReuseSurvival.snapshotRestorablePrefix(lcp: whole.count,
                                                            snapshotCount: whole.count,
                                                            promptCount: whole.count),
                     "a whole-prompt blob leaves nothing to decode for fresh logits")

        let heldBack: [Int32] = [10, 11, 12, 13]
        XCTAssertEqual(ReuseSurvival.snapshotRestorablePrefix(lcp: heldBack.count,
                                                              snapshotCount: heldBack.count,
                                                              promptCount: whole.count),
                       4,
                       "one token is decoded instead of all five")
    }

    func testSnapshotRejectedWhenThePrefixDivergesMidway() {
        XCTAssertNil(ReuseSurvival.snapshotRestorablePrefix(lcp: 1950,
                                                            snapshotCount: 2054,
                                                            promptCount: 15602),
                     "a genuine divergence still has to re-decode")
    }

    func testSnapshotRejectedWhenNoPromptTokenRemains() {
        XCTAssertNil(ReuseSurvival.snapshotRestorablePrefix(lcp: 5, snapshotCount: 5, promptCount: 5))
    }

    private func commonPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }
}
