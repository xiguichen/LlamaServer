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

    func testHybridLiveCacheRejectedWhenEmpty() {
        XCTAssertNil(ReuseSurvival.liveHybridReuseLen(cachedTokens: [],
                                                      promptTokens: [1],
                                                      seqPosMin: 0,
                                                      seqPosMax: 0))
    }
}
