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
}
