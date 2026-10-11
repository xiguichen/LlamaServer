import XCTest

/// Tests for the periodic-recreate bookkeeping shared by the streaming, chat,
/// and completions handlers.
///
/// Why this exists: all three handlers duplicated the same
/// `generationCount + 1 >= threshold` block inline. Extracting it makes the
/// counter semantics (including retry-on-failed-recreate) testable and lets
/// the recreate move OFF the response path — it used to run between the last
/// token and the finish_reason chunk, stalling every 10th streamed response
/// by the KV-cache re-allocation time (~0.3–0.5 s).
final class RecreateTrackerTests: XCTestCase {

    func testRecreatesEveryTenthSuccessfulGeneration() {
        var tracker = RecreateTracker(threshold: 10)
        for gen in 1...9 {
            XCTAssertFalse(tracker.record(), "generation \(gen) should not be due yet")
        }
        XCTAssertTrue(tracker.record(), "the 10th generation triggers the recreate")
        tracker.recreateFinished(success: true)
        // Second cycle: same cadence after the reset.
        for _ in 1...9 { XCTAssertFalse(tracker.record()) }
        XCTAssertTrue(tracker.record())
    }

    func testFailedRecreateRetriesNextGeneration() {
        var tracker = RecreateTracker(threshold: 3)
        XCTAssertFalse(tracker.record())
        XCTAssertFalse(tracker.record())
        XCTAssertTrue(tracker.record())
        // The recreate attempt failed — the counter must NOT reset, so the
        // very next generation retries instead of waiting another cycle.
        tracker.recreateFinished(success: false)
        XCTAssertTrue(tracker.record(), "failed recreate must retry on the next generation")
        tracker.recreateFinished(success: true)
        XCTAssertFalse(tracker.record(), "after a confirmed reset the cadence restarts")
    }

    func testThresholdOfOneRecreatesEveryGeneration() {
        var tracker = RecreateTracker(threshold: 1)
        XCTAssertTrue(tracker.record())
        tracker.recreateFinished(success: true)
        XCTAssertTrue(tracker.record())
    }
}
