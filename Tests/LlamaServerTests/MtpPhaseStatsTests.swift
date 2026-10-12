import XCTest

final class MtpPhaseStatsTests: XCTestCase {

    // MARK: - Empty state

    func testZeroRoundsSummary() {
        XCTAssertEqual(MtpPhaseStats().summary(), "no MTP rounds recorded")
    }

    // MARK: - Single round

    func testOneRoundAveragesEqualTheRecordedValues() {
        var s = MtpPhaseStats()
        s.record(draft: 0.00031, verify: 0.0119, sample: 0.00004)
        XCTAssertEqual(s.roundCount, 1)
        XCTAssertEqual(s.summary(), "draft 0.31ms/rd + verify 11.90ms/rd + sample 0.04ms/rd + catchup 0.00ms/rd (1 rounds)")
    }

    // MARK: - Aggregation across rounds

    func testThreeRoundsAveragePerRound() {
        var s = MtpPhaseStats()
        s.record(draft: 0.001, verify: 0.010, sample: 0.0001)
        s.record(draft: 0.001, verify: 0.020, sample: 0.0001)
        s.record(draft: 0.001, verify: 0.030, sample: 0.0001)
        XCTAssertEqual(s.roundCount, 3)
        XCTAssertEqual(s.summary(), "draft 1.00ms/rd + verify 20.00ms/rd + sample 0.10ms/rd + catchup 0.00ms/rd (3 rounds)")
    }

    // MARK: - Clock-skew clamping (same rule as StartupTimer)

    func testNegativeDurationsClampToZero() {
        var s = MtpPhaseStats()
        s.record(draft: -0.5, verify: 0.002, sample: 0)
        XCTAssertEqual(s.summary(), "draft 0.00ms/rd + verify 2.00ms/rd + sample 0.00ms/rd + catchup 0.00ms/rd (1 rounds)")
    }

    // MARK: - Fourth phase (draft catch-up decode, added with depth-2)

    func testCatchupPhaseAppearsInSummary() {
        var s = MtpPhaseStats()
        s.record(draft: 0.001, verify: 0.010, sample: 0.0001, catchup: 0.005)
        XCTAssertEqual(s.roundCount, 1)
        XCTAssertEqual(s.summary(), "draft 1.00ms/rd + verify 10.00ms/rd + sample 0.10ms/rd + catchup 5.00ms/rd (1 rounds)")
    }

    func testCatchupDefaultsToZeroWhenNotRecorded() {
        var s = MtpPhaseStats()
        s.record(draft: 0.001, verify: 0.010, sample: 0.0001, catchup: 0)
        s.record(draft: 0.001, verify: 0.010, sample: 0.0001, catchup: 0.004)
        // catchup average = 0.004 / 2 = 2.00ms/rd
        XCTAssertEqual(s.summary(), "draft 1.00ms/rd + verify 10.00ms/rd + sample 0.10ms/rd + catchup 2.00ms/rd (2 rounds)")
    }
}
