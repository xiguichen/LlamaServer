import XCTest

final class MtpPhaseStatsTests: XCTestCase {

    // MARK: - Empty state

    func testZeroRoundsSummary() {
        XCTAssertEqual(MtpPhaseStats().summary(), "no MTP rounds recorded")
    }

    // MARK: - Single round

    func testOneRoundAveragesEqualTheRecordedValues() {
        var s = MtpPhaseStats()
        s.record(draft: 0.00031, verify: 0.0009, sample: 0.00004,
                 draftGpu: 0.009, verifyGpu: 0.052, emit: 0.00002, wall: 0.06237)
        XCTAssertEqual(s.roundCount, 1)
        XCTAssertEqual(s.summary(),
                       "draft 0.31ms/rd + verify 0.90ms/rd + sample 0.04ms/rd (1 rounds); "
                       + "draft-gpu 9.00ms/rd + verify-gpu 52.00ms/rd + emit 0.02ms/rd + host 0.10ms/rd = wall 62.37ms/rd")
    }

    func testLegacyPrefixIsUnchanged() {
        // Log parsers written for the original line match its prefix.
        var s = MtpPhaseStats()
        s.record(draft: 0.00031, verify: 0.0119, sample: 0.00004,
                 draftGpu: 0, verifyGpu: 0, emit: 0, wall: 0.01225)
        XCTAssertTrue(s.summary().hasPrefix("draft 0.31ms/rd + verify 11.90ms/rd + sample 0.04ms/rd (1 rounds)"))
    }

    // MARK: - Aggregation across rounds

    func testThreeRoundsAveragePerRound() {
        var s = MtpPhaseStats()
        s.record(draft: 0.001, verify: 0.001, sample: 0.0001,
                 draftGpu: 0.010, verifyGpu: 0.040, emit: 0.0001, wall: 0.0530)
        s.record(draft: 0.001, verify: 0.001, sample: 0.0001,
                 draftGpu: 0.010, verifyGpu: 0.050, emit: 0.0001, wall: 0.0630)
        s.record(draft: 0.001, verify: 0.001, sample: 0.0001,
                 draftGpu: 0.010, verifyGpu: 0.060, emit: 0.0001, wall: 0.0730)
        XCTAssertEqual(s.roundCount, 3)
        XCTAssertEqual(s.summary(),
                       "draft 1.00ms/rd + verify 1.00ms/rd + sample 0.10ms/rd (3 rounds); "
                       + "draft-gpu 10.00ms/rd + verify-gpu 50.00ms/rd + emit 0.10ms/rd + host 0.80ms/rd = wall 63.00ms/rd")
    }

    // MARK: - Clock-skew clamping (same rule as StartupTimer)

    func testNegativeDurationsClampToZero() {
        var s = MtpPhaseStats()
        s.record(draft: -0.5, verify: 0.002, sample: 0,
                 draftGpu: -0.1, verifyGpu: 0, emit: 0, wall: 0.002)
        XCTAssertEqual(s.summary(),
                       "draft 0.00ms/rd + verify 2.00ms/rd + sample 0.00ms/rd (1 rounds); "
                       + "draft-gpu 0.00ms/rd + verify-gpu 0.00ms/rd + emit 0.00ms/rd + host 0.00ms/rd = wall 2.00ms/rd")
    }

    func testHostNeverNegativeWhenWallUnderrunsPhases() {
        var s = MtpPhaseStats()
        s.record(draft: 0.001, verify: 0.001, sample: 0.001,
                 draftGpu: 0.001, verifyGpu: 0.001, emit: 0.001, wall: 0.003)
        XCTAssertTrue(s.summary().contains("host 0.00ms/rd = wall 3.00ms/rd"))
    }
}
