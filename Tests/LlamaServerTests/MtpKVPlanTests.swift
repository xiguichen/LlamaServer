import XCTest

/// Pins the bookkeeping that lets the target KV survive an MTP generation.
///
/// The MTP loop decodes `idLast` and the drafts, then rolls back the rows that
/// were not emitted. The last emitted token is a *prediction* sampled from the
/// last resident row; it is not itself in the KV. Publishing the wrong count
/// would let the next request skip decoding tokens that are not resident, so
/// the rules and the runtime bounds guard are pure and tested here.
final class MtpKVPlanTests: XCTestCase {

    func testNormalRoundLeavesLastEmittedTokenUndecoded() {
        // prompt 100, one token already emitted, round starts at 100.
        let start = 100
        let resident = MtpKVPlan.committedResident(roundStartPos: start, emitted: 2)
        XCTAssertEqual(resident, 102)
        // generated = 1 (the first token) + 2 (this round)
        XCTAssertEqual(resident, 100 + 3 - 1)

        let ids = MtpKVPlan.residentIds(prompt: Array(0..<100),
                                        emitted: [1000, 1001, 1002],
                                        resident: resident)
        XCTAssertEqual(ids?.count, 102)
        XCTAssertEqual(ids?.last, 1001, "the last emitted token (1002) is a prediction, not resident")
    }

    func testIdleGenerationKeepsWholePrompt() {
        // eog_immediate: nothing was decoded beyond the prompt.
        let ids = MtpKVPlan.residentIds(prompt: [1, 2, 3], emitted: [], resident: 3)
        XCTAssertEqual(ids, [1, 2, 3])
    }

    func testFinalFallbackEogKeepsLastEmittedToken() {
        // prompt 100, 3 emitted, last round started at 102 and decoded idLast.
        let resident = MtpKVPlan.fallbackStopResident(roundStartPos: 102)
        XCTAssertEqual(resident, 103)
        XCTAssertEqual(resident, 100 + 3, "fallback eog leaves the last emitted token resident")

        let ids = MtpKVPlan.residentIds(prompt: Array(0..<100),
                                        emitted: [1000, 1001, 1002],
                                        resident: resident)
        XCTAssertEqual(ids?.last, 1002)
    }

    func testVerifyEogDropsLastEmittedToken() {
        // The eog row is rolled back to the round start, so the previous
        // round's last emitted token is no longer resident.
        let roundStart = 102          // prompt 100 + generated 3 - 1
        let resident = MtpKVPlan.committedResident(roundStartPos: roundStart, emitted: 0)
        XCTAssertEqual(resident, 102)
        XCTAssertEqual(resident, 100 + 3 - 1)

        let ids = MtpKVPlan.residentIds(prompt: Array(0..<100),
                                        emitted: [1000, 1001, 1002],
                                        resident: resident)
        XCTAssertEqual(ids?.last, 1001)
    }

    func testStoppedRoundTrimsToActualEmitted() {
        let resident = MtpKVPlan.committedResident(roundStartPos: 100, emitted: 1)
        let ids = MtpKVPlan.residentIds(prompt: Array(0..<100),
                                        emitted: [1000, 1001],
                                        resident: resident)
        XCTAssertEqual(ids?.last, 1000, "the unemitted second row must not be published")
    }

    func testResidentIdsRejectInconsistentCounts() {
        XCTAssertNil(MtpKVPlan.residentIds(prompt: [1, 2], emitted: [3], resident: 1),
                     "fewer tokens than the prompt cannot be a valid cache state")
        XCTAssertNil(MtpKVPlan.residentIds(prompt: [1, 2], emitted: [3], resident: 5),
                     "more generated tokens than were emitted cannot be valid")
    }

    func testBoundsGuardMatchesOnlyAnExactResidentPrefix() {
        XCTAssertTrue(MtpKVPlan.boundsMatch(posMin: 0, posMax: 101, resident: 102))
        XCTAssertFalse(MtpKVPlan.boundsMatch(posMin: 0, posMax: 100, resident: 102),
                       "cache shorter than the tracked prefix")
        XCTAssertFalse(MtpKVPlan.boundsMatch(posMin: 1, posMax: 101, resident: 102),
                       "prefix was evicted")
        XCTAssertFalse(MtpKVPlan.boundsMatch(posMin: 0, posMax: 102, resident: 102),
                       "cache longer than the tracked prefix")
    }

    func testLiveResidentCountClampsToBoundsAndRejectsEvictedFront() {
        XCTAssertEqual(MtpKVPlan.liveResidentCount(posMin: 0, posMax: 101), 102)
        XCTAssertEqual(MtpKVPlan.liveResidentCount(posMin: 0, posMax: -1), 0,
                       "empty sequence")
        XCTAssertEqual(MtpKVPlan.liveResidentCount(posMin: 1, posMax: 101), 0,
                       "an evicted front makes the tracked prefix unusable")
    }

    func testDraftReuseCoversGeneratedRows() {
        // The target preserved prompt (3) + 2 generated rows; the draft holds
        // the same sequence. The next prompt reuses all 5 tokens.
        let prompt: [Int32] = [1, 2, 3, 40, 41, 6]
        let draft: [Int32] = [1, 2, 3, 40, 41]
        XCTAssertEqual(MtpKVPlan.draftReuseLen(reuseLen: 5,
                                               draftTokens: draft,
                                               promptTokens: prompt),
                       5, "reuse past the old prompt must still trim the draft")
    }

    func testDraftReuseRejectsPrefixBeyondItsRows() {
        // Target reused 6 rows but the draft only holds the 3-token prompt:
        // trimming it to 6 is impossible, and mirroring only [6, …) would
        // leave a position gap.
        XCTAssertNil(MtpKVPlan.draftReuseLen(reuseLen: 6,
                                             draftTokens: [1, 2, 3],
                                             promptTokens: [1, 2, 3, 40, 41, 6]))
    }

    func testDraftReuseRejectsDivergedTokens() {
        XCTAssertNil(MtpKVPlan.draftReuseLen(reuseLen: 4,
                                             draftTokens: [1, 2, 3, 40],
                                             promptTokens: [1, 2, 3, 41]),
                     "a different generated token invalidates the draft prefix")
    }

    func testDraftReuseRequiresPositiveLength() {
        XCTAssertNil(MtpKVPlan.draftReuseLen(reuseLen: 0,
                                             draftTokens: [1, 2],
                                             promptTokens: [1, 2]),
                     "an empty reuse clears the draft instead of trimming")
    }
}
