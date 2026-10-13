import XCTest

/// Generalized verify-round decisions for depth-N MTP speculative decoding.
/// The depth-1 cases are restated against the new API (every historical
/// invariant is kept), followed by the depth-2 matrix.
final class MtpSpeculatorTests: XCTestCase {

    // MARK: - Depth-1 equivalents (full accept / miss)

    func testDepth1FullAcceptMatchesOldHitInvariants() {
        let d = MtpSpeculator.decide(drafts: [42], reals: [42, 7], firstStopRow: nil, nextPos: 10)
        XCTAssertEqual(d.accepted, 1)
        XCTAssertEqual(d.emitCount, 2, "idLast + the drafted token")
        XCTAssertTrue(d.hit)
        XCTAssertNil(d.rollbackPos, "nothing to roll back on a full accept")
        XCTAssertEqual(d.carryHFromRow, 1, "must carry the row-1 hidden state (pairs with the bonus token)")
    }

    func testDepth1MissMatchesOldMissInvariants() {
        let d = MtpSpeculator.decide(drafts: [42], reals: [99, 7], firstStopRow: nil, nextPos: 10)
        XCTAssertEqual(d.accepted, 0)
        XCTAssertEqual(d.emitCount, 1, "a miss only commits the resampled token")
        XCTAssertFalse(d.hit)
        XCTAssertEqual(d.rollbackPos, 11, "the drafted token sits at nextPos + 1")
        XCTAssertEqual(d.carryHFromRow, 0, "must carry the row-0 hidden state (pairs with the resampled token)")
    }

    // MARK: - Depth-2 matrix

    func testDepth2FullAcceptEmitsAllThreeAndCarriesRow2() {
        let d = MtpSpeculator.decide(drafts: [7, 8], reals: [7, 8, 33], firstStopRow: nil, nextPos: 10)
        XCTAssertEqual(d.accepted, 2)
        XCTAssertEqual(d.emitCount, 3)
        XCTAssertTrue(d.hit)
        XCTAssertNil(d.rollbackPos, "every decoded row was consumed")
        XCTAssertEqual(d.carryHFromRow, 2, "last emitted token is row 2's successor — pairs with reals[2]")
    }

    func testDepth2PartialAcceptRollsBackExactlyTheUnconsumedRows() {
        let d = MtpSpeculator.decide(drafts: [7, 8], reals: [7, 9, 33], firstStopRow: nil, nextPos: 10)
        XCTAssertEqual(d.accepted, 1)
        XCTAssertEqual(d.emitCount, 2)
        XCTAssertTrue(d.hit)
        XCTAssertEqual(d.rollbackPos, 12, "nextPos + emitCount — drop the rejected d1 only")
        XCTAssertEqual(d.carryHFromRow, 1)
    }

    func testDepth2FirstMissBehavesLikeDepth1Miss() {
        let d = MtpSpeculator.decide(drafts: [7, 8], reals: [9, 33, 34], firstStopRow: nil, nextPos: 10)
        XCTAssertEqual(d.accepted, 0)
        XCTAssertEqual(d.emitCount, 1)
        XCTAssertFalse(d.hit)
        XCTAssertEqual(d.rollbackPos, 11)
        XCTAssertEqual(d.carryHFromRow, 0)
    }

    // MARK: - Stop capping (eog sampled as a real; drafts are never eog)

    func testEogAtRow0EmitsNothing() {
        let d = MtpSpeculator.decide(drafts: [7, 8], reals: [4444, 33, 34], firstStopRow: 0, nextPos: 10)
        XCTAssertEqual(d.emitCount, 0, "the eog token is never emitted")
        XCTAssertEqual(d.rollbackPos, 10, "roll back from nextPos itself")
        XCTAssertEqual(d.carryHFromRow, 0, "clamped: no emitted row to pair with")
    }

    func testEogAtRow1CapsAWouldBeFullAccept() {
        // Invariant: reals.count == drafts.count + 1 (3 entries); eog at row 1.
        let d = MtpSpeculator.decide(drafts: [7, 8], reals: [7, 4444, 34], firstStopRow: 1, nextPos: 10)
        XCTAssertEqual(d.accepted, 1)
        XCTAssertEqual(d.emitCount, 1, "capped at the first stop row")
        XCTAssertEqual(d.rollbackPos, 11)
        XCTAssertEqual(d.carryHFromRow, 0)
    }

    // MARK: - h/token pairing invariant across every depth (the historically
    //        inverted bug): carryHFromRow is ALWAYS emitCount - 1.

    func testCarryRowAlwaysPairsWithLastEmittedRow() {
        let cases: [(drafts: [Int32], reals: [Int32], stop: Int?)] = [
            ([42], [42, 7], nil),
            ([42], [99, 7], nil),
            ([7, 8], [7, 8, 33], nil),
            ([7, 8], [7, 9, 33], nil),
            ([7, 8], [9, 33, 34], nil),
            ([7, 8], [4444, 33, 34], 0),
            ([7, 8], [7, 4444, 34], 1),
        ]
        for c in cases {
            let d = MtpSpeculator.decide(drafts: c.drafts, reals: c.reals, firstStopRow: c.stop, nextPos: 3)
            XCTAssertEqual(d.carryHFromRow, Int32(max(0, d.emitCount - 1)),
                           "carry row must pair with the last emitted row for \(c)")
        }
    }

    // MARK: - Rollback tracks nextPos (position independence)

    func testRollbackTracksNextPos() {
        for p in [0, 10, 1000] {
            let d = MtpSpeculator.decide(drafts: [7, 8], reals: [9, 33, 34], firstStopRow: nil, nextPos: p)
            XCTAssertEqual(d.rollbackPos, p + 1)
        }
    }

    func testFullConsumeHasNoRollback() {
        XCTAssertEqual(MtpSpeculator.decide(drafts: [7, 8], reals: [7, 8, 33],
                                             firstStopRow: nil, nextPos: 999).rollbackPos, nil)
        XCTAssertEqual(MtpSpeculator.decide(drafts: [7], reals: [7, 33],
                                             firstStopRow: nil, nextPos: 999).rollbackPos, nil)
    }

    // MARK: - Token-value edge cases (decision depends only on equality)

    func testExtremeTokenIdsCompareByEquality() {
        XCTAssertTrue(MtpSpeculator.decide(drafts: [0], reals: [0, 1], firstStopRow: nil, nextPos: 4).hit)
        XCTAssertTrue(MtpSpeculator.decide(drafts: [-1], reals: [-1, 1], firstStopRow: nil, nextPos: 4).hit)
        XCTAssertFalse(MtpSpeculator.decide(drafts: [0], reals: [-1, 1], firstStopRow: nil, nextPos: 4).hit)
        XCTAssertFalse(MtpSpeculator.decide(drafts: [Int32.max], reals: [Int32.min, 1], firstStopRow: nil, nextPos: 4).hit)
    }

    // MARK: - Catch-up decode eligibility (draft-context gap)

    /// Depth-1 full accept: only nextPos was written, so the gap at nextPos+1
    /// still needs filling.
    func testCatchupNeededOnDepth1FullAccept() {
        XCTAssertTrue(MtpSpeculator.catchupNeeded(emitCount: 2, draftCount: 1, draftEntries: 1))
    }

    /// Depth-2 full accept: nextPos and nextPos+1 were written; the gap at
    /// nextPos+2 is still unwritten.
    func testCatchupNeededOnDepth2FullAccept() {
        XCTAssertTrue(MtpSpeculator.catchupNeeded(emitCount: 3, draftCount: 2, draftEntries: 2))
    }

    /// Gated chain cut: the rejected second draft was still DECODED (so its
    /// position nextPos+1 is populated). Catch-up must NOT re-decode it — the
    /// recurrent draft context does not support rewriting a consumed position,
    /// and the re-decode would carry the target's h, not the chain's.
    func testNoCatchupWhenGatedChainLinkAlreadyWritten() {
        XCTAssertFalse(MtpSpeculator.catchupNeeded(emitCount: 2, draftCount: 1, draftEntries: 2))
    }

    /// Chain decode failed (position nextPos+1 never written): the gap still
    /// needs filling even though only one draft survives.
    func testCatchupNeededWhenChainDecodeFailed() {
        XCTAssertTrue(MtpSpeculator.catchupNeeded(emitCount: 2, draftCount: 1, draftEntries: 1))
    }

    /// Partial accept and miss roll back instead of catching up.
    func testNoCatchupOnPartialAcceptOrMiss() {
        XCTAssertFalse(MtpSpeculator.catchupNeeded(emitCount: 1, draftCount: 1, draftEntries: 1))
        XCTAssertFalse(MtpSpeculator.catchupNeeded(emitCount: 1, draftCount: 2, draftEntries: 2))
        XCTAssertFalse(MtpSpeculator.catchupNeeded(emitCount: 2, draftCount: 2, draftEntries: 2))
    }
}
