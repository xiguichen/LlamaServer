import XCTest

final class MtpSpeculatorTests: XCTestCase {

    // MARK: - HIT (draft accepted: d0 == s0)

    func testHitAdvancesByTwoAndCarriesRow1() {
        let d = MtpSpeculator.decide(d0: 42, s0: 42, nextPos: 10)
        XCTAssertTrue(d.hit)
        XCTAssertEqual(d.posAdvance, 2, "a hit consumes idLast + the drafted token")
        XCTAssertNil(d.rollbackPos, "nothing to roll back on a hit")
        XCTAssertEqual(d.carryHFromRow, 1, "hit must carry the row-1 hidden state (pairs with the bonus token)")
    }

    func testHitDecisionIsIndependentOfPosition() {
        let a = MtpSpeculator.decide(d0: 7, s0: 7, nextPos: 0)
        let b = MtpSpeculator.decide(d0: 7, s0: 7, nextPos: 9999)
        XCTAssertEqual(a, b)
        XCTAssertNil(a.rollbackPos)
    }

    // MARK: - MISS (draft rejected: d0 != s0)

    func testMissAdvancesByOneAndCarriesRow0() {
        let d = MtpSpeculator.decide(d0: 42, s0: 99, nextPos: 10)
        XCTAssertFalse(d.hit)
        XCTAssertEqual(d.posAdvance, 1, "a miss only commits the resampled token")
        XCTAssertEqual(d.carryHFromRow, 0, "miss must carry the row-0 hidden state (pairs with the resampled token)")
    }

    func testMissRollsBackTheDraftedPosition() {
        // The drafted token sits at nextPos + 1; that slot must be removed.
        let d = MtpSpeculator.decide(d0: 1, s0: 2, nextPos: 10)
        XCTAssertEqual(d.rollbackPos, 11)
    }

    func testMissRollbackTracksNextPos() {
        XCTAssertEqual(MtpSpeculator.decide(d0: 1, s0: 2, nextPos: 0).rollbackPos, 1)
        XCTAssertEqual(MtpSpeculator.decide(d0: 1, s0: 2, nextPos: 5).rollbackPos, 6)
        XCTAssertEqual(MtpSpeculator.decide(d0: 1, s0: 2, nextPos: 1000).rollbackPos, 1001)
    }

    // MARK: - h/token pairing invariant (the historically inverted bug)

    func testCarryRowDiffersBetweenHitAndMiss() {
        let hit = MtpSpeculator.decide(d0: 5, s0: 5, nextPos: 3)
        let miss = MtpSpeculator.decide(d0: 5, s0: 6, nextPos: 3)
        XCTAssertEqual(hit.carryHFromRow, 1)
        XCTAssertEqual(miss.carryHFromRow, 0)
        XCTAssertNotEqual(hit.carryHFromRow, miss.carryHFromRow,
                          "the hit/miss hidden-state pairing must never collapse to the same row")
    }

    // MARK: - Token-value edge cases (decision depends only on equality)

    func testNegativeAndZeroTokenIdsCompareByEquality() {
        XCTAssertTrue(MtpSpeculator.decide(d0: 0, s0: 0, nextPos: 4).hit)
        XCTAssertTrue(MtpSpeculator.decide(d0: -1, s0: -1, nextPos: 4).hit)
        XCTAssertFalse(MtpSpeculator.decide(d0: 0, s0: -1, nextPos: 4).hit)
        XCTAssertFalse(MtpSpeculator.decide(d0: Int32.max, s0: Int32.min, nextPos: 4).hit)
    }
}
