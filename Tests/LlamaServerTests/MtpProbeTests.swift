import XCTest

final class MtpProbeTests: XCTestCase {

    // MARK: - mtpCount == 1 (MTP disabled)

    func testReturns1WhenMtpDisabled() {
        var checked = false
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 1,
            genTokens: 5,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { _ in true }
        )
        XCTAssertEqual(result, 1)
        XCTAssertFalse(checked)
        XCTAssertFalse(active)
    }

    // MARK: - Phase 1: first generation token

    func testFirstTokenAlwaysReturns1() {
        var checked = false
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 1,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { _ in true }
        )
        XCTAssertEqual(result, 1)
        XCTAssertFalse(checked, "should not probe on first token")
        XCTAssertFalse(active)
    }

    // MARK: - Phase 2: probe once on second token

    func testSecondTokenProbesAndFindsMtpActive() {
        var checked = false
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 2,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 || idx == -2 }
        )
        XCTAssertEqual(result, 2)
        XCTAssertTrue(checked)
        XCTAssertTrue(active)
    }

    func testSecondTokenProbesAndFindsOnlyMainHead() {
        var checked = false
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 2,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 }
        )
        XCTAssertEqual(result, 1)
        XCTAssertTrue(checked)
        XCTAssertFalse(active)
    }

    func testSecondTokenProbesAndFindsNoHeads() {
        var checked = false
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 2,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { _ in false }
        )
        XCTAssertEqual(result, 1, "should fall back to at least 1 output")
        XCTAssertTrue(checked)
        XCTAssertFalse(active)
    }

    // MARK: - Phase 3a: MTP active, enumerate all slots

    func testEnumeratesAllMtpSlotsWhenActive() {
        var checked = true
        var active = true
        let result = MtpProbe.computeOutputs(
            mtpCount: 4,
            genTokens: 10,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in (Int(idx) >= -4) && (Int(idx) <= -1) }
        )
        XCTAssertEqual(result, 4)
    }

    func testStopsEnumerationAtFirstMissingSlot() {
        var checked = true
        var active = true
        let result = MtpProbe.computeOutputs(
            mtpCount: 5,
            genTokens: 10,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 || idx == -2 || idx == -3 }
        )
        XCTAssertEqual(result, 3)
    }

    func testEnumerationWithSingleSlot() {
        var checked = true
        var active = true
        let result = MtpProbe.computeOutputs(
            mtpCount: 2,
            genTokens: 10,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 }
        )
        XCTAssertEqual(result, 1)
    }

    // MARK: - Phase 3b: MTP unavailable, stay at 1

    func testReturns1WhenMtpUnavailable() {
        var checked = true
        var active = false
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 10,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { _ in
                XCTFail("should not probe when MTP is known unavailable")
                return false
            }
        )
        XCTAssertEqual(result, 1)
    }

    // MARK: - Idempotent: checked stays checked

    func testDoesNotReProbeWhenAlreadyChecked() {
        var checked = true
        var active = true
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 5,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 || idx == -2 }
        )
        XCTAssertEqual(result, 2)
        XCTAssertTrue(checked)
    }

    func testDoesNotReProbeOnSubsequentFirstTokenCall() {
        // Simulates: genTokens counter keeps going but the function is
        // called with checked=true, active=true from previous phase.
        var checked = true
        var active = true
        let result = MtpProbe.computeOutputs(
            mtpCount: 3,
            genTokens: 1,
            mtpAvailabilityChecked: &checked,
            mtpHeadsActive: &active,
            checkLogits: { idx in idx == -1 || idx == -2 }
        )
        // checked=true takes priority over genTokens==1 in phase 3a
        XCTAssertEqual(result, 2)
    }
}
