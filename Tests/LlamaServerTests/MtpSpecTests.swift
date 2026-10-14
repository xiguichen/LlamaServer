import XCTest

final class MtpSpecTests: XCTestCase {

    // MARK: - resolveOutputSlots

    func testResolveOutputSlotsKeepsDepth1Values() {
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: 1, explicitHeads: 0, maxDraft: 1), 2)
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: nil, explicitHeads: 0, maxDraft: 1), 3, "metadata-absent fallback preserved")
    }

    func testResolveOutputSlotsGrowsWithDraftDepth() {
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: 1, explicitHeads: 0, maxDraft: 2), 3)
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: 2, explicitHeads: 0, maxDraft: 2), 3)
    }

    func testResolveOutputSlotsHonoursExplicitHeads() {
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: 1, explicitHeads: 5, maxDraft: 2), 5)
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: nil, explicitHeads: 2, maxDraft: 3), 4, "maxDraft still floors the result")
    }

    func testResolveOutputSlotsClampsBogusGGUFHeads() {
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: 0, explicitHeads: 0, maxDraft: 1), 2)
        XCTAssertEqual(MtpSpec.resolveOutputSlots(headsFromGGUF: -3, explicitHeads: 0, maxDraft: 1), 2)
    }

    // MARK: - draft gate (pMin)

    func testDraftGateIsDisabledByDefaultSoAComputedDraftIsAlwaysVerified() {
        // A draft decode has already paid for its proposal by the time the gate
        // is consulted, so discarding it costs a full plain decode and yields
        // nothing. Upstream's `--spec-draft-p-min` default is 0.00 (never
        // discard); p_min only cuts deeper chain links before they are decoded,
        // which is also why `gateEnabled` must be false so callers never pay
        // for `draftConfidence`'s vocab-wide softmax.
        XCTAssertEqual(MtpSpec.pMin, 0.0, "a computed draft is always worth the verify row")
        XCTAssertFalse(MtpSpec.gateEnabled)
    }

    // MARK: - draftConfidence (softmax over one vocab row)

    func testDraftConfidenceOfUniformLogitsIsOneOverVocab() {
        let logits: [Float] = [0, 0, 0, 0]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.draftConfidence(row: row, token: 2), 0.25, accuracy: 1e-6)
        }
    }

    func testDraftConfidenceOfPeakedLogitsApproachesOne() {
        let logits: [Float] = [0, 50, 0]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.draftConfidence(row: row, token: 1), 1.0, accuracy: 1e-6)
            XCTAssertEqual(MtpSpec.draftConfidence(row: row, token: 0), 0.0, accuracy: 1e-6)
        }
    }

    func testDraftConfidenceIsStableForExtremeLogits() {
        let logits: [Float] = [1e4, -1e4, 1e4]
        logits.withUnsafeBufferPointer { row in
            let p = MtpSpec.draftConfidence(row: row, token: 0)
            XCTAssertEqual(p, 0.5, accuracy: 1e-5)   // max-subtraction must prevent overflow
            XCTAssertFalse(p.isNaN)
        }
    }

    func testDraftConfidenceOfOutOfRangeTokenIsZero() {
        let logits: [Float] = [1, 2, 3]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.draftConfidence(row: row, token: 3), 0)
            XCTAssertEqual(MtpSpec.draftConfidence(row: row, token: -1), 0)
        }
    }

    // MARK: - Greedy draft argmax

    func testArgmaxPicksLargestLogit() {
        let logits: [Float] = [-3, 0.5, 7, 2]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.argmax(row: row), 2)
        }
    }

    func testArgmaxPicksFirstOnTieLikeGreedySampler() {
        let logits: [Float] = [1, 9, 4, 9]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.argmax(row: row), 1)
        }
    }

    func testArgmaxHandlesAllNegativeLogits() {
        let logits: [Float] = [-8, -2, -5]
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.argmax(row: row), 1)
        }
    }

    func testArgmaxOfEmptyRowIsMinusOne() {
        let logits: [Float] = []
        logits.withUnsafeBufferPointer { row in
            XCTAssertEqual(MtpSpec.argmax(row: row), -1)
        }
    }
}
