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
}
