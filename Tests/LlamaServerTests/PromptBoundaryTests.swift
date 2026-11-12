import XCTest

/// Pins the fast path for locating the final assistant primer in token space.
///
/// The hybrid snapshot must stop before the trailing `<|im_start|>assistant`
/// primer, otherwise it is never a prefix of the next prompt. The old code
/// re-tokenized the entire prompt prefix on every turn just to find that
/// boundary. Tokenizing only the short prompt *tail* and checking that it is
/// aligned with the already-known `promptTokens` gives the same answer without
/// the whole-prompt pass; a mismatch falls back to the old method.
final class PromptBoundaryTests: XCTestCase {

    func testAlignedTailYieldsPrefixBeforePrimer() {
        let prompt: [Int32] = [10, 11, 12, 13, 14]
        let tail: [Int32] = [13, 14]
        XCTAssertEqual(PromptBoundary.stablePrefixLength(promptTokens: prompt,
                                                         tailTokens: tail),
                       3)
    }

    func testMismatchedTailIsRejected() {
        let prompt: [Int32] = [10, 11, 12, 13, 14]
        let tail: [Int32] = [12, 99]
        XCTAssertNil(PromptBoundary.stablePrefixLength(promptTokens: prompt,
                                                       tailTokens: tail),
                     "the caller must fall back to the full-prefix tokenization")
    }

    func testEmptyTailIsRejected() {
        XCTAssertNil(PromptBoundary.stablePrefixLength(promptTokens: [1, 2],
                                                       tailTokens: []))
    }

    func testTailLongerThanPromptIsRejected() {
        XCTAssertNil(PromptBoundary.stablePrefixLength(promptTokens: [1, 2],
                                                       tailTokens: [1, 2, 3]))
    }

    func testWholePromptTailYieldsEmptyPrefix() {
        XCTAssertEqual(PromptBoundary.stablePrefixLength(promptTokens: [1, 2],
                                                         tailTokens: [1, 2]),
                       0)
    }
}
