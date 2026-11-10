import XCTest

/// Tests for the incremental stop-string matcher used by the generation
/// loops.
///
/// The loops append one decoded piece per token and must decide (a) whether a
/// stop string has appeared and (b) how many trailing characters to withhold
/// from the client. Scanning the whole accumulated output for both questions
/// makes a generation quadratic in its own length; `StopScanner` only keeps
/// the tail in which a new match can still complete. These tests pin the
/// answers (absolute cut offset, held suffix) and the bounded retention.
final class StopScannerTests: XCTestCase {

    func testNoStopsNeverMatches() {
        var scanner = StopScanner(stops: [])
        XCTAssertNil(scanner.append("anything at all"))
        XCTAssertEqual(scanner.heldSuffixLength(), 0)
        XCTAssertEqual(scanner.retainedCharacterCount, 0)
    }

    func testStopWithinOnePiece() {
        var scanner = StopScanner(stops: ["STOP"])
        XCTAssertEqual(scanner.append("hello STOP world"), 6)
        XCTAssertEqual(scanner.append("ignored"), nil)
    }

    func testEarliestStopWins() {
        var scanner = StopScanner(stops: ["world", "STOP"])
        // "STOP" starts at 6, "world" later — the earlier cut wins regardless
        // of the order the caller listed the stop strings in.
        XCTAssertEqual(scanner.append("hello STOP world"), 6)
    }

    func testStopSplitAcrossPieces() {
        var scanner = StopScanner(stops: ["STOP"])
        XCTAssertNil(scanner.append("abcST"))
        XCTAssertEqual(scanner.append("OPdef"), 3)
    }

    func testStopStartingInRetainedTailAcrossPieces() {
        var scanner = StopScanner(stops: ["abc"])
        XCTAssertNil(scanner.append("xxa"))
        XCTAssertNil(scanner.append("b"))
        XCTAssertEqual(scanner.append("c"), 2)
    }

    func testCutOffsetIsAbsoluteAfterDroppingTail() {
        var scanner = StopScanner(stops: ["XYZ"])
        var offset = 0
        for _ in 0..<100 {
            XCTAssertNil(scanner.append("abcdefghij"))
            offset += 10
        }
        XCTAssertEqual(scanner.append("!!XYZ"), offset + 2)
    }

    func testHeldSuffixLengthOfPartialStop() {
        var scanner = StopScanner(stops: ["STOP"])
        _ = scanner.append("abcST")
        XCTAssertEqual(scanner.heldSuffixLength(), 2)
        XCTAssertEqual(scanner.heldSuffixLength(), 2, "repeated queries must not mutate the tail")
    }

    func testHeldSuffixLengthZeroWhenNoStopPrefix() {
        var scanner = StopScanner(stops: ["STOP"])
        _ = scanner.append("abc")
        XCTAssertEqual(scanner.heldSuffixLength(), 0)
    }

    func testHeldSuffixLengthUsesLongestPrefix() {
        var scanner = StopScanner(stops: ["ABCDE"])
        _ = scanner.append("xxABC")
        XCTAssertEqual(scanner.heldSuffixLength(), 3)
    }

    func testHeldSuffixLengthHandlesSingleCharacterStop() {
        var scanner = StopScanner(stops: ["\n"])
        _ = scanner.append("hello")
        // A hold of 0 is correct: a one-character stop completes with the very
        // next piece, so nothing needs to be withheld.
        XCTAssertEqual(scanner.heldSuffixLength(), 0)
        XCTAssertEqual(scanner.append("\n"), 5)
    }

    func testRetentionIsBoundedByLongestStop() {
        var scanner = StopScanner(stops: ["STOP"])
        _ = scanner.append("<tool_call>")
        for _ in 0..<500 {
            _ = scanner.append(String(repeating: "x", count: 128))
            XCTAssertLessThanOrEqual(scanner.retainedCharacterCount, 3,
                                     "only the last maxStopLength-1 characters can complete a new match")
        }
    }

    func testEmptyStopStringsAreIgnored() {
        var scanner = StopScanner(stops: ["", "END"])
        XCTAssertNil(scanner.append("no match here"))
        XCTAssertEqual(scanner.append("the END"), 17)
    }
}
