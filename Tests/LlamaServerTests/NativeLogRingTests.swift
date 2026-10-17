import XCTest

/// Tests for the bounded buffer of llama.cpp's own log lines.
///
/// Why this exists: the reason a `llama_decode` failed is logged by llama.cpp
/// itself at `.debug`, so a default-level device log showed only the return code
/// (the 2026-10-17 incident needed a verbose log to reveal the GPU
/// command-buffer fault). `recoverFromDecodeFailure()` dumps this tail at error
/// level, so the buffer must hold the lead-up and must not grow without bound.
final class NativeLogRingTests: XCTestCase {

    func testKeepsOnlyTheMostRecentLines() {
        var lines: [String] = []
        for i in 1...20 {
            NativeLogRing.append("line \(i)", to: &lines)
        }
        XCTAssertEqual(lines.count, NativeLogRing.capacity)
        XCTAssertEqual(lines.first, "line 5", "capacity 16 keeps lines 5…20")
        XCTAssertEqual(lines.last, "line 20", "newest stays last so a dump reads in log order")
    }

    func testKeepsEverythingBelowCapacity() {
        var lines: [String] = []
        NativeLogRing.append("only line", to: &lines)
        XCTAssertEqual(lines, ["only line"])
    }

    func testDumpPrefixesEveryLineInOrder() {
        XCTAssertEqual(NativeLogRing.dump(["a", "llama_decode: failed to decode, ret = -3"]),
                       ["  [llama] a", "  [llama] llama_decode: failed to decode, ret = -3"])
    }

    func testDumpOfEmptyBufferIsEmpty() {
        XCTAssertTrue(NativeLogRing.dump([]).isEmpty)
    }
}
