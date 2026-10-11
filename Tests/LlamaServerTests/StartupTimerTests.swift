import Foundation
import XCTest

/// StartupTimer is the pure accumulator behind the startup load-gap
/// instrumentation (#6): it records named phases with durations the caller
/// measured (Date() deltas) and formats a single breakdown line, so the log
/// says *where* the tap-to-listening gap went (model load vs context vs MTP
/// draft vs HTTP server) instead of just "loading…". No clocks and no llama C
/// API live here — the caller injects the seconds — so aggregation, rounding,
/// and formatting are unit-testable.
final class StartupTimerTests: XCTestCase {

    func testRecordsPhasesInOrderWithValues() {
        var t = StartupTimer()
        t.record("model load", seconds: 3.5)
        t.record("context", seconds: 0.25)
        XCTAssertEqual(t.phases.map(\.name), ["model load", "context"])
        XCTAssertEqual(t.phases.map(\.seconds), [3.5, 0.25])
    }

    func testTotalSecondsSumsAllPhases() {
        var t = StartupTimer()
        t.record("a", seconds: 1.0)
        t.record("b", seconds: 2.5)
        t.record("c", seconds: 0.25)
        XCTAssertEqual(t.totalSeconds, 3.75)
    }

    func testSummaryFormatsBreakdownWithTwoDecimalPlaces() {
        var t = StartupTimer()
        t.record("model load", seconds: 3.5)
        t.record("context", seconds: 0.25)
        XCTAssertEqual(t.summary(), "model load 3.50s + context 0.25s = 3.75s")
    }

    func testSummaryRoundsFractionsToTwoDecimals() {
        var t = StartupTimer()
        t.record("x", seconds: 1.237)
        XCTAssertEqual(t.summary(), "x 1.24s = 1.24s")
    }

    func testEmptyTimerReportsZeroAndNoPhases() {
        let t = StartupTimer()
        XCTAssertEqual(t.totalSeconds, 0)
        XCTAssertEqual(t.summary(), "no startup phases recorded")
    }

    func testNegativeDurationIsClampedToZero() {
        // Clock skew could produce a negative delta; it must never surface as a
        // negative phase or a bogus total.
        var t = StartupTimer()
        t.record("x", seconds: -0.5)
        XCTAssertEqual(t.phases.first?.seconds, 0)
        XCTAssertEqual(t.totalSeconds, 0)
    }

    func testSinglePhaseSummary() {
        var t = StartupTimer()
        t.record("engine init", seconds: 4.2)
        XCTAssertEqual(t.summary(), "engine init 4.20s = 4.20s")
    }
}
