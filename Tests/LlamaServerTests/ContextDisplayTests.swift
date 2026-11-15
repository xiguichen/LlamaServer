import XCTest

/// The Context size field must not claim a window the engine did not get.
///
/// The 2026-11-15 log showed `requested ctx 262144 ... reducing to 48640` while
/// the UI kept showing 262144 (and the field is disabled while running, so it
/// could not be corrected). These tests pin what the field and its note show.
final class ContextDisplayTests: XCTestCase {

    func testFieldShowsEffectiveSizeWhileRunning() {
        XCTAssertEqual(ContextDisplay.fieldText(requested: "262144",
                                                effective: 48_640,
                                                isRunning: true),
                       "48640")
    }

    func testFieldShowsTheRequestAgainWhenStopped() {
        XCTAssertEqual(ContextDisplay.fieldText(requested: "262144",
                                                effective: 48_640,
                                                isRunning: false),
                       "262144",
                       "the field is editable when stopped; the request is the next Start's input")
    }

    func testFieldKeepsTheRequestWhenNothingIsLoaded() {
        XCTAssertEqual(ContextDisplay.fieldText(requested: "8192",
                                                effective: nil,
                                                isRunning: true),
                       "8192")
    }

    func testMemoryClampNoteNamesTheBudget() {
        XCTAssertEqual(ContextDisplay.limitNote(requested: 262_144,
                                                effective: 48_640,
                                                clamp: .memoryBudget(cap: 48_640),
                                                isRunning: true),
                       "Using 48640 tokens — device memory budget (requested 262144).")
    }

    func testModelWindowClampNotePointsAtLongContext() {
        XCTAssertEqual(ContextDisplay.limitNote(requested: 262_144,
                                                effective: 32_768,
                                                clamp: .modelWindow(cap: 32_768),
                                                isRunning: true),
                       "Using 32768 tokens — the model's trained window (requested 262144; enable long context for more).")
    }

    func testNoNoteWhenNotReduced() {
        XCTAssertNil(ContextDisplay.limitNote(requested: 8192,
                                              effective: 8192,
                                              clamp: .notReduced,
                                              isRunning: true))
    }

    func testNoNoteWhenStoppedOrUnloaded() {
        XCTAssertNil(ContextDisplay.limitNote(requested: 262_144,
                                              effective: 48_640,
                                              clamp: .memoryBudget(cap: 48_640),
                                              isRunning: false))
        XCTAssertNil(ContextDisplay.limitNote(requested: 262_144,
                                              effective: nil,
                                              clamp: .memoryBudget(cap: 48_640),
                                              isRunning: true))
    }
}
