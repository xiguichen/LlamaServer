import XCTest

final class PrefillStatsTests: XCTestCase {

    /// The bug this file pins: after KV reuse the prefill phase decodes only the
    /// prompt suffix, so dividing the *full* prompt length by the phase duration
    /// reported nonsense — a 48,450-token prompt that decoded 113 tokens printed
    /// `prefill=3.14s 15431.54t/s`. The rate must use the decoded count.
    func testRateUsesDecodedTokensNotFullPrompt() {
        let segment = PrefillStats.segment(promptTokens: 48450, reusedTokens: 48337, seconds: 3.14)
        XCTAssertEqual(segment, "prefill=3.14s 35.99t/s (113 of 48450, 48337 reused)")
    }

    /// The device case behind the fix: 35,074-token prompt, 31,767 reused from the
    /// snapshot, 3,307 decoded in 20.70s. Previously reported 1694.15t/s.
    func testReuseRateMatchesDeviceCase() {
        let segment = PrefillStats.segment(promptTokens: 35074, reusedTokens: 31767, seconds: 20.70)
        XCTAssertEqual(segment, "prefill=20.70s 159.76t/s (3307 of 35074, 31767 reused)")
    }

    /// No reuse (cold conversation / unusable snapshot): rate and prompt length agree.
    func testNoReuseReportsWholePrompt() {
        let segment = PrefillStats.segment(promptTokens: 23699, reusedTokens: 0, seconds: 78.0)
        XCTAssertEqual(segment, "prefill=78.00s 303.83t/s (23699 of 23699, 0 reused)")
    }

    /// A zero-length phase must not divide by zero (nan/inf in a log line).
    func testZeroDurationIsZeroRate() {
        let segment = PrefillStats.segment(promptTokens: 464, reusedTokens: 0, seconds: 0)
        XCTAssertEqual(segment, "prefill=0.00s 0.00t/s (464 of 464, 0 reused)")
    }

    /// Clock skew must not produce a negative duration or a negative rate.
    func testNegativeSecondsClampToZero() {
        let segment = PrefillStats.segment(promptTokens: 464, reusedTokens: 0, seconds: -1)
        XCTAssertEqual(segment, "prefill=0.00s 0.00t/s (464 of 464, 0 reused)")
    }

    /// A reuse count larger than the prompt is nonsense input; it must clamp to
    /// "everything reused" rather than report a negative decoded count.
    func testReusedBeyondPromptClampsDecodedToZero() {
        let segment = PrefillStats.segment(promptTokens: 100, reusedTokens: 140, seconds: 1.0)
        XCTAssertEqual(segment, "prefill=1.00s 0.00t/s (0 of 100, 100 reused)")
    }

    /// The reported rate must stay finite for a huge reuse ratio at a tiny duration.
    func testLargeReuseRatioStaysFinite() {
        let segment = PrefillStats.segment(promptTokens: 131072, reusedTokens: 131071, seconds: 0.01)
        XCTAssertEqual(segment, "prefill=0.01s 100.00t/s (1 of 131072, 131071 reused)")
    }
}
