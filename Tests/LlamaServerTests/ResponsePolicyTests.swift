import XCTest

/// Tests for the pure response-policy decisions in `ResponsePolicy.swift`:
///
/// 1. `ContextHeadroom` — fail fast (with an actionable message) when the
///    prompt leaves no room for a usable response, instead of generating a
///    token or two and reporting a bogus `finish_reason: "length"` truncation.
///    Regression: a 19,927-token prompt in a 20,068-token context produced a
///    141-token truncated response that pi surfaced as a misleading
///    "Response was truncated before completion" banner.
///
/// 2. `FinishReasonPolicy` — map the engine's finish reason honestly:
///    `length` only for genuine max-token cutoffs; `context_full` and
///    unusable (malformed tool-call) output become an explicit `error`.
final class ResponsePolicyTests: XCTestCase {

    // MARK: - ContextHeadroom.check

    func testObservedNearFullPromptIsRejected() {
        let message = ContextHeadroom.check(promptTokens: 19_927, contextSize: 20_068)
        XCTAssertNotNil(message, "prompt leaving only 141 tokens must be rejected")
        XCTAssertTrue(message!.contains("19927"),
                      "message should name the prompt size: \(message!)")
        XCTAssertTrue(message!.contains("20068"),
                      "message should name the context size: \(message!)")
    }

    func testRoomyPromptIsAccepted() {
        XCTAssertNil(ContextHeadroom.check(promptTokens: 1_000, contextSize: 8_192))
    }

    func testHeadroomExactlyAtMinimumIsAccepted() {
        // 20,068 - 19,812 = 256 = default minimum → exactly at the line.
        XCTAssertNil(ContextHeadroom.check(promptTokens: 19_812, contextSize: 20_068))
    }

    func testHeadroomOneBelowMinimumIsRejected() {
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 19_813, contextSize: 20_068))
    }

    func testPromptFillingEntireContextIsRejected() {
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 4_096, contextSize: 4_096))
    }

    func testPromptLargerThanContextIsRejected() {
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 5_000, contextSize: 4_096))
    }

    func testTinyContextRequiresOnlyProportionalHeadroom() {
        // context 256 → required = min(256, 256 / 2) = 128, so a prompt of
        // 128 still passes while 200 (56 tokens left) does not.
        XCTAssertNil(ContextHeadroom.check(promptTokens: 128, contextSize: 256))
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 200, contextSize: 256))
    }

    func testNonPositiveContextSizeIsRejected() {
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 1, contextSize: 0))
    }

    func testCustomMinimumIsHonored() {
        XCTAssertNil(ContextHeadroom.check(promptTokens: 900, contextSize: 1_000,
                                           minimumCompletionTokens: 100))
        XCTAssertNotNil(ContextHeadroom.check(promptTokens: 950, contextSize: 1_000,
                                              minimumCompletionTokens: 100))
    }

    // MARK: - FinishReasonPolicy.map

    func testMaxTokenCutoffMapsToLength() {
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "length", hasToolCalls: false,
                                   outputUnusable: false),
            "length")
    }

    func testContextFullIsNotReportedAsLength() {
        // A context_full engine stop is not a max-token cutoff — it must not
        // masquerade as OpenAI `length` (pi renders that as a truncation
        // banner). Surface it as an explicit error instead.
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "context_full", hasToolCalls: false,
                                   outputUnusable: false),
            "error")
    }

    func testNormalStopsMapToStop() {
        for engine in ["stop", "eog", "eog_immediate", "stopped"] {
            XCTAssertEqual(
                FinishReasonPolicy.map(engineFinish: engine, hasToolCalls: false,
                                       outputUnusable: false),
                "stop", "engine finish '\(engine)' should map to stop")
        }
    }

    func testUnusableOutputAfterNormalStopIsError() {
        // A malformed tool-call envelope the parser couldn't salvage after a
        // normal stop is a model failure, not a truncation — claiming `length`
        // here is what produced pi's misleading truncation reports.
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "stop", hasToolCalls: false,
                                   outputUnusable: true),
            "error")
    }

    func testUnusableOutputAtMaxTokenCutoffKeepsLength() {
        // max_tokens genuinely cut the response mid-envelope: honest `length`.
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "length", hasToolCalls: false,
                                   outputUnusable: true),
            "length")
    }

    func testToolCallsTakePrecedence() {
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "stop", hasToolCalls: true,
                                   outputUnusable: true),
            "tool_calls")
        XCTAssertEqual(
            FinishReasonPolicy.map(engineFinish: "length", hasToolCalls: true,
                                   outputUnusable: false),
            "tool_calls")
    }
}
