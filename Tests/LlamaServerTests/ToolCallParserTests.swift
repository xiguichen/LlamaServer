import XCTest

final class ToolCallParserTests: XCTestCase {

    // MARK: - stripANSI

    func testStripANSINoMatches() {
        XCTAssertEqual(ToolCallParser.stripANSI("hello world"), "hello world")
    }

    func testStripANSISimple() {
        XCTAssertEqual(ToolCallParser.stripANSI("\u{1B}[31mred\u{1B}[0m"), "red")
    }

    func testStripANSI24Bit() {
        XCTAssertEqual(ToolCallParser.stripANSI("\u{1B}[38;2;255;0;0mhi\u{1B}[0m"), "hi")
    }

    func testStripANSIMultiple() {
        XCTAssertEqual(ToolCallParser.stripANSI("\u{1B}[1m\u{1B}[32mbold green\u{1B}[0m"), "bold green")
    }

    func testStripANSIEmpty() {
        XCTAssertEqual(ToolCallParser.stripANSI(""), "")
    }

    // MARK: - stripReasoning

    func testStripReasoningNoThink() {
        let text = "Hello, world!"
        XCTAssertEqual(ToolCallParser.stripReasoning(text), text)
    }

    func testStripReasoningFullBlock() {
        let text = "Before <think>some reasoning</think> After"
        XCTAssertEqual(ToolCallParser.stripReasoning(text), "Before  After")
    }

    func testStripReasoningDangling() {
        let text = "Before <think>unfinished reasoning"
        XCTAssertEqual(ToolCallParser.stripReasoning(text), "Before ")
    }

    func testStripReasoningMultiple() {
        let text = "A<think>r1</think>B<think>r2</think>C"
        XCTAssertEqual(ToolCallParser.stripReasoning(text), "ABC")
    }

    func testStripReasoningMultiline() {
        let text = "A<think>\nline1\nline2\n</think>B"
        XCTAssertEqual(ToolCallParser.stripReasoning(text), "AB")
    }

    // MARK: - firstBalancedJSONObject

    func testBalancedSimple() {
        XCTAssertEqual(ToolCallParser.firstBalancedJSONObject(in: #"{"a":1}"#), #"{"a":1}"#)
    }

    func testBalancedNested() {
        let input = #"{"outer":{"inner":42}}"#
        XCTAssertEqual(ToolCallParser.firstBalancedJSONObject(in: input), input)
    }

    func testBalancedWithBracesInString() {
        let input = #"{"content":"{braces} inside"}"#
        XCTAssertEqual(ToolCallParser.firstBalancedJSONObject(in: input), input)
    }

    func testBalancedWithTrailingJunk() {
        let result = ToolCallParser.firstBalancedJSONObject(in: #"{"valid":true}]}]}"#)
        XCTAssertEqual(result, #"{"valid":true}"#)
    }

    func testBalancedNoBrace() {
        XCTAssertNil(ToolCallParser.firstBalancedJSONObject(in: "no braces here"))
    }

    func testBalancedUnclosed() {
        XCTAssertNil(ToolCallParser.firstBalancedJSONObject(in: #"{"unclosed":"forever"#))
    }

    func testBalancedEscapedQuote() {
        let input = #"{"key":"value with \"quote\" inside"}"#
        XCTAssertEqual(ToolCallParser.firstBalancedJSONObject(in: input), input)
    }

    // MARK: - parse (full tool call)

    func testParseSimpleToolCall() {
        let text = #"Hello<tool_call>{"name":"get_weather","arguments":{"city":"London"}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.cleanedContent, "Hello")
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.name, "get_weather")
    }

    func testParseMultipleToolCalls() {
        let text = #"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 2)
        XCTAssertEqual(result.toolCalls[0].function.name, "a")
        XCTAssertEqual(result.toolCalls[1].function.name, "b")
    }

    func testParseUnclosedToolCall() {
        let text = #"Some text<tool_call>{"name":"fn","arguments":{"x":1}}"#
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.name, "fn")
        XCTAssertEqual(result.cleanedContent, "Some text")
    }

    func testParseStripsReasoningInside() {
        let text = "<think>hidden</think> visible <tool_call>{\"name\":\"fn\",\"arguments\":{}}</tool_call>"
        let result = ToolCallParser.parse(text)
        XCTAssertFalse(result.cleanedContent.contains("hidden"))
        XCTAssertTrue(result.cleanedContent.contains("visible"))
    }

    func testParseNoToolCalls() {
        let result = ToolCallParser.parse("just text")
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertEqual(result.cleanedContent, "just text")
    }

    func testParseStripsANSI() {
        let esc = "\u{1B}"
        let text = "hello\(esc)[31m world<tool_call>{\"name\":\"fn\",\"arguments\":{\"x\":\"\(esc)[32mgreen\(esc)[0m\"}}</tool_call>"
        let result = ToolCallParser.parse(text)
        XCTAssertTrue(result.toolCalls.count == 1, "should recover tool call even with ANSI in args")
    }

    // MARK: - parse (name hardening / no envelope fragments)

    func testParseWhitespaceOnlyNameYieldsNoToolCall() {
        let text = #"Hello<tool_call>{"name":"   ","arguments":{}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertTrue(result.toolCalls.isEmpty,
                      "whitespace-only name must not become a tool call")
        XCTAssertEqual(result.cleanedContent, "Hello")
    }

    func testParseEmptyNameYieldsNoToolCall() {
        let text = #"Hello<tool_call>{"name":"","arguments":{}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertEqual(result.cleanedContent, "Hello")
    }

    func testParseTrimsToolName() {
        let text = #"<tool_call>{"name":"  fn  ","arguments":{}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.name, "fn")
    }

    func testParseStripsEnvelopeFragmentsFromContent() {
        let text = "Hello</arg>"
        let result = ToolCallParser.parse(text)
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertFalse(result.cleanedContent.contains("</arg"),
                       "envelope fragments must never reach the client as content")
        XCTAssertTrue(result.hasUnrecoveredEnvelope,
                      "an envelope the parser dropped must be reported so the server finishes with an explicit error")
    }

    func testParsePlainTextReportsNoUnrecoveredEnvelope() {
        let result = ToolCallParser.parse("just text")
        XCTAssertFalse(result.hasUnrecoveredEnvelope)
    }

    func testParseSuccessfulToolCallReportsNoUnrecoveredEnvelope() {
        let text = #"Calling it now.<tool_call>{"name":"fn","arguments":{}}</tool_call>"#
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertFalse(result.hasUnrecoveredEnvelope)
    }

    // MARK: - parse (Qwen3.5 native XML)

    private let xmlCall = """
    <tool_call>
    <function=notes>
    <parameter=operation>
    write
    </parameter>
    <parameter=path>
    checkpoint.md
    </parameter>
    </function>
    </tool_call>
    """

    /// The model's own template emits `<function=…>/<parameter=…>`; before this
    /// the parser only matched the JSON envelope and silently dropped it.
    func testParseQwenXMLToolCall() {
        let result = ToolCallParser.parse(xmlCall)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.name, "notes")
        XCTAssertEqual(result.cleanedContent, "")
        XCTAssertFalse(result.hasUnrecoveredEnvelope)
    }

    func testParseQwenXMLArgumentsAreJSONStrings() throws {
        let result = ToolCallParser.parse(xmlCall)
        let args = try XCTUnwrap(result.toolCalls.first?.function.arguments)
        let decoded = try JSONSerialization.jsonObject(with: Data(args.utf8)) as? [String: String]
        XCTAssertEqual(decoded?["operation"], "write")
        XCTAssertEqual(decoded?["path"], "checkpoint.md")
    }

    func testParseQwenXMLWithProsePreamble() {
        let text = "I'll save that.\n\n" + xmlCall
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.cleanedContent, "I'll save that.")
    }

    func testParseQwenXMLMultipleCalls() {
        let second = xmlCall.replacingOccurrences(of: "notes", with: "read")
        let result = ToolCallParser.parse(xmlCall + "\n" + second)
        XCTAssertEqual(result.toolCalls.count, 2)
        XCTAssertEqual(result.toolCalls[0].function.name, "notes")
        XCTAssertEqual(result.toolCalls[1].function.name, "read")
    }

    /// The model sometimes stops right after `</function>` (EOG) — recover it.
    func testParseQwenXMLUnclosed() {
        let text = "Go.<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>"
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.name, "bash")
        XCTAssertEqual(result.cleanedContent, "Go.")
    }

    func testParseQwenXMLNoParametersYieldsEmptyArguments() {
        let text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        let result = ToolCallParser.parse(text)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].function.arguments, "{}")
    }

    // MARK: - ReasoningStreamFilter

    func testReasoningFilterPassesThroughNormalText() {
        let filter = ReasoningStreamFilter()
        XCTAssertEqual(filter.feed("Hello"), "Hello")
        XCTAssertEqual(filter.feed(" World"), " World")
        XCTAssertEqual(filter.flush(), "")
    }

    func testReasoningFilterStripsThinkBlock() {
        let filter = ReasoningStreamFilter()
        XCTAssertEqual(filter.feed("Before "), "Before ")
        XCTAssertEqual(filter.feed("<think>"), "")
        XCTAssertEqual(filter.feed("reasoning"), "")
        XCTAssertEqual(filter.feed("</think>"), "")
        XCTAssertEqual(filter.feed(" After"), " After")
        XCTAssertEqual(filter.flush(), "")
    }

    func testReasoningFilterSplitTags() {
        let filter = ReasoningStreamFilter()
        XCTAssertEqual(filter.feed("A<thi"), "A")
        XCTAssertEqual(filter.feed("nk>B"), "")
        XCTAssertEqual(filter.feed("C</th"), "")
        XCTAssertEqual(filter.feed("ink>D"), "D")
    }

    func testReasoningFilterDropUnclosedThink() {
        let filter = ReasoningStreamFilter()
        XCTAssertEqual(filter.feed("A<think>B"), "A")
        XCTAssertEqual(filter.flush(), "") // unterminated think → drop
    }

    func testReasoningFilterMultipleBlocks() {
        let filter = ReasoningStreamFilter()
        XCTAssertEqual(filter.feed("<think>r1</think>ok<think>r2</think>end"), "okend")
    }

    // MARK: - ToolStreamFilter

    func testToolStreamFilterPassesPlainTextThrough() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(filter.feed("Let me check "), "Let me check ")
        XCTAssertEqual(filter.feed("the metrics"), "the metrics")
    }

    func testToolStreamFilterBuffersEnvelope() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(filter.feed(#"Checking now.<tool_call>{"name":"fn","arguments":{}}</tool_call>"#),
                       "Checking now.")
        // The envelope body is never streamed as content.
    }

    func testToolStreamFilterEnvelopeSplitAcrossChunks() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(filter.feed("before<tool_call>"), "before")
        XCTAssertEqual(filter.feed(#"{"name":"fn","arguments":{"x":1}}"#), "")
        XCTAssertEqual(filter.feed("</tool_call>"), "")
        XCTAssertEqual(filter.feed("after"), "after")
    }

    func testToolStreamFilterKeepsProseAfterEnvelopeInSameChunk() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(
            filter.feed(#"pre<tool_call>{"name":"fn"}</tool_call>post"#),
            "prepost")
    }

    func testToolStreamFilterMultipleEnvelopes() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(
            filter.feed(#"<tool_call>{"name":"a"}</tool_call>mid<tool_call>{"name":"b"}</tool_call>end"#),
            "midend")
    }

    func testToolStreamFilterDropUnclosedEnvelope() {
        let filter = ToolStreamFilter()
        XCTAssertEqual(filter.feed(#"text<tool_call>{"name":"fn""#), "text")
        // Unterminated envelope stays buffered (never streamed as content).
    }

    // MARK: - ReasoningStreamFilter + ToolStreamFilter composition (tool turns)

    /// Regression: a tool turn must not stream a `<think>` block to the client.
    /// The server runs the decoded text through the reasoning filter BEFORE the
    /// tool filter; previously the tool path bypassed the reasoning filter
    /// entirely, leaking the chain-of-thought as `content` and suppressing the
    /// cleaned content that would otherwise have replaced it.
    func testToolTurnStripsThinkAndKeepsProseAndEnvelope() {
        let think = ReasoningStreamFilter()
        let tool = ToolStreamFilter()
        var emitted = ""
        for chunk in ["<think>let me", " reason", "</think>Calling ", #"it.<tool_call>{"name":"fn"}</tool_call>"#] {
            emitted += tool.feed(think.feed(chunk))
        }
        XCTAssertEqual(emitted, "Calling it.")
        XCTAssertFalse(emitted.contains("think"))
        XCTAssertFalse(emitted.contains("reason"))
    }
}
