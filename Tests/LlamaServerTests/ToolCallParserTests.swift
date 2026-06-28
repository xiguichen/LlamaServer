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
}
