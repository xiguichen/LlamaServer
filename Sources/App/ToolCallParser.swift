import Foundation

/// Parses the `<tool_call>{…}</tool_call>` text our `injectToolDefs` system
/// prompt asks the model to emit, turning it back into the structured
/// `tool_calls` array that OpenAI clients expect.
///
/// Background: the real `llama-server` constrains generation with a GBNF grammar
/// and uses model-specific chat parsers (thousands of lines in common/chat.cpp).
/// That machinery is impractical on-device, so instead we instruct the model via
/// the system prompt and parse the agreed-upon `<tool_call>` envelope back out
/// here. The *observable* OpenAI contract (a `tool_calls` array on the message
/// plus `finish_reason: "tool_calls"`) is what clients depend on, and that is
/// exactly what this reproduces.
enum ToolCallParser {

    /// Result of scanning a completed model response.
    struct ParseResult {
        /// Text with `<tool_call>` envelopes removed and ANSI escapes stripped.
        let cleanedContent: String
        /// Structured tool calls extracted from the response (may be empty).
        let toolCalls: [ToolCall]
    }

    /// Matches a `<tool_call> … </tool_call>` block, capturing the inner JSON.
    /// `.dotMatchesLineSeparators` lets the JSON span multiple lines.
    private static let toolCallRegex = try! NSRegularExpression(
        pattern: #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#,
        options: [.dotMatchesLineSeparators])

    /// ANSI SGR escape sequences (e.g. 24-bit color codes) some models emit.
    private static let ansiRegex = try! NSRegularExpression(
        pattern: "\u{1B}\\[[0-9;]*m",
        options: [])

    /// Scans a finished completion for tool-call envelopes.
    static func parse(_ text: String) -> ParseResult {
        let stripped = stripANSI(text)

        let fullRange = NSRange(stripped.startIndex..<stripped.endIndex, in: stripped)
        let matches = toolCallRegex.matches(in: stripped, options: [], range: fullRange)

        guard !matches.isEmpty else {
            return ParseResult(cleanedContent: stripped, toolCalls: [])
        }

        var toolCalls: [ToolCall] = []
        for match in matches {
            guard match.numberOfRanges >= 2,
                  let jsonRange = Range(match.range(at: 1), in: stripped) else { continue }
            let jsonText = String(stripped[jsonRange])
            if let call = makeToolCall(fromJSON: jsonText) {
                toolCalls.append(call)
            }
        }

        // Remove the envelopes from the human-visible content.
        let cleaned = toolCallRegex.stringByReplacingMatches(
            in: stripped, options: [], range: fullRange, withTemplate: "")
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(cleanedContent: trimmed, toolCalls: toolCalls)
    }

    /// Removes ANSI escape sequences from arbitrary model output.
    static func stripANSI(_ text: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return ansiRegex.stringByReplacingMatches(
            in: text, options: [], range: range, withTemplate: "")
    }

    /// Builds a `ToolCall` from a `{"name": …, "arguments": {…}}` JSON object.
    /// OpenAI's `function.arguments` field is a JSON *string*, so the parsed
    /// arguments object is re-serialized to a compact string.
    private static func makeToolCall(fromJSON jsonText: String) -> ToolCall? {
        guard let data = jsonText.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty else {
            return nil
        }

        let argumentsString: String
        if let args = obj["arguments"] {
            if let argsString = args as? String {
                // Model already gave a string — pass it through verbatim.
                argumentsString = argsString
            } else if JSONSerialization.isValidJSONObject(args),
                      let argsData = try? JSONSerialization.data(withJSONObject: args),
                      let s = String(data: argsData, encoding: .utf8) {
                argumentsString = s
            } else {
                argumentsString = "{}"
            }
        } else {
            argumentsString = "{}"
        }

        return ToolCall(
            id: "call_\(UUID().uuidString.prefix(24))",
            type: "function",
            function: ToolCallFunction(name: name, arguments: argumentsString))
    }
}
