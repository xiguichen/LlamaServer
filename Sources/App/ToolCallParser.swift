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

    /// Fallback for a `<tool_call>` whose closing `</tool_call>` is missing —
    /// common when the model stops (EOG) right after emitting the JSON. Greedy
    /// to the last brace so a complete nested object is captured. Only used when
    /// the strict regex above finds nothing, so it never mis-handles closed tags.
    private static let unclosedToolCallRegex = try! NSRegularExpression(
        pattern: #"<tool_call>\s*(\{.*\})"#,
        options: [.dotMatchesLineSeparators])

    /// A well-formed `<think> … </think>` reasoning block. Models like QwOpus
    /// wrap their chain-of-thought in these tags; clients (e.g. OpenCode) expect
    /// only the final answer as `content`, so the reasoning is stripped out.
    private static let thinkBlockRegex = try! NSRegularExpression(
        pattern: #"<think>.*?</think>"#,
        options: [.dotMatchesLineSeparators])

    /// An unterminated `<think>` (generation cut off mid-reasoning) — strip to end.
    private static let danglingThinkRegex = try! NSRegularExpression(
        pattern: #"<think>.*$"#,
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
            // No closed envelope — try to recover a tool call whose closing tag
            // the model omitted before falling back to plain (reasoning-free) text.
            if let recovered = recoverUnclosedToolCall(in: stripped) {
                return recovered
            }
            return ParseResult(cleanedContent: stripReasoning(stripped), toolCalls: [])
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

        // Remove the envelopes and any <think> reasoning from visible content.
        let cleaned = toolCallRegex.stringByReplacingMatches(
            in: stripped, options: [], range: fullRange, withTemplate: "")
        let trimmed = stripReasoning(cleaned).trimmingCharacters(in: .whitespacesAndNewlines)

        return ParseResult(cleanedContent: trimmed, toolCalls: toolCalls)
    }

    /// Removes `<think> … </think>` reasoning blocks (and any unterminated
    /// trailing `<think>`) from text destined for the client's `content` field.
    static func stripReasoning(_ text: String) -> String {
        var s = text
        var range = NSRange(s.startIndex..<s.endIndex, in: s)
        s = thinkBlockRegex.stringByReplacingMatches(in: s, options: [], range: range, withTemplate: "")
        range = NSRange(s.startIndex..<s.endIndex, in: s)
        s = danglingThinkRegex.stringByReplacingMatches(in: s, options: [], range: range, withTemplate: "")
        return s
    }

    /// Recovers a single tool call from an opening `<tool_call>` that has no
    /// closing tag. Returns nil if no recoverable JSON object is present.
    private static func recoverUnclosedToolCall(in text: String) -> ParseResult? {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = unclosedToolCallRegex.firstMatch(in: text, options: [], range: range),
              match.numberOfRanges >= 2,
              let jsonRange = Range(match.range(at: 1), in: text),
              let call = makeToolCall(fromJSON: String(text[jsonRange])) else {
            return nil
        }
        var cleaned = text
        if let fullMatch = Range(match.range, in: text) {
            cleaned.removeSubrange(fullMatch)
        }
        let trimmed = stripReasoning(cleaned).trimmingCharacters(in: .whitespacesAndNewlines)
        return ParseResult(cleanedContent: trimmed, toolCalls: [call])
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
    ///
    /// Models are inconsistent about the key holding the call's parameters: the
    /// OpenAI/Qwen convention is `arguments`, but some models (e.g. QwOpus) emit
    /// `parameters` instead. Accept either so the arguments aren't silently lost
    /// (which would surface as a tool call with an empty `{}` argument string).
    private static func makeToolCall(fromJSON jsonText: String) -> ToolCall? {
        // The model sometimes appends trailing junk after a complete object
        // (e.g. extra `]}]}` closers) and/or omits the `</tool_call>` tag, so the
        // greedy recovery regex hands us "{…valid…}]}]}". JSONSerialization
        // rejects any trailing content, which would silently drop the tool call.
        // Reduce the text to the first brace-balanced object before parsing.
        let balanced = firstBalancedJSONObject(in: jsonText) ?? jsonText
        guard let data = balanced.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty else {
            return nil
        }

        let argumentsString: String
        if let args = obj["arguments"] ?? obj["parameters"] {
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

    /// Returns the substring of the first complete brace-balanced JSON object in
    /// `text` (from the first `{` to its matching `}`), ignoring any trailing
    /// content. Brace counting respects string literals and `\` escapes so
    /// braces inside string values (e.g. a `write` tool's file content) don't
    /// throw off the balance. Returns nil if no balanced object is found.
    private static func firstBalancedJSONObject(in text: String) -> String? {
        let chars = Array(text)
        guard let start = chars.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var i = start
        while i < chars.count {
            let c = chars[i]
            if inString {
                if escaped {
                    escaped = false
                } else if c == "\\" {
                    escaped = true
                } else if c == "\"" {
                    inString = false
                }
            } else {
                switch c {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        return String(chars[start...i])
                    }
                default: break
                }
            }
            i += 1
        }
        return nil
    }
}

/// Stateful filter that strips `<think> … </think>` reasoning from a *streamed*
/// token sequence, where the tags may be split across token-chunk boundaries.
///
/// The buffered/non-streaming paths use `ToolCallParser.stripReasoning` on the
/// whole text, but a streaming response (e.g. OpenCode's conversation-title
/// request) emits content token-by-token, so without this filter the raw
/// chain-of-thought (and the literal `<think>` tags) would be streamed to the
/// client as visible content. Feed each decoded piece through `feed`; emit only
/// what it returns, then emit `flush()` once generation ends.
final class ReasoningStreamFilter {
    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    private var inThink = false
    /// Holds text that might contain a partial tag straddling the next chunk.
    private var buffer = ""

    /// Consumes a streamed piece, returning the text safe to emit now.
    func feed(_ text: String) -> String {
        buffer += text
        var output = ""
        while true {
            if inThink {
                // Drop everything up to and including the closing tag.
                if let r = buffer.range(of: Self.closeTag) {
                    buffer.removeSubrange(buffer.startIndex..<r.upperBound)
                    inThink = false
                    continue
                }
                // Still inside reasoning — drop all but a tail that could be the
                // start of a split `</think>`.
                buffer = String(buffer.suffix(Self.closeTag.count - 1))
                break
            } else {
                if let r = buffer.range(of: Self.openTag) {
                    output += buffer[buffer.startIndex..<r.lowerBound]
                    buffer.removeSubrange(buffer.startIndex..<r.upperBound)
                    inThink = true
                    continue
                }
                // Emit everything except a possible partial `<think>` at the end.
                let hold = Self.partialTagSuffixLength(of: buffer, tag: Self.openTag)
                let emitCount = buffer.count - hold
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    output += buffer[buffer.startIndex..<idx]
                    buffer.removeSubrange(buffer.startIndex..<idx)
                }
                break
            }
        }
        return output
    }

    /// Emits any remaining buffered text at end of generation. Anything still
    /// inside an unterminated `<think>` is discarded.
    func flush() -> String {
        guard !inThink else { buffer = ""; return "" }
        let out = buffer
        buffer = ""
        return out
    }

    /// Length of the longest suffix of `s` that is a proper prefix of `tag`.
    private static func partialTagSuffixLength(of s: String, tag: String) -> Int {
        var k = min(tag.count - 1, s.count)
        while k > 0 {
            if tag.hasPrefix(String(s.suffix(k))) { return k }
            k -= 1
        }
        return 0
    }
}
