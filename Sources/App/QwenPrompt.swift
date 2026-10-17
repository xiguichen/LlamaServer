import Foundation

/// Renders a message list the way the Qwen3.5 chat template does.
///
/// `llama_chat_apply_template` is the primary prompt renderer, but it returns
/// `< 0` for this family's template, so `LlamaInference.formatPrompt` falls back
/// to a hand-rolled renderer. That fallback used to be a plain ChatML layout:
/// it happened to reproduce the template for user and tool turns, but it dropped
/// the **thinking scaffold** the template emits for every assistant message
/// *after the last real user query*:
///
///     <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}
///
/// A multi-step tool conversation is exactly that shape — each assistant
/// tool-call turn follows the user query — so on device the model saw its own
/// previous turns with no `<think>` block while the generation prompt it was
/// continuing *did* carry one (`applyThinkingSwitch` adds it). History and
/// prefill disagreed, which is the same class of out-of-distribution prompt that
/// made the model answer with prose instead of a tool call before the wire
/// format was fixed. This type reproduces the template's message loop exactly.
///
/// Two template branches are intentionally not reproduced here:
/// - the `tools` branch: the schemas are injected into the leading system
///   message by `ToolPrompt.inject` (the vendored C API takes no `tools`), so
///   the template's own `tools` parameter is always absent;
/// - the `tool` role branch: `ToolPrompt.normalizeToolResults` rewrites those
///   turns to `<tool_response>` user turns before rendering, so no `tool` role
///   reaches this type in the server pipeline.
///
/// Pure and unit-tested (`QwenPromptTests`).
enum QwenPrompt {

    /// The generation prompt the template emits with `enable_thinking` undefined
    /// (i.e. thinking on). `applyThinkingSwitch` closes the block when a request
    /// disables thinking.
    static let generationPrompt = "<|im_start|>assistant\n<think>\n"

    /// Renders `messages` into the exact text the Qwen3.5 template would produce
    /// for a request that offers tools (its `tools` branch being handled by
    /// `ToolPrompt.inject` instead).
    ///
    /// `format` only picks the tool-call envelope syntax when replaying an
    /// assistant turn; it defaults to the Qwen XML envelope this type exists for.
    static func render(messages: [ChatMessage], format: ToolWireFormat = .qwenXML) -> String {
        var out = ""

        // The template merges the first two system messages into a single system
        // turn (joined by a newline) and skips them in the main loop.
        var systemCount = 0
        var mergedSystem = ""
        if let first = messages.first, isSystem(first.role) {
            let firstText = trim(text(of: first))
            if messages.count > 1, isSystem(messages[1].role) {
                mergedSystem = firstText + "\n" + trim(text(of: messages[1]))
                systemCount = 2
            } else {
                mergedSystem = firstText
                systemCount = 1
            }
        }
        if !mergedSystem.isEmpty {
            out += "<|im_start|>system\n" + mergedSystem + "<|im_end|>\n"
        }

        let lastQuery = lastQueryIndex(in: messages)
        for (index, message) in messages.enumerated() {
            guard index >= systemCount, !isSystem(message.role) else { continue }
            var content = trim(message.promptText(format: format))

            switch message.role {
            case "user":
                out += "<|im_start|>user\n" + content + "<|im_end|>\n"

            case "assistant":
                // The template peels a `<think>…</think>` block out of the
                // replayed content and re-emits it in the canonical shape.
                var reasoning = ""
                if content.contains("</think>") {
                    let parts = content.components(separatedBy: "</think>")
                    let head = trimNewlinesEnd(parts.first ?? "")
                    reasoning = trimNewlinesStart(lastSegment(of: head, separatedBy: "<think>"))
                    content = trimNewlinesStart(parts.last ?? "")
                }
                if index > lastQuery {
                    out += "<|im_start|>assistant\n<think>\n" + trim(reasoning)
                        + "\n</think>\n\n" + content
                } else {
                    out += "<|im_start|>assistant\n" + content
                }
                out += "<|im_end|>\n"

            default:
                // The template has no branch for other roles; keep the historic
                // fallback layout rather than silently dropping the turn.
                out += "<|im_start|>\(message.role)\n" + content + "<|im_end|>\n"
            }
        }

        out += generationPrompt
        return out
    }

    /// The template's `multi_step_tool` scan, run backwards: the index of the
    /// last `user` message that is not a `<tool_response>` turn. Assistant turns
    /// after it carry the thinking scaffold; everything at or before it does not
    /// (so a plain chat history is still rendered without `<think>` blocks).
    ///
    /// Tool-result turns are already rewritten to `role: "user"` with a
    /// `<tool_response>` wrapper, which is exactly what this scan screens out.
    static func lastQueryIndex(in messages: [ChatMessage]) -> Int {
        var last = messages.count - 1
        for index in stride(from: messages.count - 1, through: 0, by: -1) {
            guard messages[index].role == "user" else { continue }
            let content = trim(text(of: messages[index]))
            if content.hasPrefix("<tool_response>") && content.hasSuffix("</tool_response>") {
                continue
            }
            last = index
            break
        }
        return last
    }

    private static func isSystem(_ role: String) -> Bool {
        role == "system" || role == "developer"
    }

    private static func text(of message: ChatMessage) -> String {
        message.content?.textValue ?? ""
    }

    // MARK: - Jinja string-filter equivalents

    /// `|trim`.
    private static func trim(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Python's `rstrip('\n')` — newlines only, never spaces.
    private static func trimNewlinesEnd(_ value: String) -> String {
        var result = Substring(value)
        while let last = result.last, last == "\n" { result = result.dropLast() }
        return String(result)
    }

    /// Python's `lstrip('\n')`.
    private static func trimNewlinesStart(_ value: String) -> String {
        var result = Substring(value)
        while let first = result.first, first == "\n" { result = result.dropFirst() }
        return String(result)
    }

    /// `value.split(separator)[-1]`.
    private static func lastSegment(of value: String, separatedBy separator: String) -> String {
        value.components(separatedBy: separator).last ?? value
    }
}
