import XCTest

/// `QwenPrompt` reproduces the Qwen3.5 chat template's message loop, which the
/// vendored `llama_chat_apply_template` cannot apply (it returns < 0 for this
/// family), so `LlamaInference.formatPrompt` falls back to it.
///
/// The regression these tests pin: a multi-step tool conversation renders each
/// assistant tool-call turn *after* the last user query, and the template wraps
/// those turns in `<think>\n{reasoning}\n</think>\n\n`. Rendering them bare (the
/// old ChatML fallback) left the history disagreeing with the generation prompt,
/// which is itself prefilled with an empty `<think>` block.
final class QwenPromptTests: XCTestCase {

    private func message(_ role: String, _ content: String?, calls: [(String, String)] = []) -> ChatMessage {
        let toolCalls = calls.map { call in
            ToolCall(id: "call_\(call.0)",
                     type: "function",
                     function: ToolCallFunction(name: call.0, arguments: call.1))
        }
        return ChatMessage(role: role, content: content, tool_calls: toolCalls.isEmpty ? nil : toolCalls)
    }

    // MARK: - thinking scaffold

    /// The bug: the assistant tool-call turn that follows the user query must
    /// carry the empty thinking block, exactly as the template emits it.
    func testAssistantTurnAfterLastQueryCarriesThinkingScaffold() {
        let messages = [
            message("user", "hi"),
            message("assistant", "on it"),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\non it<|im_end|>\n"),
                      "post-query assistant turns must carry the empty think block; got:\n\(prompt)")
    }

    /// A plain chat history (assistant turn followed by another user turn) must
    /// NOT get the scaffold: the template only wraps turns after the last query.
    func testAssistantTurnBeforeLastQueryHasNoThinkingScaffold() {
        let messages = [
            message("user", "hi"),
            message("assistant", "hello"),
            message("user", "again"),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\nhello<|im_end|>\n"))
        // Only the generation prompt carries a `<think>` opener; the assistant
        // turn before the last query must not.
        XCTAssertEqual(prompt.components(separatedBy: "<think>").count - 1, 1,
                       "no assistant turn follows the last query")
    }

    /// Tool results are user turns, but the template's `multi_step_tool` scan
    /// screens `<tool_response>` turns out when locating the last real query —
    /// otherwise the tool-call turns before them would lose the scaffold.
    func testToolResponseTurnDoesNotCountAsTheLastQuery() {
        let messages = [
            message("user", "run ls"),
            message("assistant", "Let me check.", calls: [("bash", #"{"command":"ls"}"#)]),
            message("user", "<tool_response>\na.txt\n</tool_response>"),
        ]
        XCTAssertEqual(QwenPrompt.lastQueryIndex(in: messages), 0)
    }

    /// The full shape of one tool round trip, byte for byte.
    func testToolRoundTripRendering() {
        let messages = [
            message("system", "SYS"),
            message("user", "run ls"),
            message("assistant", nil, calls: [("bash", #"{"command":"ls"}"#)]),
            message("user", "<tool_response>\na.txt\n</tool_response>"),
        ]
        let expected = """
        <|im_start|>system
        SYS<|im_end|>
        <|im_start|>user
        run ls<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        <tool_call>
        <function=bash>
        <parameter=command>
        ls
        </parameter>
        </function>
        </tool_call><|im_end|>
        <|im_start|>user
        <tool_response>
        a.txt
        </tool_response><|im_end|>
        <|im_start|>assistant
        <think>

        """
        XCTAssertEqual(QwenPrompt.render(messages: messages), expected)
    }

    /// Prose and a call in the same turn: a blank line separates them (the
    /// template's `content|trim` branch), and the scaffold still leads.
    ///
    /// The prose here ends with newlines because that is what pi replays (the
    /// log-8 assistant turn ended `…systematically.\n\n`). The template trims
    /// the content before appending the envelope, so exactly two newlines must
    /// separate the prose from `<tool_call>` — not four.
    func testProseAndCallInOneTurn() {
        let messages = [
            message("user", "run ls"),
            message("assistant", "Let me check.\n\n", calls: [("bash", #"{"command":"ls"}"#)]),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nLet me check.\n\n<tool_call>\n<function=bash>"),
                      "got:\n\(prompt)")
        XCTAssertFalse(prompt.contains("\n\n\n\n<tool_call>"), "the prose must be trimmed before the envelope")
    }

    // MARK: - generation prompt

    /// The renderer ends with the template's generation prompt *including* the
    /// `<think>` opener; `applyThinkingSwitch` closes it for
    /// `enable_thinking: false`. Both forms must match the template.
    func testGenerationPromptAndThinkingSwitch() {
        let prompt = QwenPrompt.render(messages: [message("user", "hi")])
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n"))
        XCTAssertTrue((prompt + "\n</think>\n\n").hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    // MARK: - other template details

    /// The template merges the first two system messages into one turn.
    func testFirstTwoSystemMessagesAreMerged() {
        let messages = [
            message("system", "# Tools\n\nschemas"),
            message("system", "You are pi."),
            message("user", "hi"),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\n# Tools\n\nschemas\nYou are pi.<|im_end|>\n"),
                      "got:\n\(prompt)")
        XCTAssertEqual(prompt.components(separatedBy: "<|im_start|>system").count - 1, 1)
    }

    /// `|trim` applies to every non-system message.
    func testContentIsTrimmed() {
        let messages = [
            message("user", "  padded\n\n"),
            message("assistant", "\n\npadded too  "),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.contains("<|im_start|>user\npadded<|im_end|>\n"))
        XCTAssertTrue(prompt.contains("</think>\n\npadded too<|im_end|>\n"))
    }

    /// A replayed turn that still carries its reasoning block is peeled apart
    /// and re-emitted in the canonical shape (template lines 97-107).
    func testReasoningBlockInReplayedContentIsReEmitted() {
        let messages = [
            message("user", "hi"),
            message("assistant", "<think>\nreasoning here\n</think>\n\nvisible answer"),
        ]
        let prompt = QwenPrompt.render(messages: messages)
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\n<think>\nreasoning here\n</think>\n\nvisible answer<|im_end|>\n"),
                      "got:\n\(prompt)")
    }

    /// The server pipeline hands `QwenPrompt` messages that `ToolPrompt.inject`
    /// has already prepared: schemas injected into the leading system message
    /// and `tool` turns rewritten to `<tool_response>` user turns.
    func testRenderAfterInjectKeepsToolResultsAndAddsScaffold() {
        let tool = ToolDefinition(type: "function",
                                  function: ToolFunctionDefinition(name: "bash",
                                                                   description: nil,
                                                                   parameters: nil))
        let injected = ToolPrompt.inject(messages: [
            message("system", "You are pi."),
            message("user", "run ls"),
            message("assistant", nil, calls: [("bash", #"{"command":"ls"}"#)]),
            message("tool", "a.txt"),
            message("assistant", "Let me look."),
        ], tools: [tool], toolChoice: nil, format: .qwenXML)

        let prompt = QwenPrompt.render(messages: injected)
        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\n# Tools\n\nYou have access to the following functions:"),
                      "the injected schemas must lead the system turn")
        XCTAssertFalse(prompt.contains("<|im_start|>tool\n"), "tool turns must be rewritten to user turns")
        XCTAssertTrue(prompt.contains("<|im_start|>user\n<tool_response>\na.txt\n</tool_response><|im_end|>\n"))
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=bash>"))
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nLet me look.<|im_end|>\n"))
    }
}
