import XCTest

final class ToolPromptTests: XCTestCase {

    private func tool(_ name: String) -> ToolDefinition {
        ToolDefinition(type: "function",
                       function: ToolFunctionDefinition(name: name, description: nil, parameters: nil))
    }

    // MARK: - systemInstruction

    func testInstructionDocumentsEnvelopeFormat() {
        let text = ToolPrompt.systemInstruction(toolChoice: nil)
        XCTAssertTrue(text.contains("<tool_call>"), "the model must be shown the envelope format")
        XCTAssertTrue(text.contains("</tool_call>"))
    }

    /// Observed on device (a fresh session): the model answered "I'm going to
    /// ... let me use the X tool" and ended the turn with no `<tool_call>`. The
    /// instruction must tell it that announcing a call does nothing and that
    /// the envelope has to be emitted in the same message.
    func testInstructionForbidsNarratingWithoutCalling() {
        let text = ToolPrompt.systemInstruction(toolChoice: nil)
        XCTAssertTrue(text.contains("NEVER announce or describe a tool call without emitting it"),
                      "the anti-narration directive must survive; without it the model answers with prose and stops")
        XCTAssertTrue(text.contains("emit the <tool_call> block in that same message"))
    }

    func testInstructionAddsForcedFunctionLine() {
        let text = ToolPrompt.systemInstruction(toolChoice: .function("search"))
        XCTAssertTrue(text.contains("\"search\""))
        XCTAssertTrue(text.contains("MUST call the function"))
    }

    func testInstructionAddsRequiredLine() {
        let text = ToolPrompt.systemInstruction(toolChoice: .mode("required"))
        XCTAssertTrue(text.contains("MUST respond with a <tool_call>"))
    }

    func testInstructionAutoHasNoForcedLine() {
        let text = ToolPrompt.systemInstruction(toolChoice: .mode("auto"))
        XCTAssertFalse(text.contains("MUST call the function"))
        XCTAssertFalse(text.contains("MUST respond with a <tool_call>"))
    }

    // MARK: - inject

    func testInjectInsertsSystemMessageWhenAbsent() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: nil)
        XCTAssertEqual(out.count, 3)   // system + user + trailing reminder
        XCTAssertEqual(out.first?.role, "system")
        XCTAssertTrue((out.first?.content?.textValue ?? "").contains("Available tools"))
    }

    func testInjectPrependsToExistingSystemMessage() {
        let msgs = [ChatMessage(role: "system", content: "be terse"),
                    ChatMessage(role: "user", content: "hi")]
        let out = ToolPrompt.inject(messages: msgs, tools: [tool("search")], toolChoice: nil)
        XCTAssertEqual(out.count, 3)
        let sys = out.first?.content?.textValue ?? ""
        XCTAssertTrue(sys.hasPrefix("You have access to the following tools"))
        XCTAssertTrue(sys.hasSuffix("be terse"))
    }

    /// Observed on device: pi prepends a Superpowers user message that tells the
    /// model to "announce Using [skill] to [purpose]". A small model announces
    /// and then stops, never calling a tool — the tool instruction at the top of
    /// the system message is too far from where generation starts. Repeating a
    /// short reminder AFTER every user/tool message restores the call.
    func testInjectAppendsReminderAfterAllMessages() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: nil)
        XCTAssertEqual(out.count, 3, "system + user + trailing reminder")
        XCTAssertEqual(out.last?.role, "user")
        let reminder = out.last?.content?.textValue ?? ""
        XCTAssertTrue(reminder.contains("<tool_call>"))
        XCTAssertTrue(reminder.contains("If a tool is needed"))
    }

    /// The reminder must not push the model into calling a tool when the answer
    /// needs none (verified on device: "What is 2+2?" answers "4").
    func testReminderAllowsAnsweringWithoutATool() {
        XCTAssertTrue(ToolPrompt.reminder.contains("If no tool is needed"))
    }

    func testInjectLeavesMessagesUnchangedWithoutTools() {
        let msgs = [ChatMessage(role: "user", content: "hi")]
        XCTAssertEqual(ToolPrompt.inject(messages: msgs, tools: nil, toolChoice: nil).count, 1)
        XCTAssertEqual(ToolPrompt.inject(messages: msgs, tools: [], toolChoice: nil).count, 1)
    }

    func testInjectLeavesMessagesUnchangedWhenToolChoiceNone() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: .mode("none"))
        XCTAssertEqual(out.count, 1)
    }

    func testInjectIncludesSerializedToolSchema() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: nil)
        XCTAssertTrue((out.first?.content?.textValue ?? "").contains("\"name\":\"search\""))
    }
}
