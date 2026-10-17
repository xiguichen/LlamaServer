import Foundation

/// Builds the system-prompt instruction that teaches the model the
/// `<tool_call>` envelope, and injects it into a message list.
///
/// Pure on purpose: the exact wording is the only thing standing between the
/// model and a turn that narrates a tool call without emitting it, so it is
/// unit-tested (`ToolPromptTests`). The server calls `inject` on every request
/// that offers tools.
enum ToolPrompt {
    /// The instruction prepended to the system message when tools are offered.
    static func systemInstruction(toolChoice: ToolChoiceField?) -> String {
        var instruction = """
        You have access to the following tools. When you want to use a tool, respond with EXACTLY this format (no markdown, no extra text around it):

        <tool_call>{"name": "tool_name", "arguments": {"param1": "val1", "param2": "val2"}}</tool_call>

        IMPORTANT: The `arguments` object MUST include ALL required parameters listed in the tool's JSON schema. Never omit a required parameter.

        NEVER announce or describe a tool call without emitting it. Saying "let me use the X tool" and then stopping does nothing — if you decide to use a tool, emit the <tool_call> block in that same message.
        """

        if let forced = toolChoice?.forcedFunctionName {
            instruction += "\n\nYou MUST call the function named \"\(forced)\" with a <tool_call> and nothing else."
        } else if toolChoice?.requiresTool == true {
            instruction += "\n\nYou MUST respond with a <tool_call> and nothing else."
        }

        return instruction
    }

    /// A short reminder appended AFTER every user/tool message. Observed on
    /// device: pi prepends a Superpowers user message telling the model to
    /// "announce Using [skill] to [purpose]"; a small model (4B) announces and
    /// then ends the turn with EOG, never emitting a tool call. The tool
    /// instruction at the top of the system message is too far from where
    /// generation starts, so the requirement is repeated next to it. It is
    /// conditional ("if a tool is needed") so a question that needs no tool is
    /// still answered directly (verified: "What is 2+2?" → "4").
    static let reminder = "If a tool is needed to proceed, you MUST emit the <tool_call> block in this message — do not reply with only a sentence announcing what you will do. If no tool is needed, answer normally."

    /// Returns `messages` with the tool instruction injected — prepended to an
    /// existing system message, or inserted as a new system message at the
    /// front — and a short `reminder` appended as the last message so it sits
    /// next to the generation point. Returns `messages` unchanged when no tools
    /// are offered or `tool_choice` disables them.
    static func inject(messages: [ChatMessage],
                       tools: [ToolDefinition]?,
                       toolChoice: ToolChoiceField?) -> [ChatMessage] {
        guard let tools = tools, !tools.isEmpty else { return messages }
        if toolChoice?.disablesTools == true { return messages }

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let toolsData = try? encoder.encode(tools),
              let toolsStr = String(data: toolsData, encoding: .utf8) else {
            return messages
        }

        let instruction = systemInstruction(toolChoice: toolChoice)
            + "\n\nAvailable tools:\n\(toolsStr)"

        var modified = messages
        if let idx = modified.firstIndex(where: { $0.role == "system" }) {
            let existing = modified[idx].content?.textValue ?? ""
            modified[idx] = ChatMessage(role: "system", content: instruction + "\n\n" + existing)
        } else {
            modified.insert(ChatMessage(role: "system", content: instruction), at: 0)
        }
        modified.append(ChatMessage(role: "user", content: reminder))
        return modified
    }
}
