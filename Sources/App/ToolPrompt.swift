import Foundation

/// How a model wants tool calls expressed on the wire.
///
/// This is the model's *own* chat template's job, not ours to invent. The
/// vendored C API exposes only `llama_chat_apply_template`, whose signature has
/// no `tools` parameter, so llama.cpp's template-driven rendering
/// (`common_chat_templates_apply`, used by `llama-server`) is unavailable to us
/// (`nm` on the framework confirms no `common_chat_*` symbols). We therefore
/// detect the family from the GGUF chat-template string and emulate that
/// template's `tools` branch ourselves — see `ToolPrompt.block`. Injecting the
/// wrong format (we used a JSON envelope while QwOpus/Qwen3.5 expects XML) puts
/// the model out of distribution and it answers with prose instead of a call.
enum ToolWireFormat: Equatable {
    /// Qwen3.5 family (e.g. QwOpus3.5-4B-Coder): schemas live in a `# Tools`
    /// system block and calls are XML —
    /// `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.
    /// Tool results are `<tool_response>…</tool_response>` inside a user turn.
    case qwenXML
    /// Generic JSON/Hermes envelope:
    /// `<tool_call>{"name":…,"arguments":{…}}</tool_call>`. Kept as the fallback
    /// for templates that don't carry the XML grammar.
    case jsonHermes

    /// Picks the format from the model's chat template. Qwen3.5 templates embed
    /// the XML call grammar literally (`<function=…>` / `<parameter=…>`); a
    /// missing or unrelated template falls back to the JSON envelope.
    static func detect(template: String?) -> ToolWireFormat {
        guard let template else { return .jsonHermes }
        return template.contains("<function=") && template.contains("<parameter=")
            ? .qwenXML : .jsonHermes
    }
}

/// Builds the leading system-prompt block that teaches the model the tool
/// protocol, and injects it into a message list.
///
/// Pure on purpose: the exact wording/format is what stands between the model
/// and a turn that narrates a tool call without emitting it, so it is
/// unit-tested (`ToolPromptTests`). The server calls `inject` on every request
/// that offers tools.
enum ToolPrompt {

    /// The instruction for the generic JSON/Hermes envelope. Prepended to the
    /// system message when tools are offered and the model's template carries no
    /// known native grammar.
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

    /// The verbatim `tools`-branch text from the Qwen3.5 chat template: the
    /// format the model was trained on. Reproduced from the GGUF template so the
    /// model sees exactly what `llama-server` would have given it.
    static let qwenFormatInstruction = """
    If you choose to call a function ONLY reply in the following format with NO suffix:

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    <parameter=example_parameter_2>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>
    </tool_call>

    <IMPORTANT>
    Reminder:
    - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
    - Required parameters MUST be specified
    - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
    </IMPORTANT>
    """

    /// The full block prepended to the leading system message, matching the
    /// model template's `tools` branch (schemas + call grammar).
    static func block(toolsJSON: String, toolChoice: ToolChoiceField?, format: ToolWireFormat) -> String {
        switch format {
        case .jsonHermes:
            return systemInstruction(toolChoice: toolChoice) + "\n\nAvailable tools:\n\(toolsJSON)"
        case .qwenXML:
            var text = """
            # Tools

            You have access to the following functions:

            <tools>
            \(toolsJSON)
            </tools>

            \(qwenFormatInstruction)
            """
            if let forced = toolChoice?.forcedFunctionName {
                text += "\n\nYou MUST call the function named \"\(forced)\" and nothing else."
            } else if toolChoice?.requiresTool == true {
                text += "\n\nYou MUST respond with a function call and nothing else."
            }
            return text
        }
    }

    /// Encodes the tool definitions the way the model's template does: the whole
    /// array as JSON for the Hermes envelope, or one JSON object per line inside
    /// `<tools>` for Qwen (matching `{%- for tool in tools %}\n{{ tool | tojson }}`).
    static func encodeTools(_ tools: [ToolDefinition], format: ToolWireFormat) -> String? {
        let encoder = JSONEncoder()
        // `.sortedKeys` matches the template's `function`-before-`type` order;
        // `.withoutEscapingSlashes` matches Python `tojson` (Swift escapes `/`
        // as `\/` by default, which the model never saw in training).
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        switch format {
        case .jsonHermes:
            guard let data = try? encoder.encode(tools) else { return nil }
            return String(data: data, encoding: .utf8)
        case .qwenXML:
            var lines: [String] = []
            for tool in tools {
                guard let data = try? encoder.encode(tool),
                      let line = String(data: data, encoding: .utf8) else { return nil }
                lines.append(line)
            }
            return lines.joined(separator: "\n")
        }
    }

    /// Returns `messages` with the tool block injected — prepended to an existing
    /// system message, or inserted as a new system message at the front. For the
    /// Qwen format, tool-result turns are also rewritten to the `<tool_response>`
    /// user turn the template produces. Returns `messages` unchanged when no
    /// tools are offered or `tool_choice` disables them.
    static func inject(messages: [ChatMessage],
                       tools: [ToolDefinition]?,
                       toolChoice: ToolChoiceField?,
                       format: ToolWireFormat = .jsonHermes) -> [ChatMessage] {
        guard let tools = tools, !tools.isEmpty else { return messages }
        if toolChoice?.disablesTools == true { return messages }
        guard let toolsJSON = encodeTools(tools, format: format) else { return messages }

        let instruction = block(toolsJSON: toolsJSON, toolChoice: toolChoice, format: format)

        var modified = messages
        if let idx = modified.firstIndex(where: { $0.role == "system" }) {
            let existing = modified[idx].content?.textValue ?? ""
            modified[idx] = ChatMessage(role: "system", content: instruction + "\n\n" + existing)
        } else {
            modified.insert(ChatMessage(role: "system", content: instruction), at: 0)
        }

        if format == .qwenXML {
            modified = normalizeToolResults(modified)
        }
        return modified
    }

    /// Rewrites `role: "tool"` turns into the `<tool_response>` user turn the
    /// Qwen template renders (`<|im_start|>user\n<tool_response>\n…\n</tool_response>`).
    /// The generic template has no tool role, so leaving it raw would put the
    /// result somewhere the model never saw during training.
    static func normalizeToolResults(_ messages: [ChatMessage]) -> [ChatMessage] {
        messages.map { message in
            guard message.role == "tool" else { return message }
            let text = message.content?.textValue ?? ""
            return ChatMessage(role: "user", content: "<tool_response>\n\(text)\n</tool_response>")
        }
    }

    /// Renders one assistant `tool_calls` entry into the model's native envelope
    /// for history replay. Qwen emits XML; Hermes emits the JSON object.
    static func toolEnvelope(name: String, argumentsJSON: String, format: ToolWireFormat) -> String {
        switch format {
        case .jsonHermes:
            return "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(argumentsJSON)}</tool_call>"
        case .qwenXML:
            var text = "<tool_call>\n<function=\(name)>\n"
            for (key, value) in orderedArguments(argumentsJSON) {
                text += "<parameter=\(key)>\n\(value)\n</parameter>\n"
            }
            text += "</function>\n</tool_call>"
            return text
        }
    }

    /// Splits a `{"k": …}` arguments string into ordered (key, rendered) pairs
    /// for the Qwen XML envelope. A string value renders bare (the model's own
    /// convention); anything else renders as compact JSON, matching the
    /// template's `tojson`-for-mappings rule.
    static func orderedArguments(_ json: String) -> [(String, String)] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        return obj.keys.sorted().map { key in
            let value = obj[key] as Any
            if let string = value as? String { return (key, string) }
            // `.fragmentsAllowed` is load-bearing, not cosmetic: a scalar value
            // (number, bool, null) is a JSON *fragment*, and without this option
            // `dataWithJSONObject` raises an uncatchable ObjC exception
            // ("Invalid top-level type in JSON write") rather than returning an
            // error — `try?` cannot catch it, so the process aborted on the
            // serial inference queue. A tool call like
            // `{"path":…,"limit":150,"offset":200}` killed the server as soon as
            // the client replayed that assistant turn (device crash
            // 2026-10-18 11:58:53, request #13 after the tool call in #12).
            // Every value here comes from `JSONSerialization.jsonObject`, so with
            // fragments allowed all reachable values serialize.
            if let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]),
               let string = String(data: data, encoding: .utf8) {
                return (key, string)
            }
            return (key, "\(value)")
        }
    }
}
