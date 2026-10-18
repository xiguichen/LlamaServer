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

    // MARK: - wire format detection

    /// The QwOpus3.5/Qwen3.5 GGUF template carries the XML call grammar; we
    /// must inject that, not the JSON envelope (matched on device: the JSON form
    /// made the model narrate instead of call).
    func testDetectQwenXMLFromTemplate() {
        let template = "<tool_call>\n<function=example_function_name>\n<parameter=x>\n"
        XCTAssertEqual(ToolWireFormat.detect(template: template), .qwenXML)
    }

    func testDetectFallsBackToJSONForNilOrOtherTemplates() {
        XCTAssertEqual(ToolWireFormat.detect(template: nil), .jsonHermes)
        XCTAssertEqual(ToolWireFormat.detect(template: "{% for m in messages %}{{ m }}{% endfor %}"), .jsonHermes)
    }

    // MARK: - native (Qwen XML) block

    func testQwenBlockUsesNativeToolGrammar() {
        let block = ToolPrompt.block(toolsJSON: #"{"function":{"name":"search"}}"#,
                                     toolChoice: nil, format: .qwenXML)
        XCTAssertTrue(block.hasPrefix("# Tools"))
        XCTAssertTrue(block.contains("<tools>"))
        XCTAssertTrue(block.contains(#"{"function":{"name":"search"}}"#))
        XCTAssertTrue(block.contains("<function=example_function_name>"))
        XCTAssertTrue(block.contains("<parameter=example_parameter_1>"))
        XCTAssertFalse(block.contains("\"arguments\""),
                       "the XML format must not advertise a JSON arguments object")
    }

    func testQwenBlockAddsForcedFunctionLine() {
        let block = ToolPrompt.block(toolsJSON: "{}", toolChoice: .function("search"), format: .qwenXML)
        XCTAssertTrue(block.contains("\"search\""))
        XCTAssertTrue(block.contains("MUST call the function"))
    }

    // MARK: - tool_calls replay envelopes

    func testQwenToolEnvelopeRendersXML() {
        let env = ToolPrompt.toolEnvelope(name: "notes",
                                          argumentsJSON: #"{"path":"a.md","op":"write"}"#,
                                          format: .qwenXML)
        XCTAssertEqual(env, """
        <tool_call>
        <function=notes>
        <parameter=op>
        write
        </parameter>
        <parameter=path>
        a.md
        </parameter>
        </function>
        </tool_call>
        """)
    }

    /// A tool call whose arguments hold a top-level scalar — number, bool or
    /// null — must render as compact JSON, the template's `tojson`-for-mappings
    /// rule. These used to reach `JSONSerialization.data(withJSONObject:)` as
    /// JSON *fragments*, which raises an uncatchable ObjC exception ("Invalid
    /// top-level type in JSON write") and aborted the process on the serial
    /// inference queue: a `read` call with `{"path":…,"limit":150,"offset":200}`
    /// killed the server as soon as the client replayed the turn (device crash
    /// 2026-10-18 11:58:53, request #13 following the tool call in #12).
    func testQwenToolEnvelopeRendersScalarArguments() {
        let env = ToolPrompt.toolEnvelope(
            name: "read",
            argumentsJSON: #"{"path":"a.swift","limit":150,"offset":200,"recursive":true,"cursor":null}"#,
            format: .qwenXML)
        XCTAssertEqual(env, """
        <tool_call>
        <function=read>
        <parameter=cursor>
        null
        </parameter>
        <parameter=limit>
        150
        </parameter>
        <parameter=offset>
        200
        </parameter>
        <parameter=path>
        a.swift
        </parameter>
        <parameter=recursive>
        true
        </parameter>
        </function>
        </tool_call>
        """)
    }

    /// `orderedArguments` is the direct entry point that used to abort; keep a
    /// test on it so the fragment handling can't regress silently.
    func testOrderedArgumentsRendersScalarsAsJSON() {
        let pairs = ToolPrompt.orderedArguments(#"{"a":1,"b":1.5,"c":true,"d":null,"e":"s","f":{"k":1}}"#)
            .map { "\($0.0)=\($0.1)" }
        XCTAssertEqual(pairs, ["a=1", "b=1.5", "c=true", "d=null", "e=s", "f={\"k\":1}"])
    }

    func testJSONToolEnvelopeUnchanged() {
        let env = ToolPrompt.toolEnvelope(name: "read", argumentsJSON: #"{"path":"x"}"#, format: .jsonHermes)
        XCTAssertEqual(env, #"<tool_call>{"name": "read", "arguments": {"path":"x"}}</tool_call>"#)
    }

    // MARK: - tool results

    /// The Qwen template renders tool results as a `<tool_response>` user turn;
    /// leaving `role: "tool"` would put the result somewhere the model never saw.
    func testNormalizeToolResultsWrapsInToolResponse() {
        let out = ToolPrompt.normalizeToolResults([
            ChatMessage(role: "assistant", content: "hi"),
            ChatMessage(role: "tool", content: "file contents", tool_call_id: "call_1"),
        ])
        XCTAssertEqual(out[0].role, "assistant")
        XCTAssertEqual(out[1].role, "user")
        XCTAssertEqual(out[1].content?.textValue, "<tool_response>\nfile contents\n</tool_response>")
    }

    // MARK: - inject

    func testInjectInsertsSystemMessageWhenAbsent() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: nil)
        XCTAssertEqual(out.count, 2)   // system + user
        XCTAssertEqual(out.first?.role, "system")
        XCTAssertTrue((out.first?.content?.textValue ?? "").contains("Available tools"))
    }

    func testInjectPrependsToExistingSystemMessage() {
        let msgs = [ChatMessage(role: "system", content: "be terse"),
                    ChatMessage(role: "user", content: "hi")]
        let out = ToolPrompt.inject(messages: msgs, tools: [tool("search")], toolChoice: nil)
        XCTAssertEqual(out.count, 2)
        let sys = out.first?.content?.textValue ?? ""
        XCTAssertTrue(sys.hasPrefix("You have access to the following tools"))
        XCTAssertTrue(sys.hasSuffix("be terse"))
    }

    /// The native format injects the `# Tools` block and must NOT append a
    /// trailing reminder — the earlier reminder was a workaround for the wrong
    /// (JSON) format.
    func testInjectQwenXMLNativeBlockAndNoReminder() {
        let out = ToolPrompt.inject(messages: [ChatMessage(role: "user", content: "hi")],
                                    tools: [tool("search")], toolChoice: nil, format: .qwenXML)
        XCTAssertEqual(out.count, 2)
        XCTAssertTrue((out.first?.content?.textValue ?? "").hasPrefix("# Tools"))
        XCTAssertEqual(out.last?.role, "user")
        XCTAssertEqual(out.last?.content?.textValue, "hi")
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
