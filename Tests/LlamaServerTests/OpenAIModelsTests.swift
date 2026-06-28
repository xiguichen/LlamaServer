import XCTest

final class OpenAIModelsTests: XCTestCase {

    private let decoder = JSONDecoder()
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    // MARK: - ChatCompletionRequest

    func testDecodeMinimalChatRequest() throws {
        let json = #"{"messages":[{"role":"user","content":"hi"}]}"#.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.messages.count, 1)
        XCTAssertEqual(req.messages[0].role, "user")
        XCTAssertEqual(req.messages[0].content?.textValue, "hi")
        XCTAssertNil(req.stream)
        XCTAssertNil(req.temperature)
        XCTAssertNil(req.max_tokens)
    }

    func testDecodeChatRequestWithAllFields() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "stream": true,
            "stop": ["stop1", "stop2"],
            "seed": 42
        }
        """.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.model, "test-model")
        XCTAssertEqual(req.temperature, 0.7)
        XCTAssertEqual(req.top_p, 0.9)
        XCTAssertEqual(req.resolvedMaxTokens, 512)
        XCTAssertEqual(req.stream, true)
        XCTAssertEqual(req.stop?.values, ["stop1", "stop2"])
        XCTAssertEqual(req.seed, 42)
    }

    func testDecodeChatRequestContentArray() throws {
        let json = """
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}],
                "role": "assistant", "content": "world"}
            ]
        }
        """.data(using: .utf8)!
        // Just verify we can decode without error
        XCTAssertNoThrow(try decoder.decode(ChatCompletionRequest.self, from: json))
    }

    func testResolvedMaxTokensPrecedence() throws {
        // n_predict > max_completion_tokens > max_tokens
        let json1 = #"{"messages":[{"role":"user","content":"hi"}],"n_predict":100,"max_completion_tokens":200,"max_tokens":300}"#.data(using: .utf8)!
        let req1 = try decoder.decode(ChatCompletionRequest.self, from: json1)
        XCTAssertEqual(req1.resolvedMaxTokens, 100)

        let json2 = #"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":200,"max_tokens":300}"#.data(using: .utf8)!
        let req2 = try decoder.decode(ChatCompletionRequest.self, from: json2)
        XCTAssertEqual(req2.resolvedMaxTokens, 200)

        let json3 = #"{"messages":[{"role":"user","content":"hi"}],"max_tokens":300}"#.data(using: .utf8)!
        let req3 = try decoder.decode(ChatCompletionRequest.self, from: json3)
        XCTAssertEqual(req3.resolvedMaxTokens, 300)
    }

    func testDecodeStopAsString() throws {
        let json = #"{"messages":[{"role":"user","content":"hi"}],"stop":"STOP"}"#.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.stop?.values, ["STOP"])
    }

    func testDecodeStopAsArray() throws {
        let json = #"{"messages":[{"role":"user","content":"hi"}],"stop":["A","B","C"]}"#.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.stop?.values, ["A", "B", "C"])
    }

    func testDecodeTools() throws {
        let json = """
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }
        """.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.tools?.count, 1)
        XCTAssertEqual(req.tools?[0].function.name, "get_weather")
    }

    func testDecodeToolChoiceNone() throws {
        let json = #"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"none"}"#.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.tool_choice?.disablesTools, true)
    }

    func testToolChoiceRequired() throws {
        let choice = ToolChoiceField.mode("required")
        XCTAssertTrue(choice.requiresTool)
        XCTAssertFalse(choice.disablesTools)
        XCTAssertNil(choice.forcedFunctionName)
    }

    func testToolChoiceFunction() throws {
        let choice = ToolChoiceField.function("my_tool")
        XCTAssertTrue(choice.requiresTool)
        XCTAssertFalse(choice.disablesTools)
        XCTAssertEqual(choice.forcedFunctionName, "my_tool")
    }

    func testDecodeToolChoiceFunction() throws {
        let json = #"{"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"function","function":{"name":"my_tool"}}}"#.data(using: .utf8)!
        let req = try decoder.decode(ChatCompletionRequest.self, from: json)
        XCTAssertEqual(req.tool_choice?.forcedFunctionName, "my_tool")
    }

    // MARK: - MessageContent

    func testMessageContentText() throws {
        let json = #"{"role":"user","content":"hello"}"#.data(using: .utf8)!
        let msg = try decoder.decode(ChatMessage.self, from: json)
        XCTAssertEqual(msg.content?.textValue, "hello")
    }

    func testMessageContentParts() throws {
        let json = #"{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]}"#.data(using: .utf8)!
        let msg = try decoder.decode(ChatMessage.self, from: json)
        XCTAssertEqual(msg.content?.textValue, "hello\nworld")
    }

    func testMessageWithToolCalls() throws {
        let json = """
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "fn", "arguments": "{}"}}
            ]
        }
        """.data(using: .utf8)!
        let msg = try decoder.decode(ChatMessage.self, from: json)
        XCTAssertNil(msg.content?.textValue)
        XCTAssertEqual(msg.tool_calls?.count, 1)
        XCTAssertEqual(msg.tool_calls?[0].function.name, "fn")
    }

    // MARK: - CompletionRequest

    func testDecodeCompletionRequestString() throws {
        let json = #"{"prompt":"hello world"}"#.data(using: .utf8)!
        let req = try decoder.decode(CompletionRequest.self, from: json)
        XCTAssertEqual(req.prompt.joined, "hello world")
    }

    func testDecodeCompletionRequestArray() throws {
        let json = #"{"prompt":["a","b","c"]}"#.data(using: .utf8)!
        let req = try decoder.decode(CompletionRequest.self, from: json)
        XCTAssertEqual(req.prompt.joined, "a\nb\nc")
    }

    func testDecodeCompletionRequestResolvedMaxTokens() throws {
        let json = #"{"prompt":"hi","n_predict":50,"max_tokens":100}"#.data(using: .utf8)!
        let req = try decoder.decode(CompletionRequest.self, from: json)
        XCTAssertEqual(req.resolvedMaxTokens, 50)
    }

    // MARK: - StreamingChunk

    func testEncodeStreamingChunkIncludesFinishReason() throws {
        let chunk = StreamingChunk(
            id: "chatcmpl-abc", created: 1000, model: "m",
            choices: [StreamingChoice(index: 0, delta: Delta(role: nil, content: "hi"),
                                      finish_reason: nil)])
        let data = try encoder.encode(chunk)
        let json = String(data: data, encoding: .utf8)!
        // nil finish_reason should still appear as null (not omitted)
        XCTAssertTrue(json.contains("\"finish_reason\":null"))
    }

    func testEncodeStreamingChunkFinal() throws {
        let chunk = StreamingChunk(
            id: "chatcmpl-abc", created: 1000, model: "m",
            choices: [StreamingChoice(index: 0, delta: Delta(role: nil, content: nil),
                                      finish_reason: "stop")])
        let data = try encoder.encode(chunk)
        let json = String(data: data, encoding: .utf8)!
        XCTAssertTrue(json.contains("\"finish_reason\":\"stop\""))
    }

    // MARK: - StopField

    func testStopFieldSingleRoundTrip() throws {
        let original = StopField.one("DONE")
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(StopField.self, from: data)
        XCTAssertEqual(decoded.values, ["DONE"])
    }

    func testStopFieldArrayRoundTrip() throws {
        let original = StopField.many(["a", "b"])
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(StopField.self, from: data)
        XCTAssertEqual(decoded.values, ["a", "b"])
    }

    // MARK: - AnyCodable

    func testAnyCodableStringRoundTrip() throws {
        let original = AnyCodable("hello")
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(AnyCodable.self, from: data)
        XCTAssertEqual(decoded.value as? String, "hello")
    }

    func testAnyCodableIntRoundTrip() throws {
        let original = AnyCodable(42)
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(AnyCodable.self, from: data)
        XCTAssertEqual(decoded.value as? Int, 42)
    }

    func testAnyCodableDictRoundTrip() throws {
        let original = AnyCodable(["key": AnyCodable("val")])
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(AnyCodable.self, from: data)
        let dict = decoded.value as? [String: AnyCodable]
        XCTAssertEqual(dict?["key"]?.value as? String, "val")
    }
}
