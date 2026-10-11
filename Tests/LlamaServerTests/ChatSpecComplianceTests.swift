import Foundation
import XCTest

/// Tests for OpenAI-spec response shaping and request validation exposed by
/// llm-comply's `openai-chat` suite.
///
/// Fix A: a chat request with no `model` must be rejected (400). OpenAI treats
/// `model` as required; LlamaServer used to fall through to a 200 completion.
/// `ChatRequestSpec.validationError` is the pure predicate the handler consults.
///
/// Fix B: llm-comply's pinned spec marks `logprobs` (per choice) and
/// `refusal` (per message) as required. LlamaServer omits both, so the content
/// tests failed schema validation. The server must encode them as explicit
/// null — present, but null — without changing how it reads requests.
final class ChatSpecComplianceTests: XCTestCase {

    /// Decodes an encoded value back to a dictionary, or nil on any failure.
    private func asJSONObject(_ value: some Encodable) -> [String: Any]? {
        guard let data = try? JSONEncoder().encode(value),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return obj
    }

    // MARK: Fix A — missing model is a client error

    func testMissingModelIsRejected() {
        XCTAssertNotNil(ChatRequestSpec.validationError(model: nil, messageCount: 1),
                        "a request with no model must be rejected, not completed")
        XCTAssertNotNil(ChatRequestSpec.validationError(model: "", messageCount: 1),
                        "an empty model is as unusable as a missing one")
    }

    func testModelPlusMessagesIsAccepted() {
        XCTAssertNil(ChatRequestSpec.validationError(model: "gpt-x", messageCount: 1))
        XCTAssertNil(ChatRequestSpec.validationError(model: "Qwopus.gguf", messageCount: 3))
    }

    func testMissingMessagesIsRejected() {
        XCTAssertNotNil(ChatRequestSpec.validationError(model: "gpt-x", messageCount: 0),
                        "empty messages is already rejected by the handler; pin it here")
    }

    // MARK: Fix B — response field presence (null, not absent)

    func testAssistantMessageEncodesRefusalAsNull() {
        let obj = asJSONObject(ChatMessage(role: "assistant", content: "hello"))
        XCTAssertNotNil(obj, "assistant message should encode to a JSON object")
        XCTAssertEqual(obj?["refusal"] is NSNull, true,
                       "refusal must be present and null in the encoded message")
    }

    func testChoiceEncodesLogprobsAsNull() {
        let choice = ChatCompletionChoice(index: 0,
                                          message: ChatMessage(role: "assistant", content: "hi"),
                                          finish_reason: "stop")
        let obj = asJSONObject(choice)
        XCTAssertNotNil(obj, "choice should encode to a JSON object")
        XCTAssertEqual(obj?["logprobs"] is NSNull, true,
                       "logprobs must be present and null, since the server computes none")
    }

    func testUserMessageDecodeStillIgnoresUnknownRefusal() {
        // Reading must stay lenient: a client that sends `refusal` on input
        // must not break decoding.
        let json = #"{"role":"user","content":"hi","refusal":"nonsense"}"#
        let msg = try? JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertEqual(msg?.role, "user")
    }

    // MARK: Tool-call turns — content is present but null

    func testToolCallMessageEncodesContentAsNull() {
        // A tool-call assistant turn has no prose, only tool_calls. OpenAI
        // sends `content: null`; llm-comply's pinned spec requires the key be
        // present. Dropping it (encodeIfPresent) failed that check on-device.
        let call = ToolCall(id: "call_1", type: "function",
                            function: ToolCallFunction(name: "get_weather",
                                                       arguments: "{}"))
        let msg = ChatMessage(role: "assistant", content: nil, tool_calls: [call])
        let obj = asJSONObject(msg)
        XCTAssertNotNil(obj, "tool-call message should encode to a JSON object")
        XCTAssertEqual(obj?["content"] is NSNull, true,
                       "content must be present and null on a tool-call turn")
        XCTAssertNotNil(obj?["tool_calls"] as? [Any],
                        "tool_calls must still be encoded")
    }

    func testPlainTextMessageStillEncodesContent() {
        let obj = asJSONObject(ChatMessage(role: "assistant", content: "hi"))
        XCTAssertEqual(obj?["content"] as? String, "hi",
                       "a plain-text turn must still carry its content string")
    }

    func testInboundMessageWithNullContentDecodes() {
        // A client may legitimately send content: null alongside tool calls or
        // tool results; decoding must not throw.
        let json = #"{"role":"assistant","content":null,"tool_calls":[]}"#
        let msg = try? JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertEqual(msg?.role, "assistant")
    }
}
