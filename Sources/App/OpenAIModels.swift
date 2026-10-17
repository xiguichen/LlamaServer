import Foundation

// Minimal OpenAI-compatible request/response shapes for /v1/chat/completions
// and /v1/models. These are intentionally lenient on input.

/// Build identifier echoed back in responses (OpenAI `system_fingerprint`).
/// llama-server uses its build string here; clients only echo it for telemetry.
let openAISystemFingerprint = "llamaserver"

struct ToolCallFunction: Codable {
    let name: String
    let arguments: String
}

struct ToolCall: Codable {
    let id: String
    let type: String
    let function: ToolCallFunction
}

struct ChatMessage: Codable {
    let role: String
    let content: MessageContent?
    let tool_calls: [ToolCall]?
    let tool_call_id: String?
    /// OpenAI marks `refusal` required on assistant messages. We never refuse,
    /// so we encode it as explicit null — present, but null — to satisfy strict
    /// clients without fabricating a value. `nil` on decode (we ignore it).
    let refusal: String?

    enum CodingKeys: String, CodingKey {
        case role, content, tool_calls, tool_call_id, refusal
    }

    init(role: String, content: String?, tool_calls: [ToolCall]? = nil, tool_call_id: String? = nil) {
        self.role = role
        self.content = content.map { .text($0) }
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.refusal = nil
    }

    // Encode `content`/`refusal` as null when absent (Swift would otherwise
    // omit a nil optional); decode ignores unknown fields so inbound requests
    // stay lenient. Explicit-null `content` matters on tool-call turns: OpenAI
    // sends `content: null` alongside tool_calls, and strict clients require
    // the key be present.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        try c.encode(content, forKey: .content)
        try c.encodeIfPresent(tool_calls, forKey: .tool_calls)
        try c.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
        try c.encode(refusal, forKey: .refusal)
    }
}

extension ChatMessage {
    /// The text fed to the chat template for this message.
    ///
    /// An assistant turn may carry `tool_calls` with `content: nil`; reconstruct
    /// the `<tool_call>` envelope so multi-turn tool conversations present a
    /// consistent history to the model. It may ALSO carry prose `content`
    /// *together with* `tool_calls` — pi sends the model's "Let me check X:"
    /// preamble alongside the call it made in the same turn. Both must be
    /// rendered: dropping the envelope on those turns removes the only example
    /// of "prose + `<tool_call>`" from the history, and the model learns to
    /// announce a tool without ever emitting it (observed on device: the turn
    /// ends right after "Let me check …:" with no call).
    var promptText: String { promptText(format: .jsonHermes) }

    /// Format-aware `promptText`; the envelope syntax follows `format` (see
    /// `ToolWireFormat`). The Qwen branch reproduces its template exactly: a
    /// blank line before the first envelope when prose precedes it, a single
    /// newline before later ones.
    func promptText(format: ToolWireFormat) -> String {
        let text = content?.textValue ?? ""
        guard let toolCalls = tool_calls, !toolCalls.isEmpty else { return text }

        if format == .qwenXML {
            // The template applies `|trim` to the message content *before* it
            // appends the call envelope (`content = render_content(...)|trim`,
            // then `'\n\n<tool_call>'`), so trailing newlines in a client's
            // assistant prose must not push extra blank lines in front of the
            // envelope. pi replays the model's prose verbatim and the turn
            // usually ends with a newline, which put four newlines there.
            let prose = text.trimmingCharacters(in: .whitespacesAndNewlines)
            var out = prose
            for (index, call) in toolCalls.enumerated() {
                let envelope = ToolPrompt.toolEnvelope(name: call.function.name,
                                                       argumentsJSON: call.function.arguments,
                                                       format: format)
                if index == 0 {
                    out += prose.isEmpty ? envelope : "\n\n" + envelope
                } else {
                    out += "\n" + envelope
                }
            }
            return out
        }

        let envelopes = toolCalls.map { call in
            ToolPrompt.toolEnvelope(name: call.function.name,
                                    argumentsJSON: call.function.arguments,
                                    format: format)
        }.joined(separator: "\n")
        return text.isEmpty ? envelopes : text + "\n" + envelopes
    }
}

struct ToolDefinition: Codable {
    let type: String
    let function: ToolFunctionDefinition
}

struct ToolFunctionDefinition: Codable {
    let name: String
    let description: String?
    let parameters: [String: AnyCodable]?
}

/// Wrapper for JSON object fields where we only need pass-through / logging.
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) { self.value = value }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) { value = intVal }
        else if let doubleVal = try? container.decode(Double.self) { value = doubleVal }
        else if let boolVal = try? container.decode(Bool.self) { value = boolVal }
        else if let stringVal = try? container.decode(String.self) { value = stringVal }
        else if let dictVal = try? container.decode([String: AnyCodable].self) { value = dictVal }
        else if let arrayVal = try? container.decode([AnyCodable].self) { value = arrayVal }
        else { value = try container.decode([String: AnyCodable].self) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let intVal = value as? Int { try container.encode(intVal) }
        else if let doubleVal = value as? Double { try container.encode(doubleVal) }
        else if let boolVal = value as? Bool { try container.encode(boolVal) }
        else if let stringVal = value as? String { try container.encode(stringVal) }
        else if let dictVal = value as? [String: AnyCodable] {
            try container.encode(dictVal)
        }
        else if let arrayVal = value as? [AnyCodable] {
            try container.encode(arrayVal)
        }
        else if let dictVal = value as? [String: Any] {
            try container.encode(dictVal.mapValues { AnyCodable($0) })
        }
        else if let arrayVal = value as? [Any] {
            try container.encode(arrayVal.map { AnyCodable($0) })
        }
        else { try container.encodeNil() }
    }
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let temperature: Double?
    let top_p: Double?
    let top_k: Int?
    let min_p: Double?
    let repeat_penalty: Double?
    let repeat_last_n: Int?
    let presence_penalty: Double?
    let frequency_penalty: Double?
    let seed: Int?
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let n_predict: Int?
    let stream: Bool?
    let stop: StopField?
    let tools: [ToolDefinition]?
    let tool_choice: ToolChoiceField?
    let stream_options: StreamOptions?
    /// Qwen3 "thinking" toggle. Clients (e.g. vLLM / llama.cpp style) send this
    /// either at the top level or nested under `chat_template_kwargs`.
    let enable_thinking: Bool?
    let chat_template_kwargs: ChatTemplateKwargs?

    /// Resolves the completion-length limit using llama-server's precedence:
    /// `n_predict` > `max_completion_tokens` > `max_tokens`.
    var resolvedMaxTokens: Int? {
        n_predict ?? max_completion_tokens ?? max_tokens
    }

    /// Whether the model should emit `<think>` reasoning. Defaults to OFF so
    /// short utility calls ("reply with ONLY one word", "Return JSON only")
    /// aren't polluted/truncated by reasoning; clients opt in per request via
    /// `enable_thinking: true` (top level or in `chat_template_kwargs`).
    var thinkingEnabled: Bool {
        enable_thinking ?? chat_template_kwargs?.enable_thinking ?? false
    }
}

/// Pure request-validation rules for `/v1/chat/completions`, kept out of the
/// HTTP handler so they can be unit-tested without a live model.
///
/// OpenAI treats `model` as required: a request that omits it is a client
/// error (400), not something to complete against a default. LlamaServer
/// previously only checked `messages`, so a missing `model` fell through to a
/// normal 200 completion — a spec deviation caught by llm-comply's
/// `chat-error-handling` test. This predicate is the single source of truth
/// the handler consults before generating.
enum ChatRequestSpec {
    /// Returns a human-readable reason the request must be rejected with a 400,
    /// or nil when it is acceptable. Order matters only for the message shown.
    static func validationError(model: String?, messageCount: Int) -> String? {
        if model?.isEmpty ?? true {
            return "`model` must not be empty"
        }
        if messageCount == 0 {
            return "`messages` must not be empty"
        }
        return nil
    }
}

/// Subset of the `chat_template_kwargs` object some clients send; we only read
/// the Qwen3 thinking toggle.
struct ChatTemplateKwargs: Codable {
    let enable_thinking: Bool?
}


struct StreamOptions: Codable {
    let include_usage: Bool?
}

/// OpenAI's `stop` may be a single string or an array of strings.
enum StopField: Codable {
    case one(String)
    case many([String])

    var values: [String] {
        switch self {
        case .one(let s): return [s]
        case .many(let a): return a
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .one(s) }
        else { self = .many((try? c.decode([String].self)) ?? []) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .one(let s): try c.encode(s)
        case .many(let a): try c.encode(a)
        }
    }
}

/// `tool_choice` may be "auto" / "none" / "required" or a specific function object.
enum ToolChoiceField: Codable {
    case mode(String)               // "auto" | "none" | "required"
    case function(String)           // a named function

    var disablesTools: Bool {
        if case .mode(let m) = self { return m == "none" }
        return false
    }
    var requiresTool: Bool {
        switch self {
        case .mode(let m): return m == "required"
        case .function:    return true
        }
    }
    var forcedFunctionName: String? {
        if case .function(let n) = self { return n }
        return nil
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .mode(s)
        } else if let obj = try? c.decode([String: AnyCodable].self),
                  let fn = obj["function"]?.value as? [String: AnyCodable],
                  let name = fn["name"]?.value as? String {
            self = .function(name)
        } else {
            self = .mode("auto")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .mode(let m): try c.encode(m)
        case .function(let n):
            try c.encode(AnyCodable(["type": "function", "function": ["name": n]]))
        }
    }
}

struct ChatCompletionChoice: Codable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String
    /// OpenAI marks `logprobs` required on each choice. The server computes no
    /// token log-probabilities, so we encode it as explicit null — present, but
    /// null — to satisfy strict clients (llm-comply's pinned spec requires it).
    let logprobs: String?

    enum CodingKeys: String, CodingKey {
        case index, message, finish_reason, logprobs
    }

    init(index: Int, message: ChatMessage, finish_reason: String) {
        self.index = index
        self.message = message
        self.finish_reason = finish_reason
        self.logprobs = nil
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encode(message, forKey: .message)
        try c.encode(finish_reason, forKey: .finish_reason)
        try c.encode(logprobs, forKey: .logprobs)
    }
}

struct Usage: Codable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}

struct ChatCompletionResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChatCompletionChoice]
    let usage: Usage
    let system_fingerprint: String

    init(id: String, object: String, created: Int, model: String,
         choices: [ChatCompletionChoice], usage: Usage,
         system_fingerprint: String = openAISystemFingerprint) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = system_fingerprint
    }
}

struct ModelInfo: Codable {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
}

struct ModelList: Codable {
    let object: String
    let data: [ModelInfo]
}

struct APIError: Codable {
    struct Detail: Codable {
        let message: String
        let type: String
    }
    let error: Detail
}

/// A single content part in a multi-part message body (OpenAI `content` array).
/// Currently supports `text` parts; extensible for future types (`image_url`, etc.).
struct ContentPart: Codable {
    let type: String
    let text: String?
}

/// OpenAI message `content` can be a plain string or an array of `ContentPart`.
enum MessageContent: Codable {
    case text(String)
    case parts([ContentPart])

    var textValue: String? {
        switch self {
        case .text(let s): return s
        case .parts(let parts): return parts.compactMap(\.text).joined(separator: "\n")
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            self = .text(string)
        } else if let parts = try? container.decode([ContentPart].self) {
            self = .parts(parts)
        } else {
            throw DecodingError.typeMismatch(
                MessageContent.self,
                DecodingError.Context(codingPath: container.codingPath,
                                      debugDescription: "Expected string or array of content parts"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .parts(let parts): try container.encode(parts)
        }
    }
}

// MARK: - Streaming (SSE) models

struct StreamingChunk: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [StreamingChoice]
    let usage: Usage?
    let system_fingerprint: String

    init(id: String, created: Int, model: String, choices: [StreamingChoice], usage: Usage? = nil) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = openAISystemFingerprint
    }
}

struct StreamingChoice: Codable {
    let index: Int
    let delta: Delta
    let finish_reason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta, finish_reason
    }

    init(index: Int, delta: Delta, finish_reason: String?) {
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason
    }

    // OpenAI / llama-server always include `finish_reason` in every chunk
    // (null until the final one). Swift would otherwise omit a nil optional,
    // so encode it explicitly as null to match the reference contract.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encode(delta, forKey: .delta)
        try c.encode(finish_reason, forKey: .finish_reason)
    }
}

struct Delta: Codable {
    let role: String?
    let content: String?
    let tool_calls: [StreamingToolCall]?

    init(role: String?, content: String?, tool_calls: [StreamingToolCall]? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
    }
}

/// Streaming tool-call delta. Unlike the non-streaming `ToolCall`, OpenAI's
/// streaming format requires an `index` so clients can accumulate the call's
/// fields across chunks (mirrors llama-server's `tool_call["index"]`).
struct StreamingToolCall: Codable {
    let index: Int
    let id: String
    let type: String
    let function: ToolCallFunction

    init(index: Int, from call: ToolCall) {
        self.index = index
        self.id = call.id
        self.type = call.type
        self.function = call.function
    }
}

// MARK: - Legacy text-completion models (/v1/completions)

struct CompletionRequest: Codable {
    let model: String?
    let prompt: PromptField
    let temperature: Double?
    let top_p: Double?
    let top_k: Int?
    let min_p: Double?
    let repeat_penalty: Double?
    let presence_penalty: Double?
    let frequency_penalty: Double?
    let seed: Int?
    let max_tokens: Int?
    let n_predict: Int?
    let stop: StopField?

    /// Resolves the completion-length limit: `n_predict` > `max_tokens`.
    var resolvedMaxTokens: Int? {
        n_predict ?? max_tokens
    }
}

/// The legacy `prompt` field may be a string or an array of strings.
enum PromptField: Codable {
    case one(String)
    case many([String])

    var joined: String {
        switch self {
        case .one(let s): return s
        case .many(let a): return a.joined(separator: "\n")
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .one(s) }
        else { self = .many((try? c.decode([String].self)) ?? []) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .one(let s): try c.encode(s)
        case .many(let a): try c.encode(a)
        }
    }
}

struct CompletionChoice: Codable {
    let index: Int
    let text: String
    let finish_reason: String
}

struct CompletionResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: Usage
    let system_fingerprint: String

    init(id: String, object: String, created: Int, model: String,
         choices: [CompletionChoice], usage: Usage,
         system_fingerprint: String = openAISystemFingerprint) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = system_fingerprint
    }
}
