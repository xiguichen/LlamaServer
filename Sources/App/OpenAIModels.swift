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
    let content: String?
    let tool_calls: [ToolCall]?
    let tool_call_id: String?

    init(role: String, content: String?, tool_calls: [ToolCall]? = nil, tool_call_id: String? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
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

    /// Resolves the completion-length limit using llama-server's precedence:
    /// `n_predict` > `max_completion_tokens` > `max_tokens`.
    var resolvedMaxTokens: Int? {
        n_predict ?? max_completion_tokens ?? max_tokens
    }
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
            try c.encode(["type": "function", "function": ["name": n]])
        }
    }
}

struct ChatCompletionChoice: Codable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String
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
