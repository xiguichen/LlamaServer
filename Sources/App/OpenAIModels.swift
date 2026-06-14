import Foundation

// Minimal OpenAI-compatible request/response shapes for /v1/chat/completions
// and /v1/models. These are intentionally lenient on input.

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
    let max_tokens: Int?
    let stream: Bool?
    let tools: [ToolDefinition]?
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

    init(id: String, created: Int, model: String, choices: [StreamingChoice]) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
    }
}

struct StreamingChoice: Codable {
    let index: Int
    let delta: Delta
    let finish_reason: String?
}

struct Delta: Codable {
    let role: String?
    let content: String?
}
