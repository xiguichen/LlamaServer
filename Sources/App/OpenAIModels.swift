import Foundation

// Minimal OpenAI-compatible request/response shapes for /v1/chat/completions
// and /v1/models. These are intentionally lenient on input.

struct ChatMessage: Codable {
    let role: String
    let content: String
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
    let stream: Bool?
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
