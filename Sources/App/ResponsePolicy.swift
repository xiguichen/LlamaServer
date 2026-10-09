import Foundation

// MARK: - ContextHeadroom

/// Pure fail-fast decision for prompts that leave no room for a usable
/// response. Extracted from `LlamaInference.generate()` so it can be
/// unit-tested without a model.
///
/// Regression this guards: a 19,927-token prompt in a 20,068-token context
/// (MemoryBudget clamped the window down to fit device memory) generated 141
/// tokens, hit `context_full`, and was reported as `finish_reason: "length"`
/// — pi rendered a misleading "Response was truncated before completion"
/// banner instead of telling the user the conversation no longer fits.
enum ContextHeadroom {

    /// Minimum tokens that must remain after the prompt for a response to be
    /// worth generating.
    static let defaultMinimumCompletionTokens = 256

    /// Returns nil when the prompt leaves enough headroom for a usable
    /// response; otherwise an actionable message naming the exact token
    /// counts (suitable verbatim for an API error body).
    static func check(promptTokens: Int,
                      contextSize: Int,
                      minimumCompletionTokens: Int = defaultMinimumCompletionTokens) -> String? {
        guard contextSize > 0 else {
            return "Context size \(contextSize) cannot host a response. Check the context configuration and restart the server."
        }
        // Tiny contexts require only a proportional share so small windows
        // stay usable (a fixed minimum would reject every prompt there).
        let required = min(minimumCompletionTokens, contextSize / 2)
        let headroom = contextSize - promptTokens
        guard headroom < required else { return nil }
        return "Prompt uses \(promptTokens) of \(contextSize) context tokens; "
            + "only \(max(0, headroom)) remain for the response (minimum \(required)). "
            + "Shorten the conversation history or increase the context size."
    }
}

// MARK: - FinishReasonPolicy

/// Maps the engine's finish reason (plus the outcome of tool-call parsing)
/// onto OpenAI's `finish_reason` vocabulary, honestly. Extracted from
/// `LlamaHTTPServer` so all three response paths share one policy.
///
/// Contract:
/// - `length` is reserved for genuine max-token cutoffs — the only case OpenAI
///   semantics (and pi's truncation banner) can trust.
/// - `context_full` and unusable output (a tool-call envelope the parser
///   could not salvage) become an explicit `error`, never a fake truncation.
/// - Parsed tool calls win, per the OpenAI contract.
enum FinishReasonPolicy {

    static func map(engineFinish: String, hasToolCalls: Bool, outputUnusable: Bool) -> String {
        if hasToolCalls { return "tool_calls" }
        switch engineFinish {
        case "length":
            // max_tokens genuinely cut the response — honest truncation.
            return "length"
        case "context_full":
            // Context wall, not a max-token cutoff. After the headroom
            // check this means the conversation outgrew the window mid-
            // generation: surface it as an error the client can act on
            // instead of a truncation it would retry into the same wall.
            return "error"
        default:
            // A normal stop that yielded no usable tool call is a model
            // failure, not a truncation.
            return outputUnusable ? "error" : "stop"
        }
    }
}
