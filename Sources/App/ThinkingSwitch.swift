import Foundation

/// Applies the client's `enable_thinking` choice to an already-rendered prompt.
///
/// `enable_thinking` is a chat-template parameter: upstream llama-server hands
/// it to the model's Jinja template, whose `enable_thinking is false` branch
/// prefills an empty, already-closed reasoning block. The vendored C API
/// (`llama_chat_apply_template`) takes no template kwargs, so the model's own
/// template always renders with `enable_thinking` undefined — the *thinking on*
/// shape — and the flag has to be applied to the rendered text instead,
/// appending exactly the block the template would have emitted.
///
/// The prompt's tail says which shape the renderer produced:
/// - `<|im_start|>assistant\n` — the bare assistant primer. The model's own
///   template ends here, and so does the built-in `fallbackPrompt`, so the
///   block has to be opened *and* closed.
/// - `<|im_start|>assistant\n<think>\n` — a renderer that already opened the
///   block; only the closer is missing.
///
/// Any other tail is returned untouched: there is no block to close, and a
/// "thinking on" prompt must not grow one. That also makes the switch
/// idempotent — a prompt that carries it no longer ends in a primer.
///
/// Pure and unit-tested (`ThinkingSwitchTests`).
enum ThinkingSwitch {

    /// The empty, closed reasoning block. This is the shape the model emits
    /// itself when it is told to skip reasoning, so the prefill continues
    /// exactly as the model expects.
    static let closedBlock = "<think>\n\n</think>\n\n"

    /// Assistant primer the model's template renders for this family when
    /// `enable_thinking` is undefined (i.e. a request that never defined it).
    static let barePrimer = "<|im_start|>assistant\n"

    /// The same primer with the reasoning block already open.
    static let openPrimer = barePrimer + "<think>\n"

    /// Returns `prompt`, closed off for thinking when the client disabled it.
    static func apply(to prompt: String, thinkingEnabled: Bool) -> String {
        if thinkingEnabled { return prompt }
        if prompt.hasSuffix(openPrimer) { return prompt + "\n</think>\n\n" }
        if prompt.hasSuffix(barePrimer) { return prompt + closedBlock }
        return prompt
    }
}
