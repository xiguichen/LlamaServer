import Foundation

/// Pure helper for locating the final assistant primer in token space, without
/// re-tokenizing the whole prompt.
///
/// The hybrid snapshot must stop before the trailing `<|im_start|>assistant`
/// primer (the next turn replaces it with the real assistant message), so the
/// stable prefix is everything before that primer. The caller already has the
/// whole prompt's tokens; tokenizing just the short prompt *tail* from the last
/// primer and checking it is aligned with `promptTokens` gives the same boundary
/// in O(tail). A mismatch returns nil and the caller keeps the old full-prefix
/// tokenization as the fallback.
enum PromptBoundary {

    /// Length of the stable prompt prefix when `tailTokens` is the tokenization
    /// of the prompt suffix starting at the final primer. Returns nil unless the
    /// tail is non-empty and is an exact token suffix of `promptTokens`.
    static func stablePrefixLength(promptTokens: [Int32], tailTokens: [Int32]) -> Int? {
        guard !tailTokens.isEmpty, tailTokens.count <= promptTokens.count else { return nil }
        guard promptTokens.suffix(tailTokens.count).elementsEqual(tailTokens) else { return nil }
        return promptTokens.count - tailTokens.count
    }

    /// Token count the hybrid prompt-state snapshot may cover.
    ///
    /// A whole-prompt snapshot is never restorable: the next turn replaces the
    /// trailing assistant-primer token, and a completions append can re-merge
    /// the final BPE token, so the longest common prefix stops one token short
    /// and `restorePromptSnapshot` has to discard the blob for a full
    /// re-prefill (measured at 65 s / 15.6k tokens). Holding the last token
    /// back makes the saved state a strict prefix of every continuation; that
    /// token is decoded immediately after the save.
    static func snapshotBoundary(stablePrefix: Int, total: Int) -> Int {
        guard stablePrefix > 0, total > 1 else { return stablePrefix }
        return min(stablePrefix, total - 1)
    }
}
