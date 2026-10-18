import Foundation

/// Pure formatting for the `prefill=` segment of the `generated N tokens` line.
///
/// After prompt-prefix reuse the prefill phase decodes only the prompt suffix
/// (the rest comes back from the KV cache or the on-disk snapshot), so dividing
/// the *full* prompt length by the phase duration reports nonsense — on device
/// a 48,450-token prompt that decoded 113 tokens printed
/// `prefill=3.14s 15431.54t/s`. The rate must use the tokens actually decoded,
/// and the reuse split belongs in the line so a reader can tell a 3 s phase
/// over 113 new tokens from a 3 s phase over 48k.
enum PrefillStats {

    /// The `prefill=…` segment of the generation summary, e.g.
    /// `prefill=20.70s 159.76t/s (3307 of 35074, 31767 reused)`.
    ///
    /// `reusedTokens` is the prompt prefix already resident in the cache
    /// (`reuseLen`); it clamps into `0...promptTokens`. A zero or negative
    /// duration reports a `0.00t/s` rate instead of nan/inf.
    static func segment(promptTokens: Int, reusedTokens: Int, seconds: TimeInterval) -> String {
        let prompt = max(0, promptTokens)
        let reused = min(max(0, reusedTokens), prompt)
        let decoded = prompt - reused
        let rate = seconds > 0 ? Double(decoded) / seconds : 0
        return String(format: "prefill=%.2fs %.2ft/s (%d of %d, %d reused)",
                      max(0, seconds), rate, decoded, prompt, reused)
    }
}
