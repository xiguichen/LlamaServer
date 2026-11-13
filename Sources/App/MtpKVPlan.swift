import Foundation

/// Pure bookkeeping for keeping the target KV cache across an MTP generation.
///
/// The speculative loop decodes `idLast` plus the drafts, samples the real
/// successors, emits the committed ones, and rolls back the rows beyond what was
/// emitted. The last emitted token is a *prediction* sampled from the last
/// resident row — decoding it would only happen on the next round — so a normal
/// round leaves `prompt + generated - 1` tokens resident. A fallback round that
/// stops on eog leaves the decoded `idLast` row resident, i.e.
/// `prompt + generated`.
///
/// Publishing a wrong count would let the next request skip decoding tokens that
/// are not in the cache, so the rules here are pure and unit-tested, and the
/// caller additionally checks the live sequence bounds before trusting them
/// (`docs/MTP-KV-REUSE.md`). Token ids are `Int32` so this file stays free of the
/// llama module, like `MtpSpec`/`ReuseSurvival`.
enum MtpKVPlan {

    /// Resident token count after a speculative round that started at
    /// `roundStartPos` and committed `emitted` tokens (a verify round, or the
    /// fallback round after its successor was emitted).
    ///
    /// The round's input row is resident; accepted predictions fill the rows
    /// after it up to `emitted`, and the last emitted token is the next round's
    /// input, not a decoded row.
    static func committedResident(roundStartPos: Int, emitted: Int) -> Int {
        roundStartPos + max(0, emitted)
    }

    /// Resident token count after a fallback round whose successor was eog or a
    /// sampler error. The fallback decoded `idLast` at `roundStartPos` and then
    /// stopped, so that row — the last emitted token — is resident.
    static func fallbackStopResident(roundStartPos: Int) -> Int {
        roundStartPos + 1
    }

    /// The token ids matching `resident` (`prompt + the emitted ids that were
    /// actually decoded`), or nil when the bookkeeping is inconsistent and the
    /// caller must clear instead of publishing.
    static func residentIds(prompt: [Int32], emitted: [Int32], resident: Int) -> [Int32]? {
        let generatedResident = resident - prompt.count
        guard generatedResident >= 0, generatedResident <= emitted.count else { return nil }
        return prompt + emitted.prefix(generatedResident)
    }

    /// Whether the live sequence bounds prove the tracked prefix is exactly
    /// resident: nothing evicted at the front (`posMin == 0`) and the sequence
    /// ends where the bookkeeping says it does. A cache type that reports
    /// different bounds fails the guard and the caller falls back to the old
    /// clear path — correctness is preserved, only the optimisation is lost.
    static func boundsMatch(posMin: Int, posMax: Int, resident: Int) -> Bool {
        posMin == 0 && posMax + 1 == resident
    }

    /// The tracked prefix the live cache can actually serve: the sequence length
    /// when nothing was evicted, else 0. Used to clamp `cachedTokenIds` for dense
    /// reuse, which may now extend past the live rows if preservation was
    /// interrupted; an evicted front makes the whole tracking unusable.
    static func liveResidentCount(posMin: Int, posMax: Int) -> Int {
        posMin == 0 ? max(0, posMax + 1) : 0
    }

    /// How much of the reused prefix the draft's resident rows can cover.
    ///
    /// The draft mirrors the target's rows and is rolled back at the same
    /// positions, so after a generation it holds the same tokens as
    /// `cachedTokenIds` — including the generated rows that target-KV
    /// preservation keeps. Reusing it requires the whole `reuseLen` prefix:
    /// trimming the draft to fewer rows than the target reuses would leave a
    /// position gap, because the missing rows cannot be re-mirrored without the
    /// target's hidden states for those (already decoded) positions.
    ///
    /// - Returns: `reuseLen` when the draft holds at least that many rows and
    ///   they match the prompt; nil when the caller must clear it instead.
    static func draftReuseLen(reuseLen: Int,
                              draftTokens: [Int32],
                              promptTokens: [Int32]) -> Int? {
        guard reuseLen > 0,
              reuseLen <= draftTokens.count,
              reuseLen <= promptTokens.count else { return nil }
        for i in 0..<reuseLen where draftTokens[i] != promptTokens[i] { return nil }
        return reuseLen
    }
}
