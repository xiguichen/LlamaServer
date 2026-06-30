import Foundation

/// Pure decision logic for one verify round of depth-1 MTP speculative decoding.
///
/// No dependency on the llama C API — operates on plain token ids (`Int32`) and
/// positions (`Int`) so it can be unit-tested without a model file or the
/// xcframework, exactly like `MtpProbe`.
///
/// Per round the target context has decoded `[idLast, d0]` and produced two
/// "next-token" hidden states (row 0 = successor of `idLast`, row 1 = successor
/// of `d0`). The real successor `s0` is sampled from row 0. The draft `d0` was a
/// speculative guess at that successor:
///
///   * HIT  (`d0 == s0`): the guess held, so `d0`'s position is valid and row 1
///     yields a *second* token `s1` for free. Emit `[s0, s1]`, advance the write
///     position by 2, and carry the row-1 hidden state into the next round
///     (it pairs with `s1`). No rollback.
///   * MISS (`d0 != s0`): the guess was wrong, so the speculatively-decoded
///     `d0` at `nextPos + 1` must be rolled back from both contexts. Emit `[s0]`,
///     advance by 1, and carry the row-0 hidden state (it pairs with `s0`).
///
/// The `carryHFromRow` field is the historically bug-prone part (the h/token
/// pairing was nearly inverted during development): it MUST be row 1 on a hit and
/// row 0 on a miss.
struct MtpSpeculator {

    struct Decision: Equatable {
        /// Whether the draft token was accepted.
        let hit: Bool
        /// Amount to advance the target write position (2 on hit, 1 on miss).
        let posAdvance: Int
        /// Position to `seq_rm` from both contexts on a miss; nil on a hit.
        let rollbackPos: Int?
        /// Which verify-output row's hidden state to carry into the next round
        /// (1 on a hit so it pairs with the bonus token, 0 on a miss so it pairs
        /// with the resampled token).
        let carryHFromRow: Int32
    }

    /// Resolve one speculative verify round.
    /// - Parameters:
    ///   - d0: the speculatively drafted token.
    ///   - s0: the real successor of `idLast`, sampled from target row 0.
    ///   - nextPos: target write position where `idLast` was (re)decoded.
    static func decide(d0: Int32, s0: Int32, nextPos: Int) -> Decision {
        if d0 == s0 {
            return Decision(hit: true, posAdvance: 2, rollbackPos: nil, carryHFromRow: 1)
        }
        return Decision(hit: false, posAdvance: 1, rollbackPos: nextPos + 1, carryHFromRow: 0)
    }
}
