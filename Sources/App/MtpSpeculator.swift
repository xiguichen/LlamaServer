import Foundation

/// Pure decision logic for one verify round of depth-N MTP speculative
/// decoding.
///
/// No dependency on the llama C API — operates on plain token ids (`Int32`)
/// and positions (`Int`) so it can be unit-tested without a model file or the
/// xcframework, exactly like `MtpSpec`/`MtpProbe`.
///
/// Per round the target context has decoded `[idLast, d0, d1, ...]` — one row
/// for the last emitted token plus one row per drafted token — and produced
/// one "next-token" hidden state per row (row k = successor of the token at
/// `nextPos + k`). The driver samples `reals[k]` from row k in order and stops
/// at the first miss or eog, so every sampled real except an eog is emitted,
/// or the round ends generation at the limit / a stop string
/// (`llama_sampler_sample` accepts each token it returns). Drafts are never
/// eog (the driver filters them). The rules:
///
///   * `accepted` `a` — the leading drafts that match their sampled reals.
///   * `emitCount = min(a + 1, firstStopRow)`: the sampled real at row 0 is
///     the real successor of `idLast` and is always committed (unless eog
///     caps it); each accepted draft lets the next row's real be committed;
///     the row after the last accepted draft is the "bonus" real emitted on
///     full accept. An eog real (firstStopRow) is never emitted, so emission
///     stops there.
///   * `rollbackPos = nextPos + emitCount` when decoded rows remain
///     (`emitCount < drafts.count + 1`), nil when every decoded row was
///     consumed. The same position is seq_rm'd from BOTH contexts.
///   * `carryHFromRow = emitCount - 1` (clamped to 0): the hidden state of
///     the last emitted row, which pairs with the last emitted token — the
///     h/token pairing is the historically bug-prone part (nearly inverted
///     during depth-1 development) and must always be `emitCount - 1`.
struct MtpSpeculator {

    struct Decision: Equatable {
        /// Number of drafted tokens verified as real (0...drafts.count).
        let accepted: Int
        /// Number of sampled reals to commit as generated tokens
        /// (0...drafts.count + 1, capped at the first stop row).
        let emitCount: Int
        /// First position to `seq_rm` from both contexts when decoded rows
        /// remain beyond what was emitted; nil when all rows were consumed.
        let rollbackPos: Int?
        /// Which verify-output row's hidden state to carry into the next
        /// round — always the last emitted row (`emitCount - 1`, clamped 0).
        let carryHFromRow: Int32
        /// Whether at least one draft was accepted (stats alias).
        var hit: Bool { accepted > 0 }
    }

    /// Resolve one speculative verify round.
    /// - Parameters:
    ///   - drafts: the drafted tokens proposed this round (never eog).
    ///   - reals: sampled successors of the verify rows, in row order. The
    ///     driver samples lazily: it stops after the first real that misses
    ///     its draft or is eog, so `reals` may be shorter than
    ///     `drafts.count + 1`. Rows past that point can never be emitted, so
    ///     the decision is the same as with every row sampled. The
    ///     `reals.count` bounds below are what make a short `reals` safe.
    ///   - firstStopRow: index of the first eog among the SAMPLED reals, or
    ///     nil (an eog after the first miss is never sampled).
    ///   - nextPos: target write position where `idLast` was (re)decoded.
    static func decide(drafts: [Int32], reals: [Int32], firstStopRow: Int?, nextPos: Int) -> Decision {
        let stopRow = min(firstStopRow ?? drafts.count + 1, reals.count)
        var accepted = 0
        while accepted < drafts.count,
              accepted < reals.count,
              drafts[accepted] == reals[accepted] {
            accepted += 1
        }
        let emitCount = min(accepted + 1, stopRow)
        let rollbackPos = emitCount < drafts.count + 1 ? nextPos + emitCount : nil
        let carryHFromRow = Int32(max(0, emitCount - 1))
        return Decision(accepted: accepted,
                        emitCount: emitCount,
                        rollbackPos: rollbackPos,
                        carryHFromRow: carryHFromRow)
    }

    /// Indices into a round's sampled `reals` whose draft-context rows are
    /// missing once the round is committed, and which the next round must
    /// decode as a prefix of its own draft row so the draft KV stays contiguous
    /// with the emitted sequence.
    ///
    /// The draft context wrote positions `nextPos ... nextPos + draftEntries - 1`
    /// this round, while the emitted tokens occupy
    /// `nextPos ... nextPos + emitCount - 1` (`reals[i]` sits at
    /// `nextPos + i + 1`). The missing tail is therefore `reals[draftEntries - 1
    /// ... emitCount - 2]` — every emitted token except the last, which the next
    /// round decodes as its own input row.
    ///
    /// - Parameters:
    ///   - emitCount: committed reals this round (from `decide`).
    ///   - draftEntries: positions actually written into the draft context,
    ///     counting from `nextPos` (1 after the initial decode, 2 if the
    ///     chained decode also succeeded). A chained link that was decoded and
    ///     then cut has already written its position, so the start bound skips
    ///     it — the recurrent draft context must not rewrite a consumed
    ///     position, and a re-decode would carry the target's h rather than the
    ///     chain's.
    static func draftSeedIndices(emitCount: Int, draftEntries: Int) -> Range<Int> {
        let start = max(0, draftEntries - 1)
        let end = max(start, emitCount - 1)
        return start..<end
    }
}
