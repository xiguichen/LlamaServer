import Foundation

/// Pure MTP configuration decisions — output-slot sizing and draft-confidence
/// gating. No llama C API lives here (the caller passes the GGUF-derived
/// head count and the raw logits row), so both are unit-testable without a
/// model file, exactly like `KVCacheSpec`/`MtpSpeculator`.
enum MtpSpec {

    /// Minimum probability a *computed* draft token must carry to survive the
    /// confidence gate. Zero disables discarding entirely — upstream's
    /// `--spec-draft-p-min` default is 0.00, and here it must stay that way:
    /// the draft decode has already run by the time the gate is consulted, so
    /// its cost is sunk. The alternative to verifying a computed draft is a
    /// plain single-token decode of the same position, which costs nearly the
    /// same (one fewer target row) and can never yield a bonus token. A gate
    /// above zero therefore only ever throws away work. Upstream's p_min is
    /// used to cut deeper chain links *before* they are decoded; at depth 1
    /// (`mtpMaxDraft == 1`) there are no chain links to cut.
    static let pMin: Float = 0.0

    /// Whether the confidence gate can ever discard a computed draft. When
    /// false, callers must not pay for `draftConfidence`'s vocab-wide softmax
    /// (a full 248K-entry exp pass per draft on device).
    static var gateEnabled: Bool { pMin > 0 }

    /// How many output rows the MTP-enabled contexts must expose.
    ///
    /// Two constraints compete:
    ///  - the model's MTP head count (main head + `nextn_predict_layers` MTP
    ///    heads) — a property of the GGUF, overridable by an explicit count;
    ///  - the speculative verify batch (one row per drafted token plus the
    ///    row for `idLast`), i.e. `maxDraft + 1` — a property of the runtime
    ///    configuration, which can exceed the head count once drafting goes
    ///    deeper than the number of MTP heads allows.
    /// The result is the larger of the two; `headsFromGGUF == nil` preserves
    /// the historical metadata-absent fallback of 3.
    static func resolveOutputSlots(headsFromGGUF: Int?, explicitHeads: Int, maxDraft: Int) -> Int {
        let heads: Int
        if explicitHeads > 0 {
            heads = explicitHeads
        } else if let gguf = headsFromGGUF {
            heads = max(1, gguf)
        } else {
            heads = 3
        }
        return max(heads, maxDraft + 1)
    }

    /// Probability the draft model assigns to `token` under a softmax over
    /// one vocab-length logits row. Max-subtracted for numerical stability
    /// (raw logits reach magnitudes where `exp` overflows float32). An
    /// out-of-range token yields 0 — "the draft model said nothing about
    /// this", which correctly fails any `pMin` gate.
    static func draftConfidence(row: UnsafeBufferPointer<Float>, token: Int32) -> Float {
        guard token >= 0, Int(token) < row.count else { return 0 }
        var maxValue: Float = -.infinity
        for value in row { if value > maxValue { maxValue = value } }
        guard maxValue.isFinite else { return 0 }
        var expSum: Float = 0
        for value in row { expSum += exp(value - maxValue) }
        guard expSum > 0, expSum.isFinite else { return 0 }
        return exp(row[Int(token)] - maxValue) / expSum
    }
}
