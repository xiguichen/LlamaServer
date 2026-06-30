import Foundation

/// Pure-function MTP probe state machine. No dependency on the llama C API.
/// Testable without a model file or xcframework.
struct MtpProbe {

    /// Returns the number of output rows that can be safely sampled (≥1),
    /// mutating tracking flags as needed.
    ///
    /// State transitions:
    ///   mtpCount==1        → return 1 (MTP not applicable).
    ///   genTokens==1       → return 1 (only main head after prompt decode).
    ///   !checked           → probe -1/-2 via `checkLogits`, cache result.
    ///   checked+active     → enumerate all mtpCount slots.
    ///   checked+!active    → return 1 (no probing, no errors).
    static func computeOutputs(
        mtpCount: Int,
        genTokens: Int,
        mtpAvailabilityChecked: inout Bool,
        mtpHeadsActive: inout Bool,
        checkLogits: (Int32) -> Bool
    ) -> Int {
        guard mtpCount > 1 else { return 1 }

        // Phase 1: first generation token — only main head is available.
        if !mtpAvailabilityChecked && genTokens == 1 {
            return 1
        }

        // Phase 2: probe MTP availability exactly once.
        if !mtpAvailabilityChecked {
            let hasMain = checkLogits(-1)
            let hasMtp = checkLogits(-2)
            mtpHeadsActive = hasMain && hasMtp
            mtpAvailabilityChecked = true
            return max((hasMain ? 1 : 0) + (hasMtp ? 1 : 0), 1)
        }

        // Phase 3a: MTP confirmed — enumerate all expected slots.
        if mtpHeadsActive {
            var count = 0
            for j in stride(from: -1, through: -Int32(mtpCount), by: -1) {
                guard checkLogits(j) else { break }
                count += 1
            }
            return count
        }

        // Phase 3b: MTP unavailable — main head only.
        return 1
    }
}
