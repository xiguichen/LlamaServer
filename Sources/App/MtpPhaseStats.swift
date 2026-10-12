import Foundation

/// Pure per-phase timing accumulator for MTP speculative rounds — the logic
/// behind the round-level instrumentation that answers *where* a speculative
/// round's time goes (draft head vs target verify vs sampler). The caller
/// measures the deltas (`CFAbsoluteTimeGetCurrent()` around each C-API call)
/// and injects them; no clocks and no llama C API live here, so aggregation,
/// rounding, and formatting are unit-testable (see MtpPhaseStatsTests),
/// exactly like `StartupTimer` (startup phases) — this is its round-level
/// sibling.
struct MtpPhaseStats {

    private var draftTotal: TimeInterval = 0
    private var verifyTotal: TimeInterval = 0
    private var sampleTotal: TimeInterval = 0
    private var catchupTotal: TimeInterval = 0

    /// Number of speculative rounds recorded (one quadruplet per round).
    private(set) var roundCount: Int = 0

    /// Adds one round's phase durations. A zero/near-zero duration is real
    /// data and is kept; a negative delta is clock skew and is clamped to 0 so
    /// it can never show up as a negative phase (same rule as StartupTimer).
    mutating func record(draft: TimeInterval, verify: TimeInterval, sample: TimeInterval, catchup: TimeInterval = 0) {
        draftTotal += max(0, draft)
        verifyTotal += max(0, verify)
        sampleTotal += max(0, sample)
        catchupTotal += max(0, catchup)
        roundCount += 1
    }

    /// Per-round averages in milliseconds, e.g.
    /// `draft 0.31ms/rd + verify 11.90ms/rd + sample 0.04ms/rd + catchup 0.00ms/rd (3 rounds)`.
    /// Averages (not totals) because round counts vary with acceptance — the
    /// per-round cost is the number that stays comparable across runs.
    func summary() -> String {
        guard roundCount > 0 else { return "no MTP rounds recorded" }
        let rounds = Double(roundCount)
        return String(format: "draft %.2fms/rd + verify %.2fms/rd + sample %.2fms/rd + catchup %.2fms/rd (%d rounds)",
                      draftTotal / rounds * 1000,
                      verifyTotal / rounds * 1000,
                      sampleTotal / rounds * 1000,
                      catchupTotal / rounds * 1000,
                      roundCount)
    }
}
