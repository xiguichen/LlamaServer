import Foundation

/// Pure per-phase timing accumulator for MTP speculative rounds — the logic
/// behind the round-level instrumentation that answers *where* a speculative
/// round's time goes (draft head vs target verify vs sampler). The caller
/// measures the deltas (`CFAbsoluteTimeGetCurrent()` around each C-API call)
/// and injects them; no clocks and no llama C API live here, so aggregation,
/// rounding, and formatting are unit-testable (see MtpPhaseStatsTests),
/// exactly like `StartupTimer` (startup phases) — this is its round-level
/// sibling.
///
/// `draft`/`verify` are the CPU enqueue of each llama_decode (Metal decode
/// is async); `draftGpu`/`verifyGpu` are the explicit `llama_synchronize`
/// waits right after them, so `sample` is CPU sampling only. `emit` is the
/// token text/stop/stream work, and `host` is the rest of the round's wall
/// clock (rollback, hidden-state copies, seed planning), so the phases sum
/// to `wall`.
struct MtpPhaseStats {

    private var draftTotal: TimeInterval = 0
    private var verifyTotal: TimeInterval = 0
    private var sampleTotal: TimeInterval = 0
    private var draftGpuTotal: TimeInterval = 0
    private var verifyGpuTotal: TimeInterval = 0
    private var emitTotal: TimeInterval = 0
    private var wallTotal: TimeInterval = 0

    /// Number of speculative rounds recorded (one set of phases per round).
    private(set) var roundCount: Int = 0

    /// Adds one round's phase durations. A zero/near-zero duration is real
    /// data and is kept; a negative delta is clock skew and is clamped to 0 so
    /// it can never show up as a negative phase (same rule as StartupTimer).
    mutating func record(draft: TimeInterval, verify: TimeInterval, sample: TimeInterval,
                         draftGpu: TimeInterval, verifyGpu: TimeInterval,
                         emit: TimeInterval, wall: TimeInterval) {
        draftTotal += max(0, draft)
        verifyTotal += max(0, verify)
        sampleTotal += max(0, sample)
        draftGpuTotal += max(0, draftGpu)
        verifyGpuTotal += max(0, verifyGpu)
        emitTotal += max(0, emit)
        wallTotal += max(0, wall)
        roundCount += 1
    }

    /// Per-round averages in milliseconds, e.g.
    /// `draft 0.31ms/rd + verify 0.90ms/rd + sample 0.04ms/rd (3 rounds);
    /// draft-gpu 9.00ms/rd + verify-gpu 52.00ms/rd + emit 0.02ms/rd +
    /// host 0.10ms/rd = wall 62.37ms/rd`.
    /// The part before `;` keeps its original format so existing log parsers
    /// still match it. Averages (not totals) because round counts vary with
    /// acceptance — the per-round cost is the number that stays comparable
    /// across runs.
    func summary() -> String {
        guard roundCount > 0 else { return "no MTP rounds recorded" }
        let rounds = Double(roundCount)
        let measured = draftTotal + verifyTotal + sampleTotal + draftGpuTotal + verifyGpuTotal + emitTotal
        let host = max(0, wallTotal - measured)
        let format = "draft %.2fms/rd + verify %.2fms/rd + sample %.2fms/rd (%d rounds); "
            + "draft-gpu %.2fms/rd + verify-gpu %.2fms/rd + emit %.2fms/rd + host %.2fms/rd = wall %.2fms/rd"
        return String(format: format,
                      draftTotal / rounds * 1000,
                      verifyTotal / rounds * 1000,
                      sampleTotal / rounds * 1000,
                      roundCount,
                      draftGpuTotal / rounds * 1000,
                      verifyGpuTotal / rounds * 1000,
                      emitTotal / rounds * 1000,
                      host / rounds * 1000,
                      wallTotal / rounds * 1000)
    }
}
