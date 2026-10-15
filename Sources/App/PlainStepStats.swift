import Foundation

/// Pure per-step timing accumulator for the plain (non-MTP) decode loop —
/// the plain-path sibling of `MtpPhaseStats`. One step is one loop
/// iteration: wait for the previous decode's GPU work (`llama_synchronize`),
/// sample, then emit text and enqueue the next decode. The caller measures
/// the deltas and injects them, so aggregation and formatting are
/// unit-testable (see PlainStepStatsTests).
///
/// `gpuWait` is the GPU time the host did not hide, `sample` is CPU
/// sampling, and `host` is the rest of the step's wall clock (text, stop
/// strings, streaming, decode enqueue), so the phases sum to `wall`.
struct PlainStepStats {

    private var gpuWaitTotal: TimeInterval = 0
    private var sampleTotal: TimeInterval = 0
    private var wallTotal: TimeInterval = 0

    /// Number of decode-loop steps recorded.
    private(set) var stepCount: Int = 0

    /// Adds one step's durations. Negative deltas (clock skew) clamp to 0,
    /// the same rule as `MtpPhaseStats` and `StartupTimer`.
    mutating func record(gpuWait: TimeInterval, sample: TimeInterval, wall: TimeInterval) {
        gpuWaitTotal += max(0, gpuWait)
        sampleTotal += max(0, sample)
        wallTotal += max(0, wall)
        stepCount += 1
    }

    /// Per-step averages in milliseconds, e.g.
    /// `gpu-wait 43.10ms/step + sample 0.30ms/step + host 0.40ms/step = wall 43.80ms/step (128 steps)`.
    func summary() -> String {
        guard stepCount > 0 else { return "no plain steps recorded" }
        let steps = Double(stepCount)
        let host = max(0, wallTotal - gpuWaitTotal - sampleTotal)
        return String(format: "gpu-wait %.2fms/step + sample %.2fms/step + host %.2fms/step = wall %.2fms/step (%d steps)",
                      gpuWaitTotal / steps * 1000,
                      sampleTotal / steps * 1000,
                      host / steps * 1000,
                      wallTotal / steps * 1000,
                      stepCount)
    }
}
