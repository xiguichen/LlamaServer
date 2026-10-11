import Foundation

/// Pure startup-phase timing accumulator — the logic behind the startup
/// load-gap instrumentation (#6). Records named phases with durations the
/// caller measured (`Date()` deltas) and formats a single breakdown line, so
/// the on-device log says *where* the tap-to-listening gap went (model load vs
/// context create vs MTP draft vs HTTP server) instead of just "loading…".
///
/// No clocks and no llama C API live here — the caller injects the seconds —
/// so aggregation, rounding, and formatting are unit-testable (see
/// StartupTimerTests), exactly like `MemoryBudget` (pure policy, caller passes
/// the system readings).
struct StartupTimer {

    struct Phase: Equatable {
        let name: String
        let seconds: TimeInterval
    }

    private(set) var phases: [Phase] = []

    /// Adds a phase. A zero/near-zero duration is real data (it happened fast)
    /// and is kept; a negative delta is clock skew and is clamped to 0 so it
    /// can never show up as a negative phase or drag the total below its parts.
    mutating func record(_ name: String, seconds: TimeInterval) {
        phases.append(Phase(name: name, seconds: max(0, seconds)))
    }

    /// Sum of all recorded phases, in seconds (from the unrounded values, so it
    /// never drifts from the individual parts).
    var totalSeconds: TimeInterval {
        phases.reduce(0) { $0 + $1.seconds }
    }

    /// Human-readable breakdown, e.g. `model load 3.50s + context 0.25s = 3.75s`.
    /// Each phase and the total are rounded to 2 decimals for on-device reading.
    func summary() -> String {
        guard !phases.isEmpty else { return "no startup phases recorded" }
        let parts = phases.map { "\($0.name) \(String(format: "%.2f", $0.seconds))s" }
        return parts.joined(separator: " + ") + " = \(String(format: "%.2f", totalSeconds))s"
    }
}
