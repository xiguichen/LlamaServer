import Foundation

/// Pure-function memory budget policy. No llama C API, no system calls —
/// callers pass in `ProcessInfo.physicalMemory` and one or more readings of
/// `os_proc_available_memory()` + the process's `task_vm_info.phys_footprint`,
/// and get back the byte budget to plan model + KV + compute against.
///
/// Two measured facts drive the whole policy:
///
/// 1. iOS jetsam-kills (SIGKILL, uncatchable) any process whose footprint
///    exceeds its *per-process* limit, which is well below total RAM (on the
///    12.26GB test device the measured allowance is 6.44GB — exactly 6GiB, see
///    below). Budgeting against physical RAM plans memory the process can
///    never hold.
/// 2. `os_proc_available_memory()` reports **headroom, not the ceiling** — it
///    subtracts the caller's current footprint, so the ceiling is
///    `available + footprint`, capped at physical. `available == 0` means the
///    call failed; it never implies a ceiling.
///
/// The single rule on top of that: **the budget follows the device's own
/// measurement, and nothing else.** A 55%-of-physical floor used to override a
/// *successful* measurement whenever the reading came back low, which is how
/// the device came to log a number its own line could not explain (2026-10-18:
/// `memory budget 6208012288B (6.21 GB) [limit 6442450944B (6.44 GB) …]` —
/// 6.21GB is 55% of physical minus headroom, and the floor sat *above* the
/// measured ceiling, so the measurement never got a vote). The floor is gone.
/// Half of physical survives only as the answer when *every* reading failed,
/// and it is reported as a fallback so it can never be mistaken for a
/// measurement.
///
/// Why one reading is not enough: the allowance can rise as the process
/// settles, so the earliest sample can be the smallest — that is why every
/// reading is taken and the largest wins. Three clean samples of 2026-10-18
/// (build `698dbbf`) all give the *same* ceiling — 6,352,271,400 + 90,179,544
/// at init, 6,288,683,008 + 153,767,936 after the model load, 4,576,193,192 +
/// 1,866,257,752 idle with both contexts live; each sum is exactly
/// 6,442,450,944B (6GiB) — so here the policy has nothing to choose between.
/// 6GiB is the **entitled** class (`project.yml`: ~3.3GB un-entitled → ~6GB+
/// entitled), so the uplift is working. What is *not* real is the ~9GB class
/// these notes used to claim: that came from adding `available` to a 4.58GB
/// reading recorded as **rss**, and rss is not the ceiling operand.
///
/// What the ceiling does *not* cover is just as important as what it does: the
/// model's mmap (loading a 2.78GB model moved the footprint by 63.6MB) and the
/// Metal compute buffers (1.23GB of `sched_reserve` buffer lines, private
/// storage) are outside it. A KV budget derived from this ceiling is therefore
/// a **bound on what to attempt**, not a measurement of the engine's total.
struct MemoryBudget {

    /// Transient headroom held back from the budget: request buffers,
    /// autorelease churn, Metal setup spikes. Also capped at 1/8 of the
    /// allowance so tiny budgets (e.g. 800MB) stay usable.
    private static let transientHeadroomCap = 512 * 1024 * 1024

    /// One reading of `os_proc_available_memory()` + the footprint, labelled
    /// with when it was taken so the log can name the winner.
    struct Reading {
        let label: String
        let available: Int
        let footprint: Int

        init(_ label: String, available: Int, footprint: Int) {
            self.label = label
            self.available = available
            self.footprint = footprint
        }
    }

    /// Reconstructs the per-process ceiling from one reading.
    ///
    /// The API returns `ceiling − footprint`, so the ceiling is the sum. This is
    /// where the increased-memory-limit entitlement shows up, as a larger sum:
    /// on this install every sample of 2026-10-18 sums to 6,442,450,944B (6GiB),
    /// the ~6GB+ entitled class the entitlement is documented to grant (the
    /// un-entitled limit is ~3.3GB) — see the type comment for the samples.
    ///
    /// - Parameters:
    ///   - physicalRAM: total physical memory (`ProcessInfo.physicalMemory`).
    ///   - available: `os_proc_available_memory()` in bytes. 0 means "unknown"
    ///     (the call failed, or the caller is not an app) and yields 0 — never
    ///     an invented entitlement.
    ///   - footprint: current `task_vm_info.phys_footprint` in bytes; 0 means
    ///     unknown. A negative value is treated as unknown.
    /// - Returns: `min(physicalRAM, available + max(0, footprint))`, or 0 when
    ///   `available` is unknown.
    static func processLimit(physicalRAM: Int, available: Int, footprint: Int) -> Int {
        guard available > 0 else { return 0 }
        return min(physicalRAM, available + max(0, footprint))
    }

    /// The ceiling to plan against, chosen from every reading of this launch.
    ///
    /// The largest *successful* reading wins: the allowance rises as the process
    /// grows, so the earliest sample is the smallest. Nothing here may raise the
    /// ceiling above what the device reported, and a failed reading
    /// (`available == 0`) never wins. Only when every reading failed does the
    /// documented un-entitled half-of-physical value apply, flagged by
    /// `readingIndex == nil` so callers can log it as an assumption.
    struct Ceiling {
        /// The ceiling in bytes.
        let bytes: Int
        /// Which reading produced it; `nil` when every reading failed.
        let readingIndex: Int?
        /// True when no reading succeeded and half of physical was assumed.
        var usedFallback: Bool { readingIndex == nil }

        /// The label of the reading that won, for logging.
        func source(_ readings: [Reading]) -> String {
            guard let i = readingIndex, readings.indices.contains(i) else {
                return "fallback (every reading failed)"
            }
            return readings[i].label
        }
    }

    static func ceiling(physicalRAM: Int, readings: [Reading]) -> Ceiling {
        var best = 0
        var bestIndex: Int?
        for (i, reading) in readings.enumerated() {
            let limit = processLimit(physicalRAM: physicalRAM,
                                     available: reading.available,
                                     footprint: reading.footprint)
            if limit > best {
                best = limit
                bestIndex = i
            }
        }
        guard let index = bestIndex else {
            return Ceiling(bytes: physicalRAM / 2, readingIndex: nil)
        }
        return Ceiling(bytes: best, readingIndex: index)
    }

    /// Returns the memory budget in bytes: the ceiling minus transient headroom
    /// (`min(512MB, ceiling/8)`).
    ///
    /// - Parameters:
    ///   - physicalRAM: total physical memory in bytes.
    ///   - ceiling: the ceiling from `ceiling(physicalRAM:readings:)`.
    static func budgetBytes(physicalRAM: Int, ceiling: Int) -> Int {
        let capped = min(physicalRAM, max(0, ceiling))
        let headroom = min(transientHeadroomCap, capped / 8)
        return capped - headroom
    }

    /// Start-time memory snapshot log line. Each reading's raw inputs come
    /// first, verbatim (so a byte-for-byte A/B across toolchain, entitlement or
    /// sample-point changes is possible), then which reading won and the budget
    /// derived from it. The rounded `ByteCountFormatter` rendering follows in
    /// parentheses for humans reading the log on-device. This one line has to
    /// justify its own number: the budget is `min(ceiling, physical) − headroom`.
    static func logLine(budget: Int, ceiling: Ceiling, readings: [Reading],
                        physicalRAM: Int) -> String {
        let fmt = { (bytes: Int) in ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file) }
        let rendered = readings.map { reading -> String in
            let limit = processLimit(physicalRAM: physicalRAM,
                                     available: reading.available,
                                     footprint: reading.footprint)
            return "\(reading.label): available \(reading.available)B (\(fmt(reading.available)))"
                + " + footprint \(reading.footprint)B (\(fmt(reading.footprint)))"
                + " -> \(limit)B (\(fmt(limit)))"
        }
        return "memory budget \(budget)B (\(fmt(budget)))"
            + " [ceiling \(ceiling.bytes)B (\(fmt(ceiling.bytes))) from \(ceiling.source(readings))"
            + "; " + rendered.joined(separator: "; ")
            + "; physical \(physicalRAM)B (\(fmt(physicalRAM)))]"
    }
}
