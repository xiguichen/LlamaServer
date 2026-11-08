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
/// Why one reading is not enough: three clean samples of 2026-10-18 (build
/// `698dbbf`) all give the *same* ceiling — 6,352,271,400 + 90,179,544 at
/// init, 6,288,683,008 + 153,767,936 after the model load, 4,576,193,192 +
/// 1,866,257,752 idle with both contexts live; each sum is exactly
/// 6,442,450,944B (6GiB). While the app's game-mode state is unchanged, the
/// ceiling is a constant of the launch, not something that grows as the
/// process settles. iOS 27 adds one legitimate exception: Game Mode raises the
/// 12GB A19 Pro tier from 6GiB to 8GiB a few seconds after a
/// Game-Mode-capable app is foregrounded, so a later reading can measure a
/// *different, higher* tier. `settledCeiling` accepts that only as a repeated
/// plateau; a lone high reading is still a stale operand. Within one state,
/// when readings disagree, the choice is asymmetric: **the smallest successful
/// reading wins**, because planning *above* the real ceiling is a jetsam
/// SIGKILL while planning below it only costs context tokens. The ~9GB class
/// these notes once chased came from adding `available` to a 4.58GB reading
/// recorded as **rss**; rss is not the ceiling operand, and the policy must
/// never let such a reading inflate the budget.
///
/// What the ceiling covers is just as important as what it does not: the
/// model's file-backed mmap is mostly outside the charged footprint (loading
/// 2.78GB of weights moved it by 63.6MB), but the Metal compute buffers are
/// private storage and **are** part of it. A KV budget derived from this
/// ceiling must therefore reserve the compute buffers it will allocate. The
/// 2026-11-08 q8_0+MTP run at 252,416 tokens measured target+draft
/// `sched_reserve` buffers of ≈3.29GB (MTL0 + CPU) while the plan held back
/// only 320MB, and the first decode hit a GPU command-buffer OOM. The context
/// plan now reserves 1,800/2,600 MiB for the target micro-batch and
/// 3,000/3,800 MiB when MTP's draft context is live. The ceiling is still a
/// **bound on what to attempt**; the reserve is what makes it honest.
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
    /// The API returns `ceiling − footprint`, so the ceiling is the sum. On this
    /// install every sample of 2026-10-18 sums to 6,442,450,944B (6GiB), the
    /// device-tree tier the entitlement unlocks — see the type comment for the
    /// samples. One reading is not the ceiling by itself: `ceiling` reconciles
    /// every reading, and a disagreement can only lower the answer.
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

    /// The conservative ceiling from every reading of this launch.
    ///
    /// While the tier is unchanged every consistent reading measures the same
    /// constant. When they disagree, one of the two operands was read at a
    /// different moment (or is not the footprint at all), and the disagreement
    /// is resolved **conservatively**: the smallest successful reading wins.
    /// The failure modes are asymmetric — a ceiling above the real limit plans
    /// memory the kernel will SIGKILL, while one below it merely clamps the
    /// context. A failed reading (`available == 0`) never wins. Only when every
    /// reading failed does the documented half-of-physical value apply, flagged
    /// by `readingIndex == nil` so callers can log it as an assumption.
    ///
    /// Use `settledCeiling` for the planning decision: it can accept the one
    /// legitimate rise (Game Mode) when the new limit is corroborated by a
    /// stable plateau, while still falling back to this rule when it is not.
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
        var best = Int.max
        var bestIndex: Int?
        for (i, reading) in readings.enumerated() {
            let limit = processLimit(physicalRAM: physicalRAM,
                                     available: reading.available,
                                     footprint: reading.footprint)
            // `processLimit` returns 0 for a failed read; 0 never wins.
            guard limit > 0 else { continue }
            if limit < best {
                best = limit
                bestIndex = i
            }
        }
        guard let index = bestIndex else {
            return Ceiling(bytes: physicalRAM / 2, readingIndex: nil)
        }
        return Ceiling(bytes: best, readingIndex: index)
    }

    /// The ceiling a *planning* decision should use once the launch's tier has
    /// settled.
    ///
    /// iOS 27 can raise the per-process ceiling mid-launch: when Game Mode turns
    /// on (a few seconds after a Game-Mode-capable app is foregrounded), the
    /// 12GB A19 Pro tier rises from 6 GiB to 8 GiB. The pre-uplift reading is a
    /// real reading of the old state, so "smallest wins" would keep planning at
    /// 6 GiB forever. This function accepts a rise only when it is corroborated
    /// by `stableSamples` consecutive readings at the same limit — a plateau.
    /// A single high sample is still rejected and falls back to `ceiling`'s
    /// smallest-successful rule, which is the guard against the 9.17 GB
    /// `available + rss` regression.
    ///
    /// The *last* qualifying plateau wins, so a 6 GiB run followed by a settled
    /// 8 GiB run resolves to 8 GiB. A rise that has not yet held for
    /// `stableSamples` readings (e.g. the sampler timed out mid-change) is not
    /// accepted; the earlier stable run remains the answer.
    static func settledCeiling(physicalRAM: Int, readings: [Reading],
                               stableSamples: Int) -> Ceiling {
        let required = max(1, stableSamples)
        let limits = readings.map {
            processLimit(physicalRAM: physicalRAM,
                         available: $0.available, footprint: $0.footprint)
        }
        var runStart = 0
        var settledIndex: Int?
        for i in limits.indices {
            if i == 0 || limits[i] == 0 || limits[i] != limits[i - 1] {
                runStart = i
            }
            if limits[i] > 0, i - runStart + 1 >= required {
                settledIndex = i
            }
        }
        if let index = settledIndex {
            return Ceiling(bytes: limits[index], readingIndex: index)
        }
        // No plateau: keep the conservative rule (and its fallback).
        return ceiling(physicalRAM: physicalRAM, readings: readings)
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
