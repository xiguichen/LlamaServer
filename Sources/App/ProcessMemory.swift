import Foundation

/// The two process-level memory numbers `MemoryBudget` reasons about, wrapped
/// away from the policy so it stays pure and unit-testable on the simulator.
///
/// - `availableBytes()` is `os_proc_available_memory()`: the bytes this process
///   may still allocate before jetsam kills it. It is the per-process ceiling
///   **minus the current footprint**, so on its own it under-reports the
///   ceiling — `MemoryBudget.processLimit` adds the footprint back.
/// - `footprintBytes()` is `task_vm_info.phys_footprint`, the number jetsam
///   actually accounts against (it is not the same as RSS).
enum ProcessMemory {

    /// Bytes this process may still allocate before jetsam kills it.
    /// 0 means the reading failed (the API is app-only) — never a ceiling.
    static func availableBytes() -> Int {
        Int(os_proc_available_memory())
    }

    /// Current physical footprint in bytes (0 if the Mach call fails).
    ///
    /// Non-zero at engine init precisely when a previous engine's pages are
    /// still resident, which is the case a headroom-only reading gets wrong.
    static func footprintBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    /// The samples a settled-ceiling decision is made from, and the plateau
    /// length that decision should require.
    struct SettledReadings {
        let readings: [MemoryBudget.Reading]
        let stableSamples: Int
    }

    /// Samples the per-process ceiling until it has held still for
    /// `stableSamples` consecutive readings, or `maxWait` elapses.
    ///
    /// Why this exists: iOS 27 raises the 12GB A19 Pro per-process tier from
    /// 6GiB to 8GiB when Game Mode activates, a few seconds after a
    /// Game-Mode-capable app is foregrounded. A single reading at engine init
    /// can catch the pre-uplift 6GiB state and then never see the tier the app
    /// will actually run in. Sampling after the model load (which already takes
    /// ~2.3s) gives Game Mode time to turn on; the plateau requirement lets
    /// `MemoryBudget.settledCeiling` distinguish that real rise from a stale
    /// one-off reading.
    ///
    /// Bounded on purpose: a stable tier (including "Game Mode is off") exits
    /// after ~0.5s, and a read that keeps failing exits after the same number
    /// of samples instead of waiting out `maxWait`.
    static func settleCeiling(physicalRAM: Int,
                              label: String,
                              maxWait: TimeInterval = 5.0,
                              interval: TimeInterval = 0.25,
                              stableSamples: Int = 3) -> SettledReadings {
        let required = max(1, stableSamples)
        var readings: [MemoryBudget.Reading] = []
        let start = Date()
        var lastLimit = 0
        var stableRun = 0
        var failedRun = 0

        while true {
            let elapsed = Date().timeIntervalSince(start)
            let sampleLabel = readings.isEmpty
                ? label
                : "\(label) +" + String(format: "%.2fs", elapsed)
            let reading = MemoryBudget.Reading(sampleLabel,
                                               available: availableBytes(),
                                               footprint: footprintBytes())
            readings.append(reading)

            let limit = MemoryBudget.processLimit(physicalRAM: physicalRAM,
                                                  available: reading.available,
                                                  footprint: reading.footprint)
            if limit > 0 {
                failedRun = 0
                if limit == lastLimit {
                    stableRun += 1
                } else {
                    lastLimit = limit
                    stableRun = 1
                }
            } else {
                failedRun += 1
                stableRun = 0
                lastLimit = 0
            }

            if stableRun >= required || failedRun >= required || elapsed >= maxWait {
                break
            }
            Thread.sleep(forTimeInterval: interval)
        }
        return SettledReadings(readings: readings, stableSamples: required)
    }
}
