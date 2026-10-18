import Foundation

/// Pure-function memory budget policy. No llama C API, no system calls —
/// callers pass in `ProcessInfo.physicalMemory`, `os_proc_available_memory()`
/// and the process's current physical footprint, and get back the byte budget
/// to plan model + KV + compute against.
///
/// Why not physical RAM: iOS jetsam-kills (SIGKILL, uncatchable) any process
/// whose footprint exceeds its *per-process* limit, which is well below total
/// RAM. Budgeting against physical RAM plans memory the process can never
/// hold, so a big enough model+KV config crosses the ceiling with no log and
/// no graceful error.
///
/// Why not raw `os_proc_available_memory()` either: it reports **headroom, not
/// the ceiling** — it is the per-process limit minus the caller's *current*
/// footprint. At engine init the previous engine can still be resident (freed,
/// but its pages not yet reclaimed), so the reading can sit far below the
/// device's real allowance. Measured on the 12GB test device (2026-10-18):
/// 6.37e9 bytes reported with a 2.89e9 footprint, while `available + rss` was a
/// constant 9.06–9.26e9 across live requests — i.e. a ~9.26e9 ceiling, which is
/// the `com.apple.developer.kernel.increased-memory-limit` uplift. Reading the
/// headroom alone let the 55% floor win and computed 6.21e9, discarding a third
/// of the allowance. `processLimit` adds the footprint back to recover the
/// ceiling; the floor remains as the safety net for a zero/garbage reading, and
/// nothing ever exceeds physical RAM.
struct MemoryBudget {

    /// Transient headroom held back from the budget: request buffers,
    /// autorelease churn, Metal setup spikes. Also capped at 1/8 of the
    /// available allowance so tiny budgets (e.g. 800MB) stay usable.
    private static let transientHeadroomCap = 512 * 1024 * 1024

    /// Fraction of physical RAM the budget never falls below, even when the
    /// measured ceiling reports less.
    private static let floorPercent = 55

    /// Reconstructs the per-process ceiling (what jetsam will allow) from
    /// `os_proc_available_memory()`.
    ///
    /// The API returns `ceiling − footprint`, so the ceiling is the sum. This is
    /// where the increased-memory-limit entitlement actually shows up: the same
    /// API returns ~3.3e9-class readings on an un-entitled install and
    /// ~6.4e9-class headroom (→ ~9.3e9 ceiling) on this entitled one.
    ///
    /// - Parameters:
    ///   - physicalRAM: total physical memory (`ProcessInfo.physicalMemory`).
    ///   - available: `os_proc_available_memory()` in bytes. 0 means "unknown"
    ///     (caller is not an app, or the API failed).
    ///   - footprint: current `task_vm_info.phys_footprint` in bytes; 0 means
    ///     unknown. A negative value is treated as unknown.
    /// - Returns: `min(physicalRAM, available + max(0, footprint))`, or 0 when
    ///   `available` is unknown (the floor then applies).
    static func processLimit(physicalRAM: Int, available: Int, footprint: Int) -> Int {
        guard available > 0 else { return 0 }
        return min(physicalRAM, available + max(0, footprint))
    }

    /// Returns the memory budget in bytes.
    ///
    /// - Parameters:
    ///   - physicalRAM: Total physical memory in bytes (`ProcessInfo.physicalMemory`).
    ///   - processLimit: the per-process ceiling, from `processLimit(physicalRAM:available:footprint:)`.
    /// - Returns: `min(physicalRAM, max(processLimit, physicalRAM * 55/100))`
    ///   minus transient headroom (`min(512MB, ceiling/8)`).
    static func budgetBytes(physicalRAM: Int, processLimit: Int) -> Int {
        let floor = physicalRAM * floorPercent / 100
        let ceiling = min(physicalRAM, max(processLimit, floor))
        let headroom = min(transientHeadroomCap, ceiling / 8)
        return ceiling - headroom
    }

    /// Start-time memory snapshot log line. Raw byte counts come first,
    /// verbatim (the exact `os_proc_available_memory()` / budget values, so a
    /// byte-for-byte A/B across toolchain or entitlement changes is possible);
    /// the rounded `ByteCountFormatter` rendering follows in parentheses for
    /// humans reading the log on-device. Headroom *and* footprint are both kept
    /// raw: the ceiling is derived from them, so a future disagreement about
    /// either input stays diagnosable from this one line.
    static func logLine(budget: Int, processLimit: Int,
                        processAvailable: Int, footprint: Int,
                        physicalRAM: Int) -> String {
        let fmt = { (bytes: Int) in ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file) }
        return "memory budget \(budget)B (\(fmt(budget))) [limit \(processLimit)B (\(fmt(processLimit)))"
            + " = process-available \(processAvailable)B (\(fmt(processAvailable)))"
            + " + footprint \(footprint)B (\(fmt(footprint))),"
            + " physical \(physicalRAM)B (\(fmt(physicalRAM)))]"
    }
}
