import Foundation

/// Pure-function memory budget policy. No llama C API, no system calls —
/// callers pass in `ProcessInfo.physicalMemory` and `os_proc_available_memory()`
/// and get back the byte budget to plan model + KV + compute against.
///
/// Why not physical RAM: iOS jetsam-kills (SIGKILL, uncatchable) any process
/// whose footprint exceeds its *per-process* limit, which is well below total
/// RAM (measured: 12GB hardware with a ~6.1GB ceiling on iOS 27). Budgeting
/// against physical RAM plans memory the process can never hold, so a big
/// enough model+KV config crosses the ceiling with no log and no graceful
/// error.
///
/// Why not raw `os_proc_available_memory()` either: it can under-report at
/// init (observed: ~3.46e9 bytes on the 12GB test device, while the working
/// 2.78GB-model + 320MB-reserve config that had been loading and running fine
/// needs ~3.10e9) — gating on the raw reading refused every start. The
/// 55%-of-physical floor guarantees such a working config still passes, while
/// a *larger* reported value (entitled uplift) is still trusted, and nothing
/// ever exceeds physical RAM.
struct MemoryBudget {

    /// Transient headroom held back from the budget: request buffers,
    /// autorelease churn, Metal setup spikes. Also capped at 1/8 of the
    /// available allowance so tiny budgets (e.g. 800MB) stay usable.
    private static let transientHeadroomCap = 512 * 1024 * 1024

    /// Fraction of physical RAM the budget never falls below, even when
    /// `os_proc_available_memory()` reports less.
    private static let floorPercent = 55

    /// Returns the memory budget in bytes.
    ///
    /// - Parameters:
    ///   - physicalRAM: Total physical memory in bytes (`ProcessInfo.physicalMemory`).
    ///   - processAvailable: `os_proc_available_memory()` in bytes — how much more
    ///     this process may allocate before jetsam kills it. 0 means "unknown"
    ///     (caller is not an app, or API failed); the floor applies then too.
    /// - Returns: `min(physicalRAM, max(processAvailable, physicalRAM * 55/100))`
    ///   minus transient headroom (`min(512MB, ceiling/8)`).
    static func budgetBytes(physicalRAM: Int, processAvailable: Int) -> Int {
        let floor = physicalRAM * floorPercent / 100
        let ceiling = min(physicalRAM, max(processAvailable, floor))
        let headroom = min(transientHeadroomCap, ceiling / 8)
        return ceiling - headroom
    }
}
