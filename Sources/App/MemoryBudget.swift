import Foundation

/// Pure-function memory budget policy. No llama C API, no system calls —
/// callers pass in `ProcessInfo.physicalMemory` and `os_proc_available_memory()`
/// and get back the byte budget to plan model + KV + compute against.
///
/// Why not physical RAM: iOS jetsam-kills (SIGKILL, uncatchable) any process
/// whose footprint exceeds its *per-process* limit, which is well below total
/// RAM (measured: 12GB hardware with a ~6.1GB ceiling on iOS 27 without the
/// increased-memory-limit entitlement). Budgeting against physical RAM plans
/// memory the process can never hold, so a big enough model+KV config crosses
/// the ceiling with no log and no graceful error.
struct MemoryBudget {

    /// Transient headroom held back from the budget: request buffers,
    /// autorelease churn, Metal setup spikes. Also capped at 1/8 of the
    /// available allowance so tiny budgets (e.g. 800MB) stay usable.
    private static let transientHeadroomCap = 512 * 1024 * 1024

    /// Returns the memory budget in bytes.
    ///
    /// - Parameters:
    ///   - physicalRAM: Total physical memory in bytes (`ProcessInfo.physicalMemory`).
    ///   - processAvailable: `os_proc_available_memory()` in bytes — how much more
    ///     this process may allocate before jetsam kills it. Pass 0 when the API
    ///     reports "unknown" (caller is not an app, or already over its limit);
    ///     the budget then falls back to physical RAM (legacy behavior).
    /// - Returns: `min(processAvailable, physicalRAM)` minus transient headroom
    ///   (`min(512MB, ceiling/8)`).
    static func budgetBytes(physicalRAM: Int, processAvailable: Int) -> Int {
        let ceiling = processAvailable > 0 ? min(processAvailable, physicalRAM) : physicalRAM
        let headroom = min(transientHeadroomCap, ceiling / 8)
        return ceiling - headroom
    }
}
