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
}
