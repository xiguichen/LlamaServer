import Foundation
import Darwin

/// Reports the OS's own per-process memory limit and whether the entitlement
/// that raises it is really carried by the running binary.
///
/// Why this exists: the measured ceiling (`os_proc_available_memory() +
/// phys_footprint`, see `MemoryBudget`) tells us *what* the limit is, never
/// *why*. Two very different causes produce a small ceiling — the device's own
/// per-process tier, or a re-signer that dropped the entitlement we ship — and
/// the log has been unable to tell them apart (2026-10-18: the "un-entitled
/// ~3.3GB class" the notes carried was an assumption, never a reading).
/// Reading the kernel's own base limit separates them in one line.
///
/// The kernel side (xnu `bsd/kern/kern_memorystatus.c`):
///
/// * `memorystatus_get_default_task_active_limit()` computes
///   `limit = MAX(max_task_footprint_mb, memorystatus_entitled_max_task_footprint_mb)`
///   when `IOTaskHasEntitlement(task, "com.apple.developer.kernel.increased-memory-limit")`
///   passes. So the entitlement only *unlocks* `memorystatus_entitled_max_task_footprint_mb`.
/// * That value comes from the device tree (`TUNABLE_DT_WRITEABLE`, boot-arg
///   `entitled_max_task_pmem`), i.e. **Apple bakes it per device model** — it is
///   not derived from RAM at runtime and no app-side choice can exceed it. On
///   iPhone 17 Pro Max it lands on exactly 6 GiB (6,442,450,944B), matching an
///   independent report from the same model (6144 MB with the entitlement).
/// * `max_task_footprint_mb` is published as `kern.max_task_pmem`, so the base
///   limit is readable from the app without private entitlements. When the
///   measured ceiling sits *above* it, the entitlement is doing work; when they
///   are equal, the entitlement is decorative on that device (the base tier
///   already is the entitled tier).
///
/// A third knob exists — `com.apple.developer.kernel.increased-debugging-memory-limit`
/// resolves to the *whole* DRAM config when the device tree sets
/// `entitled_max_developer_task_pmem = -1` — but it needs a provisioning profile
/// that carries it, which sideloaded re-signs do not have. Not chased here.
enum MemoryLimitProbe {
    static let entitlementName = "com.apple.developer.kernel.increased-memory-limit"

    /// `kern.max_task_pmem` — the device's base per-process footprint limit in
    /// MB, i.e. `max_task_footprint_mb`. `nil` when the sandbox denies the read
    /// (the sysctl is `CTLFLAG_RD`, but a deny is not a zero limit).
    static func baseLimitMB() -> Int32? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("kern.max_task_pmem", &value, &size, nil, 0) == 0, value > 0 else {
            return nil
        }
        return value
    }

    /// Whether this process's *effective* entitlements carry `name`.
    ///
    /// `SecTaskCreateFromSelf` / `SecTaskCopyValueForEntitlement` are the only
    /// way to see the post-re-sign entitlement list, and they are resolved with
    /// `dlsym` so a missing symbol degrades to `nil` ("unknown") instead of
    /// becoming a link failure or a crash on a future SDK. `nil` never means
    /// "absent".
    static func entitlementPresent(_ name: String = entitlementName) -> Bool? {
        guard let handle = dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_NOW),
              let createSym = dlsym(handle, "SecTaskCreateFromSelf"),
              let copySym = dlsym(handle, "SecTaskCopyValueForEntitlement") else {
            return nil
        }
        // The Create/Copy naming means the callee hands back a +1 reference; the
        // `Unmanaged` return types make that ownership explicit (Swift 6 manages
        // CF types itself, so CFRelease is unavailable and would be wrong here).
        typealias CreateTask = @convention(c) (CFAllocator?) -> Unmanaged<CFTypeRef>?
        typealias CopyValue = @convention(c) (CFTypeRef?, CFString,
                                              UnsafeMutablePointer<Unmanaged<CFError>?>?) -> Unmanaged<CFTypeRef>?
        let createTask = unsafeBitCast(createSym, to: CreateTask.self)
        let copyValue = unsafeBitCast(copySym, to: CopyValue.self)
        guard let task = createTask(kCFAllocatorDefault)?.takeRetainedValue() else { return nil }
        guard let value = copyValue(task, name as CFString, nil)?.takeRetainedValue() else {
            return false                              // task found, entitlement not in it
        }
        if let number = value as? NSNumber { return number.boolValue }
        return true                                   // present with a non-boolean payload
    }

    /// The one log line. Pure, so the wording is pinned by tests.
    static func line(baseLimitMB: Int32?, ceilingBytes: Int?, entitlement: Bool?) -> String {
        let base = baseLimitMB.map { "\($0) MB (kern.max_task_pmem)" }
            ?? "unreadable (kern.max_task_pmem)"
        let ceiling = ceilingBytes.map { "\($0)B" } ?? "unreadable"
        let ent: String
        switch entitlement {
        case .some(true):  ent = "present"
        case .some(false): ent = "ABSENT"
        case .none:        ent = "unknown (SecTask unavailable)"
        }
        return "memory limit probe: base per-process limit \(base), "
            + "measured ceiling \(ceiling), "
            + "increased-memory-limit entitlement \(ent)"
    }
}
