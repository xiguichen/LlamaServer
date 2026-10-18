import XCTest
@testable import LlamaServer

/// The probe exists to answer one question from a device log: *why* is the
/// per-process ceiling this size, and is the entitlement that raises it
/// actually carried by the re-signed binary in the field. The line must stay
/// greppable and must never claim a measurement it did not take.
final class MemoryLimitProbeTests: XCTestCase {
    func testLineReportsBaseLimitCeilingAndEntitlement() {
        let line = MemoryLimitProbe.line(baseLimitMB: 6144,
                                         ceilingBytes: 6_442_450_944,
                                         entitlement: true)
        XCTAssertEqual(line,
                       "memory limit probe: base per-process limit 6144 MB (kern.max_task_pmem), "
                       + "measured ceiling 6442450944B, "
                       + "increased-memory-limit entitlement present")
    }

    /// The case that would prove the install path dropped the entitlement.
    func testLineSaysAbsentWhenEntitlementIsMissing() {
        let line = MemoryLimitProbe.line(baseLimitMB: 3072,
                                         ceilingBytes: 3_221_225_472,
                                         entitlement: false)
        XCTAssertTrue(line.contains("entitlement ABSENT"), line)
        XCTAssertFalse(line.contains("present"), line)
    }

    /// `SecTask` is resolved with dlsym; if the symbol is gone the probe must
    /// say so rather than report an absence it cannot see.
    func testLineDistinguishesUnknownFromAbsent() {
        let unknown = MemoryLimitProbe.line(baseLimitMB: 6144, ceilingBytes: 6_442_450_944, entitlement: nil)
        XCTAssertTrue(unknown.contains("unknown (SecTask unavailable)"), unknown)
        XCTAssertFalse(unknown.contains("ABSENT"), unknown)
    }

    /// A sandbox can deny `kern.max_task_pmem`; that is not a zero limit.
    func testLineReportsUnreadableBaseLimit() {
        let line = MemoryLimitProbe.line(baseLimitMB: nil, ceilingBytes: nil, entitlement: true)
        XCTAssertTrue(line.contains("base per-process limit unreadable (kern.max_task_pmem)"), line)
        XCTAssertTrue(line.contains("measured ceiling unreadable"), line)
    }

    /// The probe line must name the entitlement the kernel actually reads
    /// (`memorystatus_task_has_increased_memory_limit_entitlement`).
    func testEntitlementNameIsTheKernelOne() {
        XCTAssertEqual(MemoryLimitProbe.entitlementName,
                       "com.apple.developer.kernel.increased-memory-limit")
    }
}
