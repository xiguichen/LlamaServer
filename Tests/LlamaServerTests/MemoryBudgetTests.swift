import XCTest

/// Budget follows the per-process jetsam limit but never falls below a
/// 55%-of-physical floor.
///
/// Three measured facts pin this policy:
/// - A 12GB device's per-process ceiling sits well below total RAM, so
///   budgeting against full physical RAM plans memory the process can never
///   hold and jetsam SIGKILLs the app with no log.
/// - `os_proc_available_memory()` reports **headroom, not the ceiling**: it
///   subtracts the caller's current footprint. At engine init the just-freed
///   previous engine can still be resident, so the raw reading under-reports
///   the device's real allowance (observed: 3.46e9 bytes, below what a working
///   2.78GB-model + 320MB-reserve config needed). `processLimit` reconstructs
///   the ceiling; the 55% floor stays as the safety net for a zero/garbage
///   reading.
/// - The ceiling really is raised by `com.apple.developer.kernel.increased-memory-limit`:
///   measured 2026-10-18 on the test device as a constant
///   `available + rss ≈ 9.06–9.26 GB` across live requests with rss
///   5.13 → 5.55 GB. Reading `available` alone (6.37GB at init) and letting the
///   55% floor win computed 6.21GB — a third of the allowance thrown away, and
///   which is why a viable context could be clamped.
final class MemoryBudgetTests: XCTestCase {
    private let GB = 1024 * 1024 * 1024
    private let MB = 1024 * 1024

    /// Raw values from the device's own budget log line (2026-10-18 14:52:18):
    /// `memory budget 6208012288B (6.21 GB) [process-available 6368557120B,
    /// physical 12263424000B]`. The footprint is the reconstruction
    /// (limit − available) the constant per-request `available + rss` implies.
    private let devicePhysical = 12_263_424_000
    private let deviceAvailable = 6_368_557_120
    private let deviceFootprint = 2_890_000_000

    // MARK: - The ceiling, reconstructed from headroom + footprint

    func testLimitIsAvailablePlusFootprint() {
        // os_proc_available_memory() = ceiling − footprint, so the ceiling is
        // their sum. This is the number the entitlement actually grants.
        XCTAssertEqual(
            MemoryBudget.processLimit(physicalRAM: devicePhysical,
                                      available: deviceAvailable,
                                      footprint: deviceFootprint),
            9_258_557_120)
    }

    func testLimitIsZeroWhenAvailableIsUnknown() {
        // 0 = API unknown (not an app, or the call failed). The floor handles it;
        // the limit must not invent an entitlement from nothing.
        XCTAssertEqual(MemoryBudget.processLimit(physicalRAM: GB, available: 0, footprint: 3 * GB), 0)
    }

    func testLimitIsCappedAtPhysicalRAM() {
        // A footprint larger than the reported headroom must never inflate the
        // ceiling past the hardware.
        XCTAssertEqual(
            MemoryBudget.processLimit(physicalRAM: 4 * GB, available: 3 * GB, footprint: 9 * GB),
            4 * GB)
    }

    func testNegativeFootprintIsIgnored() {
        XCTAssertEqual(
            MemoryBudget.processLimit(physicalRAM: 8 * GB, available: 2 * GB, footprint: -5),
            2 * GB)
    }

    // MARK: - The regression: the entitled ceiling must drive the budget

    func testEntitledCeilingIsUsedNotTheMomentaryHeadroom() {
        // Old policy, for contrast: it fed the *headroom* in as if it were the
        // ceiling, so the 55% floor (6.74GB) won and the entitlement uplift was
        // discarded — exactly the 6,208,012,288B the device logged.
        let floorWins = MemoryBudget.budgetBytes(physicalRAM: devicePhysical,
                                                 processLimit: deviceAvailable)
        XCTAssertEqual(floorWins, 6_208_012_288,
                       "pinning the measured regression: headroom alone yields the floor")

        // New policy: reconstruct the ceiling first, then budget against it.
        let limit = MemoryBudget.processLimit(physicalRAM: devicePhysical,
                                              available: deviceAvailable,
                                              footprint: deviceFootprint)
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, processLimit: limit)
        XCTAssertEqual(budget, limit - 512 * MB)
        XCTAssertGreaterThan(budget, 8_000_000_000,
                             "the device grants ~9.26GB; the budget must not throw a third of it away")
    }

    func testEntitledBudgetFitsTheConfiguredKVWithHeadroomToSpare() {
        // The consequence that matters: at 131,072 context and 17,408 B/token the
        // q8_0 KV is 2.28GB. Under the old 6.21GB budget only ~0.78GB of slack
        // was left, so a slightly bigger model or KV type got clamped.
        let limit = MemoryBudget.processLimit(physicalRAM: devicePhysical,
                                              available: deviceAvailable,
                                              footprint: deviceFootprint)
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, processLimit: limit)
        let kvBudget = budget - 2_780_000_000 - 320 * MB
        XCTAssertGreaterThan(kvBudget, 131_072 * 17_408 + 1024 * MB,
                             "131,072-token q8_0 KV must leave real slack, not ~0.8GB")
    }

    // MARK: - The floor still protects a bad reading

    func testUnderReportedAvailableFallsToFloor() {
        let physical = 12 * GB
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processLimit: 3_460_000_000)
        XCTAssertEqual(budget, floor - 512 * MB,
                       "a low reading must not shrink the budget below the 55% floor")
        // The guarantee that matters: the previously-working config fits.
        let need = 2_780_000_000 + 320 * MB
        XCTAssertLessThanOrEqual(need, budget,
                                 "2.78GB model + 320MB reserve must fit the budget")
    }

    func testZeroLimitFallsToFloor() {
        // 0 = API unknown; floor, not full physical RAM.
        let physical = 12 * GB
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processLimit: 0)
        XCTAssertEqual(budget, floor - 512 * MB)
    }

    // MARK: - Healthy readings still win

    func testHealthyLimitAboveFloorWins() {
        let physical = 12 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processLimit: 9_000_000_000)
        XCTAssertEqual(budget, 9_000_000_000 - 512 * MB,
                       "a healthy ceiling must not be capped down to the floor")
    }

    // MARK: - Defensive cap

    func testLimitAbovePhysicalIsCappedAtPhysical() {
        // The ceiling should never exceed physical RAM, but a guard keeps a bad
        // value from inflating the budget past hardware.
        let physical = 4 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processLimit: 8 * GB)
        XCTAssertEqual(budget, physical - 512 * MB)
    }

    // MARK: - Start-log line

    func testLogLineCarriesRawByteValues() {
        // Raw bytes must appear verbatim — the rounded rendering alone can't be
        // compared byte-for-byte across toolchain/entitlement changes. The raw
        // headroom AND the footprint are both kept: the ceiling is derived, and
        // a future disagreement about either input stays diagnosable from the log
        // alone.
        let line = MemoryBudget.logLine(budget: 8_721_686_208,
                                        processLimit: 9_258_557_120,
                                        processAvailable: 6_368_557_120,
                                        footprint: 2_890_000_000,
                                        physicalRAM: 12_263_424_000)
        XCTAssertTrue(line.contains("memory budget 8721686208B"), line)
        XCTAssertTrue(line.contains("limit 9258557120B"), line)
        XCTAssertTrue(line.contains("process-available 6368557120B"), line)
        XCTAssertTrue(line.contains("footprint 2890000000B"), line)
        XCTAssertTrue(line.contains("physical 12263424000B"), line)
    }

    // MARK: - Headroom scales down for small budgets

    func testHeadroomShrinksWhenBudgetIsSmall() {
        // On a tight budget the 512MB cap must not eat more than 1/8 of the
        // allowance, or small-budget devices could never load anything.
        // physical 1600MB → floor 880MB; a 800MB ceiling is below the floor,
        // so ceiling = floor and headroom = ceiling/8 (not the 512MB cap).
        let physical = 1600 * MB
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processLimit: 800 * MB)
        XCTAssertEqual(budget, floor - floor / 8)
    }
}
