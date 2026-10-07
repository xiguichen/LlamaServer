import XCTest

/// Budget follows the per-process jetsam limit but never falls below a
/// 55%-of-physical floor.
///
/// Two measured facts pin this policy:
/// - A 12GB device's per-process ceiling sits well below total RAM (~6.1GB
///   measured as rss 4148MB + available 1960MB), so budgeting against full
///   physical RAM plans memory the process can never hold and jetsam SIGKILLs
///   the app with no log.
/// - `os_proc_available_memory()` can under-report at init (~3.46e9 bytes on
///   the same device) below what a previously-working config needs
///   (2.78GB model + 320MB reserve ≈ 3.10e9) — gating on the raw reading
///   refused every start. The floor keeps that config loading.
final class MemoryBudgetTests: XCTestCase {
    private let GB = 1024 * 1024 * 1024
    private let MB = 1024 * 1024

    // MARK: - The regression: under-reported available must fall to the floor

    func testUnderReportedAvailableFallsToFloor() {
        let physical = 12 * GB
        let available = 3_460_000_000   // observed at init on the test device
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, floor - 512 * MB,
                       "a low process-available reading must not shrink the budget below the 55% floor")
        // The guarantee that matters: the previously-working config fits.
        let need = 2_780_000_000 + 320 * MB
        XCTAssertLessThanOrEqual(need, budget,
                                 "2.78GB model + 320MB reserve must fit the budget")
    }

    func testZeroAvailableFallsToFloor() {
        // 0 = API unknown; floor, not full physical RAM.
        let physical = 12 * GB
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: 0)
        XCTAssertEqual(budget, floor - 512 * MB)
    }

    // MARK: - Healthy readings still win

    func testHealthyAvailableAboveFloorWins() {
        let physical = 12 * GB
        let available = 9_000_000_000   // above the floor
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, available - 512 * MB,
                       "a healthy process-available reading must not be capped down to the floor")
    }

    func testEntitledUpliftPassesThrough() {
        // Entitled installs report more (iOS 27 uplift ~8.6e9); trust it.
        let physical = 12 * GB
        let available = 8_600_000_000
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, available - 512 * MB)
    }

    // MARK: - Defensive cap

    func testAvailableAbovePhysicalIsCappedAtPhysical() {
        // os_proc_available_memory() should never exceed physical RAM, but a
        // guard keeps a bad value from inflating the budget past hardware.
        let physical = 4 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: 8 * GB)
        XCTAssertEqual(budget, physical - 512 * MB)
    }

    // MARK: - Start-log line

    func testLogLineCarriesRawByteValues() {
        // Raw os_proc_available_memory() bytes must appear verbatim — the
        // rounded rendering alone can't be compared byte-for-byte across
        // toolchain/entitlement changes.
        let line = MemoryBudget.logLine(budget: 6_549_825_126,
                                        processAvailable: 3_460_000_000,
                                        physicalRAM: 12 * GB)
        XCTAssertTrue(line.contains("memory budget 6549825126B"), line)
        XCTAssertTrue(line.contains("process-available 3460000000B"), line)
        XCTAssertTrue(line.contains("physical 12884901888B"), line)
    }

    // MARK: - Headroom scales down for small budgets

    func testHeadroomShrinksWhenBudgetIsSmall() {
        // On a tight budget the 512MB cap must not eat more than 1/8 of the
        // allowance, or small-budget devices could never load anything.
        // physical 1600MB → floor 880MB; available 800MB is below the floor,
        // so ceiling = floor and headroom = ceiling/8 (not the 512MB cap).
        let physical = 1600 * MB
        let available = 800 * MB
        let floor = physical * 55 / 100
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, floor - floor / 8)
    }
}
