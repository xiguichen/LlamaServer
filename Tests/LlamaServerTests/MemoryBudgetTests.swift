import XCTest

/// Budget must follow the per-process jetsam limit, not physical RAM.
///
/// Measured on-device (iOS 27, 12GB hardware): rss 4148MB + available 1960MB
/// = a ~6.1GB per-process ceiling. Budgeting against the full 11.7GB physical
/// RAM plans memory the process can never hold, and the overshoot gets the app
/// SIGKILLed by jetsam with no log and no chance to fail gracefully.
final class MemoryBudgetTests: XCTestCase {
    private let GB = 1024 * 1024 * 1024
    private let MB = 1024 * 1024

    // MARK: - Real limit wins over physical RAM

    func testBudgetUsesProcessAvailableWhenLowerThanPhysicalRAM() {
        let physical = 12 * GB
        let available = 6_108 * MB   // measured rss + available ceiling
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, available - 512 * MB,
                       "budget must derive from the jetsam limit, leaving 512MB transient headroom")
    }

    // MARK: - Defensive cap

    func testBudgetNeverExceedsPhysicalRAM() {
        // os_proc_available_memory() should never exceed physical RAM, but a
        // guard keeps a bad value from inflating the budget past hardware.
        let physical = 4 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: 8 * GB)
        XCTAssertEqual(budget, physical - 512 * MB)
    }

    // MARK: - Headroom scales down for small budgets

    func testHeadroomShrinksWhenBudgetIsSmall() {
        // On a tight budget the 512MB cap must not eat more than 1/8 of the
        // allowance, or small-budget devices could never load anything.
        let physical = 12 * GB
        let available = 800 * MB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: available)
        XCTAssertEqual(budget, 700 * MB, "headroom = min(512MB, available/8) = 100MB")
    }

    // MARK: - API-unavailable fallback

    func testFallsBackToPhysicalRAMWhenAPIReportsZero() {
        // os_proc_available_memory() returns 0 when the caller is not an app
        // or already over its limit; fall back to legacy behavior in that case.
        let physical = 12 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, processAvailable: 0)
        XCTAssertEqual(budget, physical - 512 * MB)
    }
}
