import XCTest

/// The budget follows the per-process ceiling the device *reports*, and nothing
/// else: no assumed fraction of RAM may override a successful measurement.
///
/// Four measured facts pin this policy (all on the 12.26GB test device):
/// - A 12GB device's per-process ceiling sits well below total RAM, so budgeting
///   against full physical RAM plans memory the process can never hold and
///   jetsam SIGKILLs the app with no log.
/// - `os_proc_available_memory()` reports **headroom, not the ceiling**: it
///   subtracts the caller's current footprint, so the ceiling is
///   `available + footprint`. On its own the raw reading under-reports the
///   device's real allowance (6.35GB available with a 92MB footprint at init,
///   while the same run held 4.58GB with 4.16GB still available).
/// - The ceiling really is raised by `com.apple.developer.kernel.increased-memory-limit`:
///   entitled installs reach the ~9GB class, un-entitled ones land near half of
///   physical.
/// - **The ceiling rises with the process**, so one reading is not enough: the
///   same launch reported 6.44GB at the end of backend init and ~8.74GiB once
///   the model was loaded. The largest successful reading is the one to plan
///   against.
///
/// The 55%-of-physical floor was removed on 2026-10-18: it silently overrode a
/// *successful* measurement, and it is exactly why the device logged a number
/// its own line could not explain — `budget 6208012288B` is 55% of physical
/// minus headroom, while the line's own measured limit was 6442450944B, *below*
/// the floor, so the measurement never got a vote. Half of physical survives
/// only as the answer when every reading failed, reported as a fallback so it
/// can never be mistaken for a measurement.
final class MemoryBudgetTests: XCTestCase {
    private let GB = 1024 * 1024 * 1024
    private let MB = 1024 * 1024

    /// Raw values from the device's own budget log line (2026-10-18 16:53:31):
    /// `memory budget 6208012288B (6.21 GB) [limit 6442450944B (6.44 GB)
    /// = process-available 6350387288B (6.35 GB) + footprint 92063656B (92.1 MB),
    /// physical 12263424000B (12.26 GB)]`.
    private let devicePhysical = 12_263_424_000
    private let initAvailable = 6_350_387_288
    private let initFootprint = 92_063_656

    /// The same run later: the heartbeat logged `available=4359432632B(4157MB)`
    /// with the model resident, so `available + footprint` was ≈9.17GB. The
    /// footprint here is chosen to land on that class (the pre-change heartbeat
    /// only carried rss; the line now prints the footprint itself).
    private let loadedAvailable = 4_359_432_632
    private let loadedFootprint = 4_806_000_000

    private var initReading: MemoryBudget.Reading {
        MemoryBudget.Reading("at init", available: initAvailable, footprint: initFootprint)
    }
    private var loadedReading: MemoryBudget.Reading {
        MemoryBudget.Reading("after model load", available: loadedAvailable, footprint: loadedFootprint)
    }

    // MARK: - The ceiling, reconstructed from headroom + footprint

    func testLimitIsAvailablePlusFootprint() {
        // os_proc_available_memory() = ceiling − footprint, so the ceiling is
        // their sum. This is the number the entitlement actually grants.
        XCTAssertEqual(
            MemoryBudget.processLimit(physicalRAM: devicePhysical,
                                      available: 6_368_557_120,
                                      footprint: 2_890_000_000),
            9_258_557_120)
    }

    func testLimitIsZeroWhenAvailableIsUnknown() {
        // 0 = API unknown (not an app, or the call failed). The limit must not
        // invent an entitlement from nothing; `ceiling` decides what to do.
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

    // MARK: - Which reading wins

    func testLargestSuccessfulReadingWins() {
        // The case the device actually presents: a low reading at init, a higher
        // one once the model is in. The higher ceiling is the real allowance.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, loadedReading])
        XCTAssertEqual(ceiling.bytes, loadedAvailable + loadedFootprint)
        XCTAssertEqual(ceiling.readingIndex, 1)
        XCTAssertEqual(ceiling.source([initReading, loadedReading]), "after model load")
        XCTAssertFalse(ceiling.usedFallback)
    }

    func testMoreReadingsDoNotRaiseTheCeilingAboveTheBestMeasurement() {
        // Guard against "average" or "latest wins" creeping in: a *later* low
        // reading must not shrink the ceiling either.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [loadedReading, initReading])
        XCTAssertEqual(ceiling.bytes, loadedAvailable + loadedFootprint)
        XCTAssertEqual(ceiling.readingIndex, 0)
    }

    func testFailedReadingNeverWins() {
        // available == 0 is a failed read, not a zero ceiling: it must not drag
        // a good reading down, and it must not win on its own.
        let failed = MemoryBudget.Reading("after model load", available: 0, footprint: loadedFootprint)
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, failed])
        XCTAssertEqual(ceiling.bytes, initAvailable + initFootprint)
        XCTAssertEqual(ceiling.readingIndex, 0)
        XCTAssertFalse(ceiling.usedFallback)
    }

    func testEveryReadingFailedFallsBackToHalfOfPhysical() {
        let failed = MemoryBudget.Reading("at init", available: 0, footprint: 0)
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical, readings: [failed])
        XCTAssertEqual(ceiling.bytes, devicePhysical / 2)
        XCTAssertNil(ceiling.readingIndex)
        XCTAssertTrue(ceiling.usedFallback, "an assumption must be logged as one")
        XCTAssertEqual(ceiling.source([failed]), "fallback (every reading failed)")
    }

    func testNoReadingsFallsBackToHalfOfPhysical() {
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical, readings: [])
        XCTAssertEqual(ceiling.bytes, devicePhysical / 2)
        XCTAssertTrue(ceiling.usedFallback)
    }

    func testCeilingIsCappedAtPhysical() {
        let reading = MemoryBudget.Reading("garbage", available: 3 * GB, footprint: 9 * GB)
        let ceiling = MemoryBudget.ceiling(physicalRAM: 4 * GB, readings: [reading])
        XCTAssertEqual(ceiling.bytes, 4 * GB)
    }

    // MARK: - The regression: the measured ceiling drives the budget

    func testMeasuredCeilingDrivesTheBudget() {
        // The device grants ~9.17GB once loaded; the budget must not throw a
        // third of it away (the old floor computed 6,208,012,288B, below the
        // *init* ceiling of 6,442,450,944B even before the second reading).
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, loadedReading])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        XCTAssertEqual(budget, ceiling.bytes - 512 * MB)
        XCTAssertGreaterThan(budget, 8_000_000_000)
    }

    func testMeasuredBudgetFitsTheConfiguredKVWithHeadroomToSpare() {
        // The consequence that matters: at 131,072 context and 17,408 B/token the
        // q8_0 KV is 2.28GB. Under the old 6.21GB budget only ~0.78GB of slack
        // was left, so a slightly bigger model or KV type got clamped.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, loadedReading])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        let kvBudget = budget - 2_780_000_000 - 320 * MB
        XCTAssertGreaterThan(kvBudget, 131_072 * 17_408 + 1024 * MB,
                             "131,072-token q8_0 KV must leave real slack, not ~0.8GB")
    }

    // MARK: - A low but valid reading is respected, not overridden

    func testLowReadingIsRespectedNotFloored() {
        // The whole point of removing the floor: a measurement the device
        // actually returned is now the answer, even when it is below the old
        // 55%-of-physical floor. The pre-check then refuses with its numbers
        // logged rather than planning memory jetsam will not allow.
        let physical = devicePhysical
        let reading = MemoryBudget.Reading("at init", available: 3_000_000_000, footprint: 460_000_000)
        let ceiling = MemoryBudget.ceiling(physicalRAM: physical, readings: [reading])
        XCTAssertEqual(ceiling.bytes, 3_460_000_000)

        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, ceiling: ceiling.bytes)
        // headroom = ceiling/8 (432.5MB) because that is below the 512MB cap.
        XCTAssertEqual(budget, 3_460_000_000 - 432_500_000)

        let oldFloorBudget = physical * 55 / 100 - 512 * MB
        XCTAssertEqual(oldFloorBudget, 6_208_012_288, "the number the device logged under the old policy")
        XCTAssertLessThan(budget, oldFloorBudget,
                          "no invented floor may override a successful measurement")

        // Consequence, stated: this budget cannot hold the 2.78GB model plus its
        // 320MB reserve, so the load is refused with the numbers instead of
        // being sized against 6.74GB and jetsam-killed later.
        XCTAssertGreaterThan(2_780_000_000 + 320 * MB, budget)
    }

    // MARK: - Healthy ceilings still win, caps still hold

    func testHealthyCeilingWins() {
        let physical = 12 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, ceiling: 9_000_000_000)
        XCTAssertEqual(budget, 9_000_000_000 - 512 * MB,
                       "a healthy ceiling must not be capped down")
    }

    func testCeilingAbovePhysicalIsCappedAtPhysical() {
        // The ceiling should never exceed physical RAM, but a guard keeps a bad
        // value from inflating the budget past hardware.
        let physical = 4 * GB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, ceiling: 8 * GB)
        XCTAssertEqual(budget, physical - 512 * MB)
    }

    // MARK: - Headroom scales down for small ceilings

    func testHeadroomShrinksWhenCeilingIsSmall() {
        // On a tight budget the 512MB cap must not eat more than 1/8 of the
        // allowance, or small-budget devices could never load anything.
        let physical = 1600 * MB
        let ceiling = 800 * MB
        let budget = MemoryBudget.budgetBytes(physicalRAM: physical, ceiling: ceiling)
        XCTAssertEqual(budget, ceiling - ceiling / 8)
    }

    // MARK: - Start-log line

    func testLogLineCarriesRawByteValuesAndNamesTheWinner() {
        // Raw bytes must appear verbatim — the rounded rendering alone can't be
        // compared byte-for-byte across toolchain/entitlement changes. Every
        // reading's inputs and derived ceiling are kept, plus which one won: a
        // future disagreement about any input stays diagnosable from the log.
        let readings = [initReading, loadedReading]
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical, readings: readings)
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        let line = MemoryBudget.logLine(budget: budget, ceiling: ceiling,
                                        readings: readings, physicalRAM: devicePhysical)
        XCTAssertTrue(line.contains("memory budget \(budget)B"), line)
        XCTAssertTrue(line.contains("ceiling \(loadedAvailable + loadedFootprint)B"), line)
        XCTAssertTrue(line.contains("from after model load"), line)
        XCTAssertTrue(line.contains("at init: available \(initAvailable)B"), line)
        XCTAssertTrue(line.contains("footprint \(initFootprint)B"), line)
        XCTAssertTrue(line.contains("-> \(initAvailable + initFootprint)B"), line)
        XCTAssertTrue(line.contains("after model load: available \(loadedAvailable)B"), line)
        XCTAssertTrue(line.contains("footprint \(loadedFootprint)B"), line)
        XCTAssertTrue(line.contains("physical \(devicePhysical)B"), line)
    }

    func testLogLineReportsTheFallback() {
        // An assumed ceiling must say so in the line, or a fallback can be read
        // as a measurement.
        let failed = MemoryBudget.Reading("at init", available: 0, footprint: 0)
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical, readings: [failed])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        let line = MemoryBudget.logLine(budget: budget, ceiling: ceiling,
                                        readings: [failed], physicalRAM: devicePhysical)
        XCTAssertTrue(line.contains("fallback (every reading failed)"), line)
    }
}
