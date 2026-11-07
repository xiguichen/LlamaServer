import XCTest

/// The budget follows the per-process ceiling the device *reports*, and nothing
/// else: no assumed fraction of RAM may override a successful measurement.
///
/// Measured facts that pin this policy (all on the 12.26GB test device):
/// - A 12GB device's per-process ceiling sits well below total RAM, so budgeting
///   against full physical RAM plans memory the process can never hold and
///   jetsam SIGKILLs the app with no log.
/// - `os_proc_available_memory()` reports **headroom, not the ceiling**: it
///   subtracts the caller's current footprint, so the ceiling is
///   `available + footprint`.
/// - The device's ceiling is **exactly 6 GiB (6,442,450,944B)** — every clean
///   reading of the launch sums to it, whatever the footprint. It is Apple's
///   per-device tier, not a fraction of RAM, and no app-side choice exceeds it.
/// - While the tier is unchanged, the ceiling is **constant for the launch, not
///   something that rises with the process**. When readings disagree, one
///   operand was stale. The policy resolves that conservatively — the smallest
///   successful reading wins — because the failure modes are asymmetric:
///   planning above the real limit is a SIGKILL, planning below it only costs
///   context tokens.
/// - iOS 27 adds one legitimate mid-launch rise: Game Mode raises the 12GB
///   A19 Pro tier from 6 GiB to 8 GiB. `settledCeiling` accepts it only when the
///   new limit is corroborated by a plateau; a single high reading still loses.
///
/// The ~9GB class the notes used to claim was never real: it came from adding
/// `available` to an **rss** reading, and rss is not the ceiling operand. A
/// policy that lets such a reading win budgets KV the kernel will not grant.
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

    /// The same run later, idle with both contexts live: `available` 4.36GB
    /// with a 1.78GB footprint. The sum is the *same* 6,442,450,944B — the
    /// ceiling did not rise when the model loaded.
    private let loadedAvailable = 4_576_193_192
    private let loadedFootprint = 1_866_257_752

    /// The disproved reading the old policy planned against: `available` paired
    /// with a 4.58GB **rss** number instead of `phys_footprint`, summing to
    /// ≈9.17GB. rss is not the ceiling operand; no reading on the device ever
    /// produced this sum. It exists here only to prove it cannot inflate the
    /// budget.
    private let inflatedAvailable = 4_359_432_632
    private let inflatedFootprint = 4_806_000_000

    private var initReading: MemoryBudget.Reading {
        MemoryBudget.Reading("at init", available: initAvailable, footprint: initFootprint)
    }
    private var loadedReading: MemoryBudget.Reading {
        MemoryBudget.Reading("after model load", available: loadedAvailable, footprint: loadedFootprint)
    }
    private var inflatedReading: MemoryBudget.Reading {
        MemoryBudget.Reading("rss mixed in", available: inflatedAvailable, footprint: inflatedFootprint)
    }

    // MARK: - The ceiling, reconstructed from headroom + footprint

    func testLimitIsAvailablePlusFootprint() {
        // os_proc_available_memory() = ceiling − footprint, so the ceiling is
        // their sum. On this device every real reading sums to 6 GiB.
        XCTAssertEqual(
            MemoryBudget.processLimit(physicalRAM: devicePhysical,
                                      available: initAvailable,
                                      footprint: initFootprint),
            6_442_450_944)
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

    func testConsistentReadingsAllMeasureTheSameCeiling() {
        // The case the device actually presents: every reading of the launch
        // sums to the same 6 GiB tier, whatever the footprint at the time.
        let readings = [initReading, loadedReading]
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical, readings: readings)
        XCTAssertEqual(ceiling.bytes, 6_442_450_944)
        XCTAssertEqual(ceiling.bytes, initAvailable + initFootprint)
        XCTAssertEqual(ceiling.bytes, loadedAvailable + loadedFootprint)
        XCTAssertEqual(ceiling.readingIndex, 0)
        XCTAssertEqual(ceiling.source(readings), "at init")
        XCTAssertFalse(ceiling.usedFallback)
    }

    func testDisagreeingReadingsResolveToTheSmallest() {
        // A reading that sums above another is arithmetically inconsistent —
        // one operand was stale. The smallest wins regardless of order: the
        // failure modes are asymmetric, since planning above the real limit is
        // a jetsam SIGKILL while planning below it only costs context tokens.
        let forward = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, inflatedReading])
        XCTAssertEqual(forward.bytes, initAvailable + initFootprint)
        let reversed = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                            readings: [inflatedReading, initReading])
        XCTAssertEqual(reversed.bytes, initAvailable + initFootprint)
    }

    func testInflatedRssReadingCannotRaiseTheBudget() {
        // The regression: pairing `available` with a 4.58GB rss number produced
        // a ~9.17GB "ceiling", which approved a 262,144-token q8_0 KV the
        // device cannot hold. The budget must stay under the measured 6 GiB.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, inflatedReading])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        XCTAssertEqual(budget, 6_442_450_944 - 512 * MB)
        XCTAssertLessThan(budget, 6 * GB, "the budget can never reach the whole 6 GiB tier")
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

    // MARK: - iOS 27 Game Mode can raise the tier mid-launch

    /// The Game Mode tier the 12GB A19 Pro devices expose under iOS 27:
    /// 8,192 MiB = 8 GiB. It is only available while Game Mode is active.
    private let gameModeLimit = 8 * 1024 * 1024 * 1024

    private func gameModeReading(_ label: String) -> MemoryBudget.Reading {
        MemoryBudget.Reading(label,
                             available: gameModeLimit - 128 * MB,
                             footprint: 128 * MB)
    }

    func testGameModeUpliftWinsOnlyAfterAStablePlateau() {
        // iOS 27 turns Game Mode on a few seconds after the app is foregrounded
        // and then raises the per-process ceiling from 6 GiB to 8 GiB. The
        // planning reading must follow the settled plateau, not the earlier
        // 6 GiB sample (which is a real reading of the pre-uplift state).
        let readings = [initReading,
                        gameModeReading("after model load"),
                        gameModeReading("after model load +0.25s"),
                        gameModeReading("after model load +0.50s")]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, gameModeLimit)
        XCTAssertEqual(ceiling.readingIndex, 3)
        XCTAssertEqual(ceiling.source(readings), "after model load +0.50s")
        XCTAssertFalse(ceiling.usedFallback)
    }

    func testGameModeUpliftWinsAfterAStableBaseRun() {
        // The real sampler shape: the 6 GiB tier is stable first, then Game Mode
        // raises it and the 8 GiB tier becomes the last stable plateau. The
        // earlier run must not pin the budget to 6 GiB.
        let readings = [initReading, loadedReading, loadedReading,
                        gameModeReading("after model load +0.75s"),
                        gameModeReading("after model load +1.00s"),
                        gameModeReading("after model load +1.25s")]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, gameModeLimit)
        XCTAssertEqual(ceiling.readingIndex, 5)
        XCTAssertEqual(ceiling.source(readings), "after model load +1.25s")
    }

    func testGameModeUpliftIsNotAcceptedFromASingleSample() {
        // A single high reading is exactly the 9.17GB-rss regression shape. It
        // must never win: only a repeated plateau is evidence of a state change.
        let readings = [initReading, gameModeReading("after model load")]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, initAvailable + initFootprint)
        XCTAssertEqual(ceiling.readingIndex, 0)
    }

    func testSingleInflatedReadingCannotDisplaceAStableBasePlateau() {
        // A stable 6 GiB run followed by one 9.17GB outlier: the outlier is not
        // a plateau, so the stable base remains the answer. This is the exact
        // failure mode the smallest-wins rule was introduced to prevent.
        let readings = [initReading, loadedReading, loadedReading, inflatedReading]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, initAvailable + initFootprint)
        XCTAssertEqual(ceiling.readingIndex, 2)
    }

    func testStableBaseTierStillWinsWhenGameModeDoesNotRaiseTheCeiling() {
        // No Game Mode (8GB device, older iOS, or the user disabled it): the
        // ceiling stays 6 GiB and the settled policy must return it unchanged.
        let readings = [initReading, loadedReading, loadedReading]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, 6_442_450_944)
        XCTAssertEqual(ceiling.readingIndex, 2)
    }

    func testSettledCeilingFallsBackToTheSmallestWhenNothingSettles() {
        // Two different valid ceilings and no plateau: the conservative rule
        // still applies. This is the guard that keeps a one-off inflated
        // reading from raising the budget.
        let readings = [initReading, inflatedReading]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, initAvailable + initFootprint)
    }

    func testSettledCeilingFallsBackWhenEveryReadingFailed() {
        let failed = MemoryBudget.Reading("after model load", available: 0, footprint: 0)
        let readings = [failed, failed, failed]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        XCTAssertEqual(ceiling.bytes, devicePhysical / 2)
        XCTAssertNil(ceiling.readingIndex)
        XCTAssertTrue(ceiling.usedFallback)
    }

    func testSettledCeilingLogLineShowsTheUpliftAndTheSettledSource() {
        let readings = [initReading,
                        gameModeReading("after model load"),
                        gameModeReading("after model load +0.25s"),
                        gameModeReading("after model load +0.50s")]
        let ceiling = MemoryBudget.settledCeiling(physicalRAM: devicePhysical,
                                                  readings: readings,
                                                  stableSamples: 3)
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        let line = MemoryBudget.logLine(budget: budget, ceiling: ceiling,
                                        readings: readings, physicalRAM: devicePhysical)
        XCTAssertTrue(line.contains("ceiling \(gameModeLimit)B"), line)
        XCTAssertTrue(line.contains("from after model load +0.50s"), line)
        XCTAssertTrue(line.contains("at init: available \(initAvailable)B"), line)
        XCTAssertTrue(line.contains("after model load +0.50s: available \(gameModeLimit - 128 * MB)B"), line)
    }

    // MARK: - The regression: the measured ceiling drives the budget

    func testMeasuredCeilingDrivesTheBudget() {
        // The device grants exactly 6 GiB; the budget is that minus the
        // transient headroom, and never a byte above the tier itself.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, loadedReading])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        XCTAssertEqual(budget, ceiling.bytes - 512 * MB)
        XCTAssertEqual(budget, 5_905_580_032)
        XCTAssertLessThan(budget, 6 * GB)
    }

    func testMeasuredBudgetFitsTheMeasuredQ8Context() {
        // Consequence that matters at 131,072 context and 17,408 B/token: the
        // q8_0 KV is 2.28GB. Under the real 5.91GB budget (2.78GB model plus
        // the 320MB compute reserve) it fits with ~0.5GB of slack — which is
        // exactly why the device measures 160,512 tokens for q8_0, while the
        // disproved 9.17GB budget would have approved the full 262,144.
        let ceiling = MemoryBudget.ceiling(physicalRAM: devicePhysical,
                                           readings: [initReading, loadedReading])
        let budget = MemoryBudget.budgetBytes(physicalRAM: devicePhysical, ceiling: ceiling.bytes)
        let kvBudget = budget - 2_780_000_000 - 320 * MB
        XCTAssertGreaterThan(kvBudget, 131_072 * 17_408 + 400 * MB,
                             "131,072-token q8_0 KV must fit the measured budget")
        XCTAssertLessThan(kvBudget, 262_144 * 17_408,
                          "the full window does not fit q8_0 on this device")
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
        XCTAssertTrue(line.contains("from at init"), line)
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
