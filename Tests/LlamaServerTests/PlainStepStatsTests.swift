import XCTest

final class PlainStepStatsTests: XCTestCase {

    func testZeroStepsSummary() {
        XCTAssertEqual(PlainStepStats().summary(), "no plain steps recorded")
    }

    func testHostIsWallMinusWaitAndSample() {
        var s = PlainStepStats()
        s.record(gpuWait: 0.043, sample: 0.0003, wall: 0.0437)
        XCTAssertEqual(s.stepCount, 1)
        XCTAssertEqual(s.summary(),
                       "gpu-wait 43.00ms/step + sample 0.30ms/step + host 0.40ms/step = wall 43.70ms/step (1 steps)")
    }

    func testStepsAverage() {
        var s = PlainStepStats()
        s.record(gpuWait: 0.040, sample: 0.001, wall: 0.042)
        s.record(gpuWait: 0.050, sample: 0.001, wall: 0.052)
        XCTAssertEqual(s.stepCount, 2)
        XCTAssertEqual(s.summary(),
                       "gpu-wait 45.00ms/step + sample 1.00ms/step + host 1.00ms/step = wall 47.00ms/step (2 steps)")
    }

    func testNegativeDurationsClampAndHostNeverNegative() {
        var s = PlainStepStats()
        s.record(gpuWait: -1, sample: 0.002, wall: 0.001)
        XCTAssertEqual(s.summary(),
                       "gpu-wait 0.00ms/step + sample 2.00ms/step + host 0.00ms/step = wall 1.00ms/step (1 steps)")
    }
}
