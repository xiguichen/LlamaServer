import XCTest

/// Tests for the flush/bypass policy that keeps FileLogger off the hot path.
///
/// Why this exists: FileLogger used to open + write + fsync + NSLog for EVERY
/// line on the CALLING thread (`queue.sync`). At `.verbose` that ran once per
/// generated token, on the inference thread (fsync ~0.5–5 ms against a
/// ~10–20 ms token budget), and ~1,500 llama.cpp debug lines during model
/// load each paid the same tax. The rewrite splits levels:
///
///   - error/warn are rare and crash-critical  → keep synchronous + fsync
///     (the pre-abort breadcrumbs must be on disk before the next thing the
///     caller does may crash);
///   - info/debug/verbose are routine          → buffered, fsync'd at most
///     once per interval / size threshold (bounded loss on hard crash).
///
/// `shouldNSLog` is similarly narrowed: NSLog costs ~10–100 µs per line and
/// only warn+ lines earn it.
final class FileLoggerTests: XCTestCase {

    func testCriticalLevelsFlushImmediately() {
        XCTAssertEqual(FileLogger.flushPolicy(for: .error), .immediate,
                       "error must be on disk + fsynced before log() returns")
        XCTAssertEqual(FileLogger.flushPolicy(for: .warn), .immediate,
                       "warn breadcrumbs precede aborts — same guarantee")
    }

    func testRoutineLevelsUseBufferedWrites() {
        XCTAssertEqual(FileLogger.flushPolicy(for: .info), .buffered)
        XCTAssertEqual(FileLogger.flushPolicy(for: .debug), .buffered)
        XCTAssertEqual(FileLogger.flushPolicy(for: .verbose), .buffered,
                       "per-token verbose lines must never fsync on the inference thread")
    }

    func testNSLogLimitedToWarningsAndAbove() {
        XCTAssertTrue(FileLogger.shouldNSLog(.error))
        XCTAssertTrue(FileLogger.shouldNSLog(.warn))
        XCTAssertFalse(FileLogger.shouldNSLog(.info),
                       "routine lines flooding the system log slows model load")
        XCTAssertFalse(FileLogger.shouldNSLog(.debug))
        XCTAssertFalse(FileLogger.shouldNSLog(.verbose))
    }
}
