import XCTest

/// Tests for the log split (`LogRouter`) and the in-app panel's rendering
/// budget (`LogPanelSpec`).
///
/// Why this exists: the in-app log panel was the app's heaviest CPU consumer.
/// A `cpu_resource` report (bug_type 202: 90 s CPU over 157 s = 57 %, limit
/// 50 %) put the main thread's hottest stack in `SwiftUI -> CoreText ->
/// libicucore` — text layout. The panel drew up to 500 lines in a non-lazy
/// stack with `.textSelection(.enabled)` on each, and at `.verbose` a single
/// line could be a 72 k-character prompt, re-laid-out on every 10 Hz mirror
/// batch.
///
/// The fix has two halves, both pinned here:
///   - the panel renders only basic (error/warn/info) lines, capped by count
///     and line length, so no payload can blow up layout;
///   - the bulky payloads go to their own file, and routing must send each line
///     to exactly one file.
final class LogRoutingTests: XCTestCase {

    // MARK: - LogRouter

    func testVerboseAlwaysGoesToThePayloadFile() {
        XCTAssertEqual(LogRouter.category(level: .verbose, sourceFile: "/a/Sources/App/LlamaHTTPServer.swift"), .payloads)
        XCTAssertEqual(LogRouter.category(level: .verbose, sourceFile: "/a/Sources/App/LlamaInference.swift"), .payloads,
                       "verbose is the payload channel regardless of origin")
        XCTAssertEqual(LogRouter.category(level: .verbose, sourceFile: "Unknown.swift"), .payloads)
    }

    func testEngineLinesGoToTheInferenceFile() {
        for level in [LogLevel.error, .warn, .info, .debug] {
            XCTAssertEqual(LogRouter.category(level: level, sourceFile: "/a/Sources/App/LlamaInference.swift"), .inference,
                           "\(level) from the engine belongs in the inference log")
        }
    }

    func testEngineRoutingMatchesBareFileName() {
        // `#file` is the full path in Swift 5 mode but can be module-relative in
        // newer modes; the suffix match covers both.
        XCTAssertEqual(LogRouter.category(level: .info, sourceFile: "LlamaInference.swift"), .inference)
    }

    func testEverythingElseGoesToTheServerFile() {
        for level in [LogLevel.error, .warn, .info, .debug] {
            XCTAssertEqual(LogRouter.category(level: level, sourceFile: "/a/Sources/App/LlamaHTTPServer.swift"), .server)
            XCTAssertEqual(LogRouter.category(level: level, sourceFile: "/a/Sources/App/ServerViewModel.swift"), .server)
            XCTAssertEqual(LogRouter.category(level: level, sourceFile: "/a/Sources/App/BackgroundKeepAlive.swift"), .server)
        }
    }

    func testCategoryFileNamesAreDistinctAndDocumented() {
        let names = LogCategory.allCases.map(\.fileName)
        XCTAssertEqual(Set(names).count, LogCategory.allCases.count, "one file per category")
        XCTAssertEqual(LogCategory.server.fileName, "llamaserver.log",
                       "the documented Files-app path must keep resolving")
        XCTAssertEqual(LogCategory.inference.fileName, "llamaserver-inference.log")
        XCTAssertEqual(LogCategory.payloads.fileName, "llamaserver-payloads.log")
    }

    // MARK: - LogPanelSpec: which lines the panel renders

    func testBasicLevelsAreErrorWarnInfo() {
        XCTAssertTrue(LogPanelSpec.isBasic(.error))
        XCTAssertTrue(LogPanelSpec.isBasic(.warn))
        XCTAssertTrue(LogPanelSpec.isBasic(.info))
        XCTAssertFalse(LogPanelSpec.isBasic(.debug))
        XCTAssertFalse(LogPanelSpec.isBasic(.verbose))
    }

    func testParsesTheTagFromAFormattedLine() {
        XCTAssertEqual(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] [ERROR  ] boom"), .error)
        XCTAssertEqual(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] [WARN   ] hmm"), .warn)
        XCTAssertEqual(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] [INFO   ] ok"), .info)
        XCTAssertEqual(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] [DEBUG  ] trace"), .debug)
        XCTAssertEqual(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] [VERBOSE] dump"), .verbose)
    }

    func testIgnoresMalformedLines() {
        XCTAssertNil(LogPanelSpec.level(ofLine: "no brackets at all"))
        XCTAssertNil(LogPanelSpec.level(ofLine: "["))
        XCTAssertNil(LogPanelSpec.level(ofLine: "[2026-10-17 10:08:46.000] not a tag"))
        XCTAssertFalse(LogPanelSpec.isBasic(line: "no brackets at all"))
    }

    func testTagInsideTheMessageDoesNotCount() {
        // A message body could quote a tag; only the field after the timestamp
        // decides, so this debug line stays out of the panel.
        let line = "[2026-10-17 10:08:46.000] [DEBUG  ] echoed [INFO   ] from elsewhere"
        XCTAssertEqual(LogPanelSpec.level(ofLine: line), .debug)
        XCTAssertFalse(LogPanelSpec.isBasic(line: line))
    }

    // MARK: - LogPanelSpec: the rendering budget

    func testShortLinesAreUntouched() {
        let line = "[2026-10-17 10:08:46.000] [INFO   ] generated 42 tokens"
        XCTAssertEqual(LogPanelSpec.truncate(line), line)
    }

    func testOverlongLinesAreCappedAndSayHowMuchWasHidden() {
        let line = String(repeating: "x", count: LogPanelSpec.maxLineLength + 70000)
        let capped = LogPanelSpec.truncate(line)
        XCTAssertTrue(capped.hasPrefix(String(repeating: "x", count: LogPanelSpec.maxLineLength)))
        XCTAssertTrue(capped.contains("[+70000 chars"), "the count tells the user to open the file")
        XCTAssertLessThan(capped.count, LogPanelSpec.maxLineLength + 60)
    }

    func testAppendCapsTheBufferAndDropsOldestFirst() {
        var lines: [String] = []
        for i in 0..<(LogPanelSpec.maxLines + 5) {
            LogPanelSpec.append("line \(i)", to: &lines)
        }
        XCTAssertEqual(lines.count, LogPanelSpec.maxLines)
        XCTAssertEqual(lines.first, "line 5", "oldest lines are dropped")
        XCTAssertEqual(lines.last, "line \(LogPanelSpec.maxLines + 4)")
    }

    func testAppendTruncatesEachLine() {
        var lines: [String] = []
        LogPanelSpec.append(String(repeating: "y", count: LogPanelSpec.maxLineLength + 10), to: &lines)
        XCTAssertEqual(lines.count, 1)
        XCTAssertTrue(lines[0].contains("[+10 chars"))
    }
}
