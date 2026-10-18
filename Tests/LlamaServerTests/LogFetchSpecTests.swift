import XCTest

/// `LogFetchSpec` is the `GET /logs` contract: which files, how much of each,
/// and how the body is shaped. It exists so logs can be pulled off the running
/// device over the LAN instead of via the Files app.
final class LogFetchSpecTests: XCTestCase {

    private func query(_ pairs: [(String, String)]) -> [URLQueryItem] {
        pairs.map { URLQueryItem(name: $0.0, value: $0.1) }
    }

    // MARK: - plan

    /// No query means everything: all three files, whole, with backups, as text.
    func testDefaultPlanReturnsEverything() {
        let plan = LogFetchSpec.plan(queryItems: [])
        XCTAssertEqual(plan.categories, LogCategory.allCases)
        XCTAssertNil(plan.tail)
        XCTAssertTrue(plan.includeBackups)
        XCTAssertEqual(plan.format, .text)
    }

    func testNameSelectsOneFile() {
        let plan = LogFetchSpec.plan(queryItems: query([("name", "inference")]))
        XCTAssertEqual(plan.categories, [.inference])
    }

    func testNameSelectsSeveralInFileOrder() {
        let plan = LogFetchSpec.plan(queryItems: query([("name", "payloads,server")]))
        XCTAssertEqual(plan.categories, [.server, .payloads])
    }

    func testNameAllAndUnknownNamesBothMeanEverything() {
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("name", "all")])).categories,
                       LogCategory.allCases)
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("name", "nonsense")])).categories,
                       LogCategory.allCases)
    }

    func testTailIsParsedAndClamped() {
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("tail", "200")])).tail, 200)
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("tail", "0")])).tail, 0)
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("tail", "-5")])).tail, 0)
        XCTAssertNil(LogFetchSpec.plan(queryItems: query([("tail", "soon")])).tail)
    }

    func testBackupsToggleOff() {
        for value in ["0", "false", "no", "FALSE"] {
            XCTAssertFalse(LogFetchSpec.plan(queryItems: query([("backups", value)])).includeBackups,
                           "\(value) should disable backups")
        }
        XCTAssertTrue(LogFetchSpec.plan(queryItems: query([("backups", "1")])).includeBackups)
    }

    func testFormatJsonAndUnknownFormatKeepsText() {
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("format", "JSON")])).format, .json)
        XCTAssertEqual(LogFetchSpec.plan(queryItems: query([("format", "csv")])).format, .text)
    }

    // MARK: - chunk (what FileLogger reads off disk)

    private func oneFile(tail: Int?) -> LogFetchSpec.Plan {
        LogFetchSpec.Plan(categories: [.server], tail: tail, includeBackups: true, format: .text)
    }

    /// The backup holds the older lines, so it comes first and a tail slice is
    /// the newest history.
    func testChunkJoinsBackupThenLive() {
        let chunk = LogFetchSpec.chunk(name: "llamaserver.log", backupText: "old\n",
                                       liveText: "new\n", plan: oneFile(tail: nil))
        XCTAssertEqual(chunk.content, "old\nnew\n")
        XCTAssertEqual(chunk.lines, 2)
        XCTAssertEqual(chunk.bytes, 8)
        XCTAssertFalse(chunk.truncated)
    }

    func testChunkWithoutBackupIsJustTheLiveFile() {
        let chunk = LogFetchSpec.chunk(name: "x.log", liveText: "a\nb\n", plan: oneFile(tail: nil))
        XCTAssertEqual(chunk.content, "a\nb\n")
        XCTAssertEqual(chunk.lines, 2)
    }

    func testChunkCountsLinesWithoutTheTrailingNewline() {
        XCTAssertEqual(LogFetchSpec.chunk(name: "x.log", liveText: "a\nb\n", plan: oneFile(tail: nil)).lines, 2)
        XCTAssertEqual(LogFetchSpec.chunk(name: "x.log", liveText: "a\nb", plan: oneFile(tail: nil)).lines, 2)
    }

    func testChunkTailKeepsTheNewestLinesAndFlagsIt() {
        let chunk = LogFetchSpec.chunk(name: "x.log", backupText: "a\nb\n",
                                       liveText: "c\nd\n", plan: oneFile(tail: 2))
        XCTAssertEqual(chunk.content, "c\nd\n")
        XCTAssertEqual(chunk.lines, 2)
        XCTAssertTrue(chunk.truncated)
    }

    func testChunkTailLargerThanTheFileIsNotTruncation() {
        let chunk = LogFetchSpec.chunk(name: "x.log", liveText: "a\nb\n", plan: oneFile(tail: 10))
        XCTAssertEqual(chunk.content, "a\nb\n")
        XCTAssertEqual(chunk.lines, 2)
        XCTAssertFalse(chunk.truncated)
    }

    func testChunkTailZeroReturnsNothing() {
        let chunk = LogFetchSpec.chunk(name: "x.log", liveText: "a\nb\n", plan: oneFile(tail: 0))
        XCTAssertEqual(chunk.content, "")
        XCTAssertEqual(chunk.lines, 0)
        XCTAssertEqual(chunk.bytes, 0)
        XCTAssertTrue(chunk.truncated)
    }

    func testChunkOfMissingFilesIsEmpty() {
        let chunk = LogFetchSpec.chunk(name: "x.log", plan: oneFile(tail: nil))
        XCTAssertEqual(chunk, LogFetchSpec.Chunk(name: "x.log", lines: 0, bytes: 0,
                                                 truncated: false, content: ""))
    }

    // MARK: - body

    func testTextBodyIsHeaderThenContent() {
        let chunk = LogFetchSpec.Chunk(name: "llamaserver.log", lines: 2, bytes: 12,
                                       truncated: false, content: "a\nb\n")
        XCTAssertEqual(LogFetchSpec.textBody([chunk]),
                       "===== llamaserver.log — 2 lines, 12 bytes =====\na\nb\n")
    }

    /// The on-device file always ends in a newline, but a tail slice does too —
    /// this only guards a header glued to the last line.
    func testTextBodyAddsMissingTrailingNewline() {
        let chunk = LogFetchSpec.Chunk(name: "x.log", lines: 1, bytes: 1,
                                       truncated: false, content: "a")
        XCTAssertEqual(LogFetchSpec.textBody([chunk]), "===== x.log — 1 lines, 1 bytes =====\na\n")
    }

    func testTextBodyMarksTruncation() {
        let chunk = LogFetchSpec.Chunk(name: "x.log", lines: 2, bytes: 4,
                                       truncated: true, content: "a\nb\n")
        XCTAssertEqual(LogFetchSpec.textBody([chunk]),
                       "===== x.log — 2 lines, 4 bytes, last 2 lines =====\na\nb\n")
    }

    func testTextBodyIsSilentAboutFilesWithNoLines() {
        let chunk = LogFetchSpec.Chunk(name: "x.log", lines: 0, bytes: 0, truncated: false, content: "")
        XCTAssertEqual(LogFetchSpec.textBody([chunk]), "===== x.log — 0 lines, 0 bytes =====\n")
    }

    func testJSONBodyRoundTripsTheChunks() throws {
        let chunks = [
            LogFetchSpec.Chunk(name: "llamaserver.log", lines: 1, bytes: 4,
                               truncated: false, content: "a\n"),
            LogFetchSpec.Chunk(name: "llamaserver-inference.log", lines: 0, bytes: 0,
                               truncated: true, content: ""),
        ]
        let data = try LogFetchSpec.jsonBody(chunks)
        XCTAssertEqual(try JSONDecoder().decode([LogFetchSpec.Chunk].self, from: data), chunks)
    }
}
