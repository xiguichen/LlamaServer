import XCTest

/// The on-device half of the self-upgrade surface: `Documents/updates/` and the
/// rotation.
///
/// The property that matters is the one a user hits on a bad upload: the file an
/// upload replaces is the *only* copy of a working build. A successful store must
/// leave the replaced build in the `.prev` slot (so a build that turns out to be
/// broken on device can be rolled back), and a store that fails must not have
/// destroyed anything.
final class UpdateStoreTests: XCTestCase {
    private var temp: URL!
    private var store: UpdateStore!

    override func setUpWithError() throws {
        temp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("updates-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        store = UpdateStore(documentsPath: temp.path)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: temp)
    }

    /// A body with a real zip header, so these tests exercise the same shape the
    /// route accepts.
    private func ipa(_ byte: UInt8, size: Int = 64) -> Data {
        Data(UpdateSpec.zipMagic) + Data(repeating: byte, count: size)
    }

    private func file(_ path: String) -> Data? {
        try? Data(contentsOf: URL(fileURLWithPath: path))
    }

    func testStoreCreatesTheDirectoryAndWritesBothFiles() throws {
        let locations = UpdateSpec.locations(documentsPath: temp.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: locations.directory),
                       "precondition: nothing exists yet")
        let body = ipa(0x41)
        try store.store(body: body, build: "441aadc")

        XCTAssertEqual(file(locations.current), body, "the current slot holds the body verbatim")
        let record = UpdateSpec.decodeRecord(file(locations.metadata) ?? Data())
        XCTAssertEqual(record?.build, "441aadc")
        XCTAssertEqual(record?.bytes, body.count)
    }

    func testASecondStoreRotatesTheReplacedBuildIntoThePreviousSlot() throws {
        let locations = UpdateSpec.locations(documentsPath: temp.path)
        let first = ipa(0x41)
        let second = ipa(0x42, size: 128)
        try store.store(body: first, build: "aaaaaaaa")
        try store.store(body: second, build: "bbbbbbb")

        XCTAssertEqual(file(locations.current), second, "current is the newest build")
        XCTAssertEqual(file(locations.previous), first,
                       "the build just replaced is the rollback copy")

        // A third upload keeps one rollback slot: the build it replaced.
        let third = ipa(0x43)
        try store.store(body: third, build: "ccccccc")
        XCTAssertEqual(file(locations.current), third)
        XCTAssertEqual(file(locations.previous), second)
        XCTAssertEqual(UpdateSpec.decodeRecord(file(locations.metadata) ?? Data())?.build, "ccccccc")
    }

    func testStatusIsEmptyBeforeAnyUpload() {
        let status = store.status()
        XCTAssertFalse(status.current.present)
        XCTAssertFalse(status.previous.present)
        XCTAssertNil(status.current.bytes)
        XCTAssertNil(status.uploadedAt)
    }

    func testStatusDescribesWhatIsOnDisk() throws {
        let first = ipa(0x41)
        let second = ipa(0x42, size: 200)
        try store.store(body: first, build: "aaaaaaaa")
        try store.store(body: second, build: "bbbbbbb")

        let status = store.status()
        XCTAssertTrue(status.current.present)
        XCTAssertEqual(status.current.bytes, second.count)
        XCTAssertEqual(status.current.build, "bbbbbbb")
        XCTAssertNotNil(status.current.modifiedAt)
        XCTAssertTrue(status.previous.present)
        XCTAssertEqual(status.previous.bytes, first.count)
        XCTAssertNil(status.previous.build, "only the current slot is described by latest.json")
        XCTAssertNotNil(status.uploadedAt)
    }

    func testAFileDroppedInByHandIsReportedEvenWithoutMetadata() throws {
        // The Files app is right there: a build copied in by hand has no
        // latest.json, and must still be reported — just without a build id.
        let locations = UpdateSpec.locations(documentsPath: temp.path)
        try FileManager.default.createDirectory(atPath: locations.directory,
                                                withIntermediateDirectories: true)
        try ipa(0x41).write(to: URL(fileURLWithPath: locations.current))

        let status = store.status()
        XCTAssertTrue(status.current.present)
        XCTAssertEqual(status.current.bytes, UpdateSpec.zipMagic.count + 64)
        XCTAssertNil(status.current.build)
        XCTAssertNil(status.uploadedAt)
    }

    func testCorruptMetadataIsReportedWithoutABuildId() throws {
        let locations = UpdateSpec.locations(documentsPath: temp.path)
        try store.store(body: ipa(0x41), build: "441aadc")
        try Data("half-written".utf8).write(to: URL(fileURLWithPath: locations.metadata))

        let status = store.status()
        XCTAssertTrue(status.current.present)
        XCTAssertNil(status.current.build, "a corrupt latest.json must not fail the request")
        XCTAssertNil(status.uploadedAt)
    }

    func testStoreThrowsWhenTheDirectoryCannotBeCreated() throws {
        // A store that cannot write must fail loudly: the caller reports 500 and
        // the user knows the upload did not land.
        let blocker = temp.appendingPathComponent("blocker")
        try Data("x".utf8).write(to: blocker)
        let broken = UpdateStore(documentsPath: blocker.appendingPathComponent("nested").path)
        XCTAssertThrowsError(try broken.store(body: ipa(0x41), build: "441aadc"))
    }

    func testALandedBodyIsSuccessEvenIfTheMetadataCannotBeWritten() throws {
        // The IPA is what the user installs. Once the body has landed the upload
        // has succeeded, so a metadata write that fails (disk full between the two
        // writes) must not be reported as a failed upload — that would send the
        // user back to re-push a file that is already there. What is lost is the
        // record, and it shows up honestly as "no build id" in GET /v1/updates.
        let locations = UpdateSpec.locations(documentsPath: temp.path)
        try FileManager.default.createDirectory(atPath: locations.directory,
                                                withIntermediateDirectories: true)
        // A directory where latest.json should go makes the atomic write fail.
        try FileManager.default.createDirectory(atPath: locations.metadata,
                                                withIntermediateDirectories: true)
        let body = ipa(0x41)
        XCTAssertNoThrow(try store.store(body: body, build: "441aadc"))
        XCTAssertEqual(file(locations.current), body, "the IPA is installable")
        XCTAssertNil(store.status().current.build, "the record could not be written")
    }
}
