import XCTest

/// The `GET /v1/updates` + `PUT /v1/updates/ipa` contract.
///
/// The guards exist because the thing being overwritten is the *only* copy of a
/// working build: a rejected upload must leave both slots exactly as they were,
/// which is why rejection is decided purely, before any filesystem call.
final class UpdateSpecTests: XCTestCase {
    private func zip(_ extra: Int = 16) -> Data {
        Data(UpdateSpec.zipMagic) + Data(repeating: 0, count: extra)
    }

    // MARK: - Storage layout

    func testLocationsLiveInDocumentsUpdates() {
        let l = UpdateSpec.locations(documentsPath: "/var/mobile/Documents")
        XCTAssertEqual(l.directory, "/var/mobile/Documents/updates")
        XCTAssertEqual(l.current, "/var/mobile/Documents/updates/LlamaServer-unsigned.ipa")
        XCTAssertEqual(l.previous, "/var/mobile/Documents/updates/LlamaServer-unsigned.prev.ipa")
        XCTAssertEqual(l.metadata, "/var/mobile/Documents/updates/latest.json")
    }

    func testCurrentFileNameCarriesNoBuildHash() {
        // Files sorts by name; the newest build must be the one at the top, and
        // the build identity must come from latest.json instead.
        XCTAssertEqual(UpdateSpec.currentFileName, "LlamaServer-unsigned.ipa")
        XCTAssertEqual(UpdateSpec.previousFileName, "LlamaServer-unsigned.prev.ipa")
    }

    // MARK: - Query

    func testBuildReadsTheQueryCaseInsensitivelyAndTrims() {
        XCTAssertEqual(
            UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "BUILD", value: " 441aadc ")]),
            "441aadc")
        XCTAssertEqual(
            UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "other", value: "x"),
                                              URLQueryItem(name: "build", value: "abc1234")]),
            "abc1234")
    }

    func testBuildIsNilWhenAbsentOrEmpty() {
        XCTAssertNil(UpdateSpec.build(fromQueryItems: []))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: "")]))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: "   ")]))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: nil)]))
    }

    // MARK: - Rejection

    func testMissingBuildIsRejectedFirst() {
        // Even a valid IPA is refused: the build id is the point of the record.
        XCTAssertEqual(UpdateSpec.rejection(build: nil, body: zip()), .missingBuild)
        XCTAssertEqual(UpdateSpec.rejection(build: "", body: zip()), .missingBuild)
        XCTAssertEqual(UpdateSpec.rejection(build: "   ", body: zip()), .missingBuild)
    }

    func testOversizedBodyIsRejected() {
        let body = Data(UpdateSpec.zipMagic) + Data(repeating: 0, count: UpdateSpec.maxUploadBytes)
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: body),
                       .tooLarge(bytes: body.count))
    }

    func testBodyExactlyAtTheCapIsAccepted() {
        // The boundary: > max is dead, == max is fine.
        let body = Data(UpdateSpec.zipMagic)
            + Data(repeating: 0, count: UpdateSpec.maxUploadBytes - UpdateSpec.zipMagic.count)
        XCTAssertEqual(body.count, UpdateSpec.maxUploadBytes)
        XCTAssertNil(UpdateSpec.rejection(build: "441aadc", body: body))
    }

    func testNonIPAIsRejected() {
        // A failed download saves an HTML error page; writing it would replace a
        // working build with something AltStore cannot install.
        let html = Data("<!DOCTYPE html><html>404</html>".utf8)
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: html),
                       .notAnIPA(bytes: html.count))
    }

    func testEmptyBodyIsRejectedAsNotAnIPA() {
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: Data()), .notAnIPA(bytes: 0))
    }

    func testRealIPAIsAccepted() {
        XCTAssertNil(UpdateSpec.rejection(build: "441aadc", body: zip(4096)))
    }

    func testRejectionMessagesCarryTheNumbers() {
        XCTAssertTrue(UpdateSpec.Rejection.tooLarge(bytes: 999).message.contains("999"))
        XCTAssertTrue(UpdateSpec.Rejection.tooLarge(bytes: 999).message.contains("\(UpdateSpec.maxUploadBytes)"))
        XCTAssertTrue(UpdateSpec.Rejection.notAnIPA(bytes: 12).message.lowercased().contains("ipa"))
        XCTAssertTrue(UpdateSpec.Rejection.missingBuild.message.contains("build"))
    }

    // MARK: - Records and status

    func testRecordRoundTripsWithISO8601Dates() throws {
        // An integral timestamp on purpose: ISO-8601 has no sub-second precision,
        // so a fractional Date would not survive the round trip.
        let record = UpdateSpec.Record(build: "441aadc", bytes: 4_083_117,
                                       uploadedAt: Date(timeIntervalSince1970: 1_761_000_000))
        let data = try UpdateSpec.encode(record)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"build\":\"441aadc\""), text)
        XCTAssertTrue(text.contains("T") && text.contains("Z"), "ISO-8601, got \(text)")
        XCTAssertFalse(text.contains("1761000000"), "the epoch number must not be the encoding: \(text)")
        XCTAssertEqual(UpdateSpec.decodeRecord(data), record)
    }

    func testCorruptMetadataDecodesToNilInsteadOfThrowing() {
        XCTAssertNil(UpdateSpec.decodeRecord(Data("not json".utf8)))
    }

    func testStatusReportsBothSlotsAndFillsBuildFromMetadata() throws {
        let record = UpdateSpec.Record(build: "441aadc", bytes: 100,
                                       uploadedAt: Date(timeIntervalSince1970: 1_761_000_000))
        let status = UpdateSpec.status(currentBytes: 4_083_117,
                                       currentModified: Date(timeIntervalSince1970: 1_761_000_000),
                                       previousBytes: 4_083_116,
                                       previousModified: Date(timeIntervalSince1970: 1_760_000_000),
                                       record: record)
        XCTAssertTrue(status.current.present)
        XCTAssertEqual(status.current.bytes, 4_083_117)
        XCTAssertEqual(status.current.build, "441aadc")
        XCTAssertEqual(status.current.fileName, "LlamaServer-unsigned.ipa")
        XCTAssertTrue(status.previous.present)
        XCTAssertEqual(status.previous.bytes, 4_083_116)
        XCTAssertNil(status.previous.build, "only the current slot is described by latest.json")
        XCTAssertEqual(status.uploadedAt, record.uploadedAt)

        // And the whole body survives the encoder the handler will use.
        let data = try UpdateSpec.encode(status)
        let decoded = try JSONDecoder.iso8601.decode(UpdateSpec.Status.self, from: data)
        XCTAssertEqual(decoded.current.bytes, 4_083_117)
        XCTAssertEqual(decoded.previous.bytes, 4_083_116)
    }

    func testStatusReportsAnEmptySlotAsAbsent() {
        let status = UpdateSpec.status(currentBytes: nil, currentModified: nil,
                                       previousBytes: nil, previousModified: nil, record: nil)
        XCTAssertFalse(status.current.present)
        XCTAssertNil(status.current.bytes)
        XCTAssertNil(status.current.build)
        XCTAssertFalse(status.previous.present)
        XCTAssertNil(status.uploadedAt)
    }
}

private extension JSONDecoder {
    /// The same ISO-8601 strategy `UpdateSpec.encode` writes with. Read back
    /// through the public encoder/decoder pair so the test fails if the two ever
    /// disagree on dates.
    static var iso8601: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
