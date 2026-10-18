import Foundation

/// Pure contract for the self-upgrade upload surface: `PUT /v1/updates/ipa`
/// stores a CI-built IPA on the phone, `GET /v1/updates` reports what is stored.
///
/// Why the *inference* server carries the route: the app has no other listener.
/// A second port would mean a second local-network permission prompt, a second
/// failure mode, and — worst — a server that is down exactly when you need to
/// replace the build. Riding the existing listener has one real consequence: the
/// routes exist only while a model is loaded. With the engine stopped (or a
/// model that will not start) a client gets connection-refused, and the escape
/// hatch is to pick a smaller model in the UI — not to reinstall over the
/// network.
///
/// Storage: `Documents/updates/`, inside the app container. `UIFileSharingEnabled`
/// + `LSSupportsOpeningDocumentsInPlace` (project.yml) expose Documents in the
/// Files app, so the flow is: `scripts/upload-ipa.sh` pushes the IPA → the user
/// opens Files → taps the `.ipa` → AltStore installs it in place. No cable, no
/// AirDrop, no AltServer, and the install preserves the container (the models
/// stay). Note the consequence: iOS deletes the whole container on uninstall, so
/// the stored IPA and the models go with it. Nothing in-container survives that.
///
/// Naming, deliberately: the current IPA keeps the name CI gives it
/// (`LlamaServer-unsigned.ipa`) instead of carrying the build SHA. Files sorts by
/// name, and a hash-suffixed name makes the newest build impossible to spot — the
/// user would have to work out which hash is newer before deciding what to tap.
/// The build identity lives in `latest.json` and in the response/log instead; the
/// previous IPA rotates to `LlamaServer-unsigned.prev.ipa`, a one-slot rollback
/// for a build that turns out to be broken on device.
///
/// Pure: Foundation only, no `FileManager`, no Telegraph (the tests target in
/// `project.yml` compiles neither). Every guard is decided here and unit-tested
/// without a device; the caller does the I/O.
enum UpdateSpec {

    /// Directory under `Documents/` (created on demand).
    static let directoryName = "updates"
    /// The file name CI produces (`.github/workflows/build.yml`) and the one the
    /// user taps. Deliberately hash-free — see the type comment.
    static let currentFileName = "LlamaServer-unsigned.ipa"
    /// The single backup slot, overwritten on every upload.
    static let previousFileName = "LlamaServer-unsigned.prev.ipa"
    /// Build metadata for the current file, so the *kind* of build stays
    /// identifiable even though the file name does not carry it.
    static let metadataFileName = "latest.json"
    /// Largest accepted upload. The IPA is ~3.8 MB today (CI artifact 4,083,117
    /// bytes); this is room for the app to grow, not a target. The limit is
    /// best-effort: Telegraph hands the handler a fully buffered body, so an
    /// oversized upload is *refused* here but has already been read into memory.
    static let maxUploadBytes = 128 * 1024 * 1024
    /// An IPA is a zip. Require the header so a failed download — an HTML error
    /// page, a truncated file — cannot overwrite a working build with something
    /// AltStore cannot install.
    static let zipMagic: [UInt8] = [0x50, 0x4B, 0x03, 0x04]

    /// Where the three files live. `documentsPath` is the app container's
    /// Documents directory (`FileManager` owns that lookup; this stays pure).
    struct Locations: Equatable {
        let directory: String
        let current: String
        let previous: String
        let metadata: String
    }

    static func locations(documentsPath: String) -> Locations {
        let dir = (documentsPath as NSString).appendingPathComponent(directoryName)
        return Locations(
            directory: dir,
            current: (dir as NSString).appendingPathComponent(currentFileName),
            previous: (dir as NSString).appendingPathComponent(previousFileName),
            metadata: (dir as NSString).appendingPathComponent(metadataFileName))
    }

    // MARK: - Request

    /// `?build=<sha>` — the build identity to record. The parameter *name* is
    /// matched case-insensitively (like `LogFetchSpec`); the value is trimmed but
    /// keeps its case, because a build id may be a tag rather than hex.
    static func build(fromQueryItems items: [URLQueryItem]) -> String? {
        for item in items where item.name.lowercased() == "build" {
            let value = (item.value ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value }
        }
        return nil
    }

    /// Why an upload was refused. Kept as a value rather than a thrown error or a
    /// status code, so the decision is testable without Telegraph (the tests
    /// target cannot see `HTTPStatus`); the server maps `.tooLarge` to 413 and the
    /// rest to 400.
    enum Rejection: Equatable {
        /// No usable `?build=` — naming the build is the record's whole purpose.
        case missingBuild
        case tooLarge(bytes: Int)
        case notAnIPA(bytes: Int)

        /// Client-facing text, carrying the raw numbers so a script or a human can
        /// act on it without reading the phone's log.
        var message: String {
            switch self {
            case .missingBuild:
                return "missing ?build=<sha> query parameter"
            case .tooLarge(let bytes):
                return "upload is \(bytes) bytes; the cap is \(maxUploadBytes) bytes"
            case .notAnIPA(let bytes):
                return "body is not an IPA (no zip header): \(bytes) bytes received"
            }
        }
    }

    /// `nil` = accept. The order matters, and so does the fact that nothing here
    /// touches the filesystem: the file an upload replaces is the only copy of a
    /// working build, so every refusal is decided *before* the caller rotates
    /// anything.
    static func rejection(build: String?, body: Data) -> Rejection? {
        guard let build, !build.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .missingBuild
        }
        guard body.count <= maxUploadBytes else { return .tooLarge(bytes: body.count) }
        guard body.starts(with: zipMagic) else { return .notAnIPA(bytes: body.count) }
        return nil
    }

    // MARK: - Stored state

    /// What `latest.json` holds: which build is in the current slot.
    struct Record: Codable, Equatable {
        let build: String
        let bytes: Int
        let uploadedAt: Date
    }

    /// The `GET /v1/updates` body: what is on disk in each slot, plus the recorded
    /// upload. `bytes`/`modifiedAt` come from the filesystem, so a file dropped in
    /// by hand (Files app) is reported too; only `build` comes from
    /// `latest.json`.
    struct Status: Codable, Equatable {
        struct Slot: Codable, Equatable {
            let fileName: String
            let present: Bool
            let bytes: Int?
            let modifiedAt: Date?
            let build: String?
        }
        let current: Slot
        let previous: Slot
        let uploadedAt: Date?
    }

    static func status(currentBytes: Int?, currentModified: Date?,
                       previousBytes: Int?, previousModified: Date?,
                       record: Record?) -> Status {
        Status(current: Status.Slot(fileName: currentFileName, present: currentBytes != nil,
                                    bytes: currentBytes, modifiedAt: currentModified,
                                    build: record?.build),
               previous: Status.Slot(fileName: previousFileName, present: previousBytes != nil,
                                     bytes: previousBytes, modifiedAt: previousModified, build: nil),
               uploadedAt: record?.uploadedAt)
    }

    // MARK: - Wire format

    /// ISO-8601 dates and sorted keys, so the bytes on disk (`latest.json`), the
    /// HTTP response and the script's output all agree, and so a re-encode of the
    /// same record is byte-identical (a diff-able metadata file).
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    /// `nil` for a missing or corrupt `latest.json` — the caller then reports the
    /// slots without a build id instead of failing the request.
    static func decodeRecord(_ data: Data) -> Record? {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(Record.self, from: data)
    }
}
