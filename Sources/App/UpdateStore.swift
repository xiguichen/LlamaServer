import Foundation

/// The on-device half of the self-upgrade surface: everything that touches
/// `Documents/updates/`.
///
/// Foundation-only (no Telegraph, no llama) so the rotation is unit-tested on the
/// simulator: the file an upload replaces is the *only* copy of a working build,
/// and the guarantees below are what a user gets on a bad upload.
///
/// Two orderings are load-bearing:
/// - The caller validates **before** calling `store` (`UpdateSpec.rejection`), so
///   a refused upload — no `?build=`, over the cap, not a zip — never runs any of
///   this and leaves both slots exactly as they were.
/// - `store` rotates (current → `.prev`) **before** writing the new body, and
///   writes the body with `.atomic` (a complete temp file renamed into place).
///   A write that fails therefore leaves the last good build in `.prev` rather
///   than a half-written current file. `latest.json` is written last, so a crash
///   between the two leaves a new IPA carrying the previous build id — visible in
///   `GET /v1/updates`, never a decode failure (`decodeRecord` returns nil for
///   anything unreadable).
struct UpdateStore {
    /// The app container's Documents directory; `updates/` lives under it.
    let documentsPath: String

    var locations: UpdateSpec.Locations {
        UpdateSpec.locations(documentsPath: documentsPath)
    }

    /// One slot's filesystem facts; `nil` bytes = the file is not there.
    private func slot(_ path: String) -> (bytes: Int?, modifiedAt: Date?) {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path) else {
            return (nil, nil)
        }
        return ((attrs[.size] as? NSNumber)?.intValue, attrs[.modificationDate] as? Date)
    }

    /// What is on disk right now, as `GET /v1/updates` reports it. Sizes and
    /// timestamps come from the filesystem, so a build dropped in by hand (Files
    /// app) is reported too; only the build id needs `latest.json`.
    func status() -> UpdateSpec.Status {
        let paths = locations
        let current = slot(paths.current)
        let previous = slot(paths.previous)
        let record = (try? Data(contentsOf: URL(fileURLWithPath: paths.metadata)))
            .flatMap(UpdateSpec.decodeRecord)
        return UpdateSpec.status(currentBytes: current.bytes,
                                 currentModified: current.modifiedAt,
                                 previousBytes: previous.bytes,
                                 previousModified: previous.modifiedAt,
                                 record: record)
    }

    /// Store `body` as the current IPA, rotating the build it replaces into the
    /// single `.prev` slot, and record `build` in `latest.json`.
    ///
    /// - Throws: any filesystem error (the directory cannot be created, the write
    ///   is refused). Nothing is removed before its replacement is in hand.
    func store(body: Data, build: String) throws {
        let fm = FileManager.default
        let paths = locations
        try fm.createDirectory(atPath: paths.directory, withIntermediateDirectories: true)

        // Rotate first: the rollback copy must exist before the new build
        // replaces it.
        if fm.fileExists(atPath: paths.current) {
            if fm.fileExists(atPath: paths.previous) {
                try fm.removeItem(atPath: paths.previous)
            }
            try fm.moveItem(atPath: paths.current, toPath: paths.previous)
        }

        try body.write(to: URL(fileURLWithPath: paths.current), options: .atomic)

        let record = UpdateSpec.Record(build: build, bytes: body.count, uploadedAt: Date())
        try UpdateSpec.encode(record)
            .write(to: URL(fileURLWithPath: paths.metadata), options: .atomic)
    }
}
