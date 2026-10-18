import Foundation

/// Plans a `GET /logs` response, and serializes it.
///
/// The log files live in the app's Documents directory on the device, which is
/// awkward to reach from a LAN client (Files app, share sheet, cable). This
/// endpoint exists so a debugging client — a script, an agent, `curl` — can pull
/// them off the *running* server:
///
///     curl 'http://<device>:8443/logs?name=inference&tail=500'
///     curl 'http://<device>:8443/logs?format=json' > logs.json
///
/// Query, all optional (unknown values are ignored rather than rejected — a typo
/// should still return logs):
///   - `name=`     comma-separated `server`, `inference`, `payloads`, or `all`.
///                 Default: all three.
///   - `tail=`     lines kept per file, counted from the end. Default: the whole
///                 file (each file rotates at 1 MB, so that stays bounded).
///   - `backups=0` drop the rolled-over `*.1.log`, which holds the lines that
///                 rotated out when the live file crossed 1 MB. Default: include.
///   - `format=`   `text` (default) or `json`.
///
/// Pure and unit-tested (`LogFetchSpecTests`); reading the files is
/// `FileLogger.snapshot(_:)`.
enum LogFetchSpec {

    /// How the response body is serialized.
    enum Format: String {
        /// A `===== name — N lines =====` header per file, then its content:
        /// what a human (or an agent reading the response) wants.
        case text
        /// A JSON array of `Chunk`s, for scripts.
        case json
    }

    struct Plan: Equatable {
        var categories: [LogCategory]
        /// Lines kept per file, counted from the end; `nil` = whole file.
        var tail: Int?
        /// Include the single rolled-over `*.1.log` per category.
        var includeBackups: Bool
        var format: Format

        static let `default` = Plan(categories: LogCategory.allCases, tail: nil,
                                    includeBackups: true, format: .text)
    }

    /// One file's contribution to the response.
    struct Chunk: Equatable, Codable {
        /// File name, e.g. `llamaserver-inference.log`.
        let name: String
        /// Lines this chunk carries.
        let lines: Int
        /// Bytes of `content`.
        let bytes: Int
        /// Lines were dropped to honour `tail`.
        let truncated: Bool
        let content: String
    }

    // MARK: - Request

    static func plan(queryItems: [URLQueryItem]) -> Plan {
        var plan = Plan.default
        for item in queryItems {
            let value = (item.value ?? "").lowercased()
            switch item.name.lowercased() {
            case "name", "names", "category", "categories":
                let wanted = value.split(separator: ",")
                    .map { String($0).trimmingCharacters(in: .whitespaces) }
                guard !wanted.isEmpty else { continue }
                if wanted.contains("all") {
                    plan.categories = LogCategory.allCases
                } else {
                    let picked = LogCategory.allCases.filter { wanted.contains($0.rawValue) }
                    if !picked.isEmpty { plan.categories = picked }
                }
            case "tail", "lines":
                if let n = Int(value) { plan.tail = max(0, n) }
            case "backup", "backups":
                plan.includeBackups = !["0", "false", "no"].contains(value)
            case "format":
                if let f = Format(rawValue: value) { plan.format = f }
            default:
                break
            }
        }
        return plan
    }

    // MARK: - Response

    /// Combine one category's rolled-over backup and its live file into the
    /// chunk the response carries, applying `plan.tail`. `backupText` is empty
    /// when there is no backup (or backups were excluded).
    ///
    /// The backup is *older*, so it comes first: the joined text is the file's
    /// history in order, and a tail slice is therefore the newest lines.
    static func chunk(name: String, backupText: String = "", liveText: String = "",
                      plan: Plan) -> Chunk {
        // Drop the empty element a trailing newline leaves behind, so `lines`
        // counts real lines rather than the final newline.
        var lines = (backupText + liveText)
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if lines.last?.isEmpty == true { lines.removeLast() }
        let truncated = plan.tail.map { lines.count > $0 } ?? false
        let kept = plan.tail.map { Array(lines.suffix($0)) } ?? lines
        let content = kept.isEmpty ? "" : kept.joined(separator: "\n") + "\n"
        return Chunk(name: name, lines: kept.count, bytes: content.utf8.count,
                     truncated: truncated, content: content)
    }

    /// The `text` body: one header line per file, then its lines verbatim, so
    /// the result can be read (or `grep`ped) exactly like the on-device file.
    static func textBody(_ chunks: [Chunk]) -> String {
        chunks.map { chunk in
            let note = chunk.truncated ? ", last \(chunk.lines) lines" : ""
            let content = chunk.content.isEmpty || chunk.content.hasSuffix("\n")
                ? chunk.content : chunk.content + "\n"
            return "===== \(chunk.name) — \(chunk.lines) lines, \(chunk.bytes) bytes\(note) =====\n\(content)"
        }.joined()
    }

    /// The `json` body: the chunks as an array, so a script can read them field
    /// by field instead of parsing headers.
    static func jsonBody(_ chunks: [Chunk]) throws -> Data {
        try JSONEncoder().encode(chunks)
    }
}
