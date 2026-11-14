import Foundation

/// One row in the in-app panel.
///
/// `id` is a monotonically increasing sequence number, not the row's index in
/// the capped buffer. SwiftUI keys `ForEach` by it, so when the buffer drops its
/// oldest line the remaining rows keep their identity: the batch becomes one
/// removal plus insertions at the tail. With index identity (the old
/// `ForEach(lines.indices, id: \.self)`) a full buffer rewrote the content of
/// every row on every batch, re-laying-out the panel while requests streamed
/// log lines.
struct LogPanelLine: Identifiable, Equatable {
    let id: Int
    let text: String
}

/// Rendering budget for the in-app log panel.
///
/// Why this exists: the panel was the app's heaviest CPU consumer. A
/// `cpu_resource` report (bug_type 202, 90 s CPU over 157 s = 57 %, limit
/// 50 % over 180 s) showed the main thread's hottest stack in
/// `SwiftUI -> CoreText -> libicucore` — text layout. The panel rendered up to
/// 500 lines in a non-lazy `VStack` with `.textSelection(.enabled)` on every
/// one, and at `.verbose` a single line could be a 72 k-character full prompt,
/// re-laid-out on every 10 Hz mirror batch.
///
/// The panel now renders only basic (error/warn/info) lines, in a `LazyVStack`,
/// capped by count and by line length, so no single payload can blow up layout.
/// The full text always remains in the log files and the Share sheet.
enum LogPanelSpec {
    /// Most lines the panel keeps and renders.
    static let maxLines = 500
    /// Most characters of one line the panel renders.
    static let maxLineLength = 500

    /// The level tag of a formatted line (e.g. `.info` for a line tagged
    /// `[INFO   ]`).
    ///
    /// `FileLogger` formats every line as `[yyyy-MM-dd HH:mm:ss.SSS] [TAG] msg`,
    /// so the tag is the second bracketed field. Parse that rather than searching
    /// the whole line, which a message body could accidentally match.
    static func level(ofLine line: String) -> LogLevel? {
        guard line.hasPrefix("["), let close = line.dropFirst().firstIndex(of: "]") else { return nil }
        let rest = line[line.index(after: close)...]
        for level in LogLevel.allCases where rest.hasPrefix(" [\(level.tag)]") {
            return level
        }
        return nil
    }

    /// Whether a level is one the panel renders: the "basic" log.
    static func isBasic(_ level: LogLevel) -> Bool { level <= .info }

    /// Whether a formatted line is one the panel renders.
    static func isBasic(line: String) -> Bool {
        guard let level = level(ofLine: line) else { return false }
        return isBasic(level)
    }

    /// Cap one line to the render budget, noting how much was hidden so the user
    /// knows to open the file for the rest.
    static func truncate(_ line: String) -> String {
        guard line.count > maxLineLength else { return line }
        let hidden = line.count - maxLineLength
        return "\(line.prefix(maxLineLength))… [+\(hidden) chars — open the log file]"
    }

    /// Append a display-ready line to a bounded panel buffer, assigning it the
    /// next sequence id.
    static func append(_ line: String, to lines: inout [LogPanelLine], nextID: inout Int) {
        append(contentsOf: [line], to: &lines, nextID: &nextID)
    }

    /// Append a display-ready batch to a bounded panel buffer. One trim for the
    /// whole batch, which is how `FileLogger` mirrors a flush. Each row gets a
    /// fresh sequence id, so trimming the front never re-identifies the rest.
    static func append(contentsOf newLines: [String], to lines: inout [LogPanelLine],
                       nextID: inout Int) {
        guard !newLines.isEmpty else { return }
        for line in newLines {
            lines.append(LogPanelLine(id: nextID, text: truncate(line)))
            nextID += 1
        }
        if lines.count > maxLines { lines.removeFirst(lines.count - maxLines) }
    }
}
