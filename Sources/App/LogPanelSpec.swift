import Foundation

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

    /// Append a display-ready line to a bounded panel buffer.
    static func append(_ line: String, to lines: inout [String]) {
        lines.append(truncate(line))
        if lines.count > maxLines { lines.removeFirst(lines.count - maxLines) }
    }
}
