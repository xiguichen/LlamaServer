import Foundation
import os

/// Severity of a log line. Ordered from least to most verbose; the logger emits
/// a line only when its level is at or below the configured `minimumLevel`
/// (e.g. `minimumLevel == .debug` emits error/warn/info/debug but not verbose).
///
/// Guidance for choosing a level when logging:
///   - `.error`   a request/operation failed or produced an invalid result.
///   - `.warn`    recoverable or suspicious condition (memory pressure, empty output).
///   - `.info`    user-meaningful lifecycle (request received, generation done).
///   - `.debug`   per-stage breadcrumbs (each SSE chunk stage, KV cache decisions).
///   - `.verbose` full payloads: raw request bodies, full prompts, full responses,
///                per-chunk content — everything needed to diagnose an issue from
///                the server log alone, without the client's logs.
enum LogLevel: Int, CaseIterable, Comparable, Identifiable {
    case error = 0
    case warn
    case info
    case debug
    case verbose

    var id: Int { rawValue }

    static func < (lhs: LogLevel, rhs: LogLevel) -> Bool { lhs.rawValue < rhs.rawValue }

    /// Fixed-width tag written into each line, e.g. `[INFO   ]`.
    var tag: String {
        switch self {
        case .error:   return "ERROR  "
        case .warn:    return "WARN   "
        case .info:    return "INFO   "
        case .debug:   return "DEBUG  "
        case .verbose: return "VERBOSE"
        }
    }

    /// Human-friendly name for the UI picker.
    var label: String {
        switch self {
        case .error:   return "Error"
        case .warn:    return "Warning"
        case .info:    return "Info"
        case .debug:   return "Debug"
        case .verbose: return "Verbose"
        }
    }

    /// One-line description shown under the UI picker.
    var detail: String {
        switch self {
        case .error:   return "Only failures."
        case .warn:    return "Failures and suspicious conditions."
        case .info:    return "Normal activity (default)."
        case .debug:   return "Per-stage breadcrumbs for diagnosing flow."
        case .verbose: return "Everything: full requests, prompts, and responses."
        }
    }
}

/// Crash-survivable logger. Appends timestamped, level-tagged lines to a file in
/// the app's Documents directory, flushing each line to disk synchronously so the
/// last breadcrumb before a native abort (GGML_ABORT / jetsam SIGKILL) is preserved.
///
/// The log is readable on-device with NO Mac required:
///   Files app -> On My iPhone -> LlamaServer -> llamaserver.log
/// (the app sets UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace), and
/// via the in-app "Share log" button.
///
/// Set `minimumLevel` (from the UI) to control how much is captured. At `.verbose`
/// the server records the full request/response payloads it exchanges with the
/// client, so any issue can be diagnosed from this log alone.
final class FileLogger {
    static let shared = FileLogger()

    /// Public so the UI can offer it to ShareLink / show its path.
    let fileURL: URL

    /// Called on the main thread for every line that passes the level filter,
    /// with the fully formatted line (timestamp + tag + message). The UI sets
    /// this to mirror the on-disk log live in the in-app panel.
    var onLine: ((String) -> Void)?

    private let queue = DispatchQueue(label: "llamaserver.filelogger")
    private let maxBytes = 1_000_000          // rotate at ~1 MB
    private let dateFormatter: DateFormatter

    private static let levelDefaultsKey = "FileLogger.minimumLevel"
    private let levelLock = OSAllocatedUnfairLock<LogLevel>(initialState: .info)

    /// The most verbose level that will be written. Persisted across launches.
    var minimumLevel: LogLevel {
        get { levelLock.withLock { $0 } }
        set {
            levelLock.withLock { $0 = newValue }
            UserDefaults.standard.set(newValue.rawValue, forKey: Self.levelDefaultsKey)
        }
    }

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("llamaserver.log")

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        dateFormatter = df

        if let stored = UserDefaults.standard.object(forKey: Self.levelDefaultsKey) as? Int,
           let level = LogLevel(rawValue: stored) {
            levelLock.withLock { $0 = level }
        }

        rotateIfNeeded()
    }

    // MARK: - Level-tagged logging

    // Each convenience method checks the level BEFORE evaluating its message, so
    // an expensive payload dump (e.g. a full request body) is never built when
    // that level is disabled.
    func error(_ message: @autoclosure () -> String) {
        guard LogLevel.error <= minimumLevel else { return }
        log(message(), level: .error)
    }
    func warn(_ message: @autoclosure () -> String) {
        guard LogLevel.warn <= minimumLevel else { return }
        log(message(), level: .warn)
    }
    func info(_ message: @autoclosure () -> String) {
        guard LogLevel.info <= minimumLevel else { return }
        log(message(), level: .info)
    }
    func debug(_ message: @autoclosure () -> String) {
        guard LogLevel.debug <= minimumLevel else { return }
        log(message(), level: .debug)
    }
    func verbose(_ message: @autoclosure () -> String) {
        guard LogLevel.verbose <= minimumLevel else { return }
        log(message(), level: .verbose)
    }

    /// Append a line at `level`. The message is only built and written when the
    /// level passes the filter (`@autoclosure` keeps expensive payload dumps free
    /// when they're disabled). Synchronous + fsync so it is on disk before
    /// returning — important when the next thing the caller does may crash.
    func log(_ message: @autoclosure () -> String, level: LogLevel = .info) {
        guard level <= minimumLevel else { return }
        let msg = message()
        let line = "[\(dateFormatter.string(from: Date()))] [\(level.tag)] \(msg)\n"
        // Also emit to the system log (visible in Console.app when a Mac IS available).
        NSLog("[LlamaServer] %@", msg)

        queue.sync {
            guard let data = line.data(using: .utf8) else { return }
            let fm = FileManager.default
            if !fm.fileExists(atPath: fileURL.path) {
                try? data.write(to: fileURL, options: .atomic)
            } else if let handle = try? FileHandle(forWritingTo: fileURL) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.synchronize()   // fsync — guarantee it survives a crash
            }
        }

        // Mirror to the in-app panel (without the trailing newline).
        if let onLine = onLine {
            let display = String(line.dropLast())
            DispatchQueue.main.async { onLine(display) }
        }
    }

    /// Return the last `maxLines` lines for display in the UI.
    func tail(maxLines: Int = 200) -> [String] {
        queue.sync {
            guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { return [] }
            let lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
            return Array(lines.suffix(maxLines))
        }
    }

    func clear() {
        queue.sync {
            try? Data().write(to: fileURL, options: .atomic)
        }
    }

    /// Keep one rolled-over backup so the file can't grow unbounded.
    private func rotateIfNeeded() {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
              let size = (attrs[.size] as? NSNumber)?.intValue,
              size > maxBytes else { return }
        let backup = fileURL.deletingPathExtension().appendingPathExtension("1.log")
        try? fm.removeItem(at: backup)
        try? fm.moveItem(at: fileURL, to: backup)
    }
}
