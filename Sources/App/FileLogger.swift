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
/// the app's Documents directory. Crash-critical levels (error/warn) are written
/// AND fsynced synchronously so the last breadcrumb before a native abort
/// (GGML_ABORT / jetsam SIGKILL) is on disk before log() returns; routine levels
/// (info/debug/verbose) are buffered and fsync'd at most once per
/// `diskSyncInterval`/`diskSyncBytes` so the per-token inference path and the
/// model-load log flood never block on storage I/O.
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

    // MARK: Flush policy (pure, unit-tested)

    /// How a line at a given level reaches disk.
    enum FlushPolicy: Equatable {
        /// `queue.sync` + fsync before log() returns — crash-critical levels.
        case immediate
        /// Buffered append; fsync at most once per interval/size threshold.
        case buffered
    }

    static func flushPolicy(for level: LogLevel) -> FlushPolicy {
        level <= .warn ? .immediate : .buffered
    }

    /// NSLog (system log) mirror: only warnings and above. Routine lines cost
    /// ~10–100 µs each in NSLog machinery — with ~1,500 llama.cpp debug lines
    /// per model load that was real time off the critical path.
    static func shouldNSLog(_ level: LogLevel) -> Bool { level <= .warn }

    /// Buffered lines are fsynced at most this long after the first unflushed
    /// write — bounds breadcrumb loss on a hard crash to ~1s of routine lines.
    static let diskSyncInterval: TimeInterval = 1.0
    /// …or when this many buffered bytes have accumulated.
    static let diskSyncBytes = 64 * 1024
    /// The on-screen mirror delivers batches at most this often (the old
    /// per-line main-queue dispatch flooded SwiftUI during model load).
    static let mirrorInterval: TimeInterval = 0.1

    /// All of the following are mutated ONLY on `queue`.
    private var writeHandle: FileHandle?
    private var pendingSyncBytes = 0
    private var syncScheduled = false
    private var mirrorLines: [String] = []
    private var mirrorScheduled = false

    private static let levelDefaultsKey = "FileLogger.minimumLevel"
    private let levelLock = OSAllocatedUnfairLock<LogLevel>(initialState: .info)
    private static let promptKey = "FileLogger.logPromptContent"
    private let promptLock = OSAllocatedUnfairLock(initialState: false)

    /// The most verbose level that will be written. Persisted across launches.
    var minimumLevel: LogLevel {
        get { levelLock.withLock { $0 } }
        set {
            levelLock.withLock { $0 = newValue }
            UserDefaults.standard.set(newValue.rawValue, forKey: Self.levelDefaultsKey)
        }
    }

    /// Whether to log full prompt/response payloads. Separate from `minimumLevel`
    /// so a user can keep `.verbose` logging for other diagnostics without leaking
    /// sensitive or bulky prompt content. Persisted across launches.
    var logPromptContent: Bool {
        get { promptLock.withLock { $0 } }
        set {
            promptLock.withLock { $0 = newValue }
            UserDefaults.standard.set(newValue, forKey: Self.promptKey)
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
        promptLock.withLock { $0 = UserDefaults.standard.bool(forKey: Self.promptKey) }

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
    /// when they're disabled). Per `flushPolicy(for:)`, error/warn are on disk
    /// (fsynced) before returning; routine levels never block the caller.
    func log(_ message: @autoclosure () -> String, level: LogLevel = .info) {
        guard level <= minimumLevel else { return }
        let msg = message()
        if Self.flushPolicy(for: level) == .immediate {
            queue.sync { self.writeLine(msg, level: level, syncNow: true) }
        } else {
            queue.async { self.writeLine(msg, level: level, syncNow: false) }
        }
    }

    /// Formats and writes one line. Runs ONLY on `queue` (which also
    /// serializes DateFormatter usage — it is not thread-safe).
    private func writeLine(_ msg: String, level: LogLevel, syncNow: Bool) {
        let line = "[\(dateFormatter.string(from: Date()))] [\(level.tag)] \(msg)\n"
        if Self.shouldNSLog(level) {
            NSLog("[LlamaServer] %@", msg)
        }
        guard let data = line.data(using: .utf8), let handle = openWriteHandle() else { return }
        do {
            try handle.write(contentsOf: data)
        } catch {
            return
        }
        pendingSyncBytes += data.count
        if syncNow || pendingSyncBytes >= Self.diskSyncBytes {
            syncToDisk()
        } else {
            scheduleSync()
        }
        // Queue for the coalesced on-screen mirror.
        mirrorLines.append(String(line.dropLast()))
        scheduleMirrorFlush()
    }

    /// Lazily opens (and creates) the append handle; kept open across lines
    /// (the old code opened + seeked + closed per line).
    private func openWriteHandle() -> FileHandle? {
        if let writeHandle { return writeHandle }
        let fm = FileManager.default
        if !fm.fileExists(atPath: fileURL.path) {
            fm.createFile(atPath: fileURL.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: fileURL) else { return nil }
        try? handle.seekToEnd()
        writeHandle = handle
        return handle
    }

    /// fsync the buffered tail. Callers are on `queue`.
    private func syncToDisk() {
        guard pendingSyncBytes > 0 else { return }
        try? writeHandle?.synchronize()
        pendingSyncBytes = 0
    }

    private func scheduleSync() {
        guard !syncScheduled else { return }
        syncScheduled = true
        queue.asyncAfter(deadline: .now() + Self.diskSyncInterval) { [weak self] in
            guard let self else { return }
            self.syncScheduled = false
            self.syncToDisk()
        }
    }

    /// Deliver accumulated mirror lines to the UI in one main-thread batch at
    /// most every `mirrorInterval` — keeps SwiftUI from re-diffing per line
    /// during the model-load log flood.
    private func scheduleMirrorFlush() {
        guard !mirrorScheduled, !mirrorLines.isEmpty else { return }
        mirrorScheduled = true
        queue.asyncAfter(deadline: .now() + Self.mirrorInterval) { [weak self] in
            guard let self else { return }
            self.mirrorScheduled = false
            let batch = self.mirrorLines
            self.mirrorLines.removeAll(keepingCapacity: true)
            guard let onLine = self.onLine else { return }
            DispatchQueue.main.async { batch.forEach(onLine) }
        }
    }

    /// Return the last `maxLines` lines for display in the UI.
    func tail(maxLines: Int = 200) -> [String] {
        queue.sync {
            syncToDisk()   // make buffered lines visible to the read
            guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { return [] }
            let lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
            return Array(lines.suffix(maxLines))
        }
    }

    func clear() {
        queue.sync {
            syncToDisk()
            try? writeHandle?.close()
            writeHandle = nil
            pendingSyncBytes = 0
            mirrorLines = []
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
