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
///                the server log alone, without the client's logs. Routed to
///                `llamaserver-payloads.log` and never mirrored to the UI.
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

/// Crash-survivable logger. Appends timestamped, level-tagged lines to files in
/// the app's Documents directory, one file per subsystem (`LogCategory`):
///
///   - `llamaserver.log`           server: HTTP serving + app lifecycle
///   - `llamaserver-inference.log` engine: model load, KV cache, MTP/plain stats
///   - `llamaserver-payloads.log`  the bulky `.verbose` request/prompt/response dumps
///
/// Crash-critical levels (error/warn) are written AND fsynced synchronously so
/// the last breadcrumb before a native abort (GGML_ABORT / jetsam SIGKILL) is on
/// disk before log() returns; routine levels (info/debug/verbose) are buffered
/// and fsync'd at most once per `diskSyncInterval`/`diskSyncBytes` so the
/// per-token inference path and the model-load log flood never block on storage
/// I/O.
///
/// The logs are readable on-device with NO Mac required:
///   Files app -> On My iPhone -> LlamaServer -> llamaserver.log
/// (the app sets UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace), and
/// via the in-app "Share log" button (which shares every file).
///
/// Set `minimumLevel` (from the UI) to control how much is captured. At `.verbose`
/// the server records the full request/response payloads it exchanges with the
/// client, so any issue can be diagnosed from these logs alone.
final class FileLogger {
    static let shared = FileLogger()

    /// Public so the UI can offer it to ShareLink / show its path. This is the
    /// basic ("server") log — the one the app documents for on-device reading.
    var fileURL: URL { url(for: .server) }

    /// Every log file, for the Share sheet.
    var allFileURLs: [URL] { LogCategory.allCases.map(url(for:)) }

    /// Called on the main thread for every BASIC line (error/warn/info) that
    /// passes the level filter, with the fully formatted line. The UI sets this
    /// to mirror the on-disk log live in the in-app panel. Debug/verbose lines
    /// are never mirrored: they are payload-sized, and laying them out was what
    /// made the panel the app's heaviest CPU consumer.
    var onLine: ((String) -> Void)?

    private let queue = DispatchQueue(label: "llamaserver.filelogger")
    private let maxBytes = 1_000_000          // rotate each file at ~1 MB
    private let documentsDirectory: URL
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
    private var writeHandles: [LogCategory: FileHandle] = [:]
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
        documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

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

    /// Path of one category's file in the app's Documents directory.
    private func url(for category: LogCategory) -> URL {
        documentsDirectory.appendingPathComponent(category.fileName)
    }

    // MARK: - Level-tagged logging

    // Each convenience method checks the level BEFORE evaluating its message, so
    // an expensive payload dump (e.g. a full request body) is never built when
    // that level is disabled. `file` (which the compiler fills in with #file)
    // selects the destination file via `LogRouter`.
    func error(_ message: @autoclosure () -> String, file: String = #file) {
        guard LogLevel.error <= minimumLevel else { return }
        log(message(), level: .error, file: file)
    }
    func warn(_ message: @autoclosure () -> String, file: String = #file) {
        guard LogLevel.warn <= minimumLevel else { return }
        log(message(), level: .warn, file: file)
    }
    func info(_ message: @autoclosure () -> String, file: String = #file) {
        guard LogLevel.info <= minimumLevel else { return }
        log(message(), level: .info, file: file)
    }
    func debug(_ message: @autoclosure () -> String, file: String = #file) {
        guard LogLevel.debug <= minimumLevel else { return }
        log(message(), level: .debug, file: file)
    }
    func verbose(_ message: @autoclosure () -> String, file: String = #file) {
        guard LogLevel.verbose <= minimumLevel else { return }
        log(message(), level: .verbose, file: file)
    }

    /// Append a line at `level`. The message is only built and written when the
    /// level passes the filter (`@autoclosure` keeps expensive payload dumps free
    /// when they're disabled). Per `flushPolicy(for:)`, error/warn are on disk
    /// (fsynced) before returning; routine levels never block the caller.
    func log(_ message: @autoclosure () -> String, level: LogLevel = .info, file: String = #file) {
        guard level <= minimumLevel else { return }
        let msg = message()
        let category = LogRouter.category(level: level, sourceFile: file)
        if Self.flushPolicy(for: level) == .immediate {
            queue.sync { self.writeLine(msg, level: level, category: category, syncNow: true) }
        } else {
            queue.async { self.writeLine(msg, level: level, category: category, syncNow: false) }
        }
    }

    /// Formats and writes one line. Runs ONLY on `queue` (which also
    /// serializes DateFormatter usage — it is not thread-safe).
    private func writeLine(_ msg: String, level: LogLevel, category: LogCategory, syncNow: Bool) {
        let line = "[\(dateFormatter.string(from: Date()))] [\(level.tag)] \(msg)\n"
        if Self.shouldNSLog(level) {
            NSLog("[LlamaServer] %@", msg)
        }
        guard let data = line.data(using: .utf8), let handle = openWriteHandle(for: category) else { return }
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
        // Queue the basic lines for the coalesced on-screen mirror. Payload/debug
        // lines are disk-only: the panel never renders them.
        if LogPanelSpec.isBasic(level) {
            mirrorLines.append(String(line.dropLast()))
            scheduleMirrorFlush()
        }
    }

    /// Lazily opens (and creates) the append handle for one file; kept open
    /// across lines (the old code opened + seeked + closed per line).
    private func openWriteHandle(for category: LogCategory) -> FileHandle? {
        if let handle = writeHandles[category] { return handle }
        let fm = FileManager.default
        let path = url(for: category)
        if !fm.fileExists(atPath: path.path) {
            fm.createFile(atPath: path.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: path) else { return nil }
        _ = try? handle.seekToEnd()
        writeHandles[category] = handle
        return handle
    }

    /// fsync every buffered file. Callers are on `queue`. Error/warn lines are
    /// rare, so syncing all handles when one trips the threshold is fine.
    private func syncToDisk() {
        guard pendingSyncBytes > 0 else { return }
        for handle in writeHandles.values { try? handle.synchronize() }
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

    /// Return the last `maxLines` BASIC (error/warn/info) lines across the
    /// non-payload files, oldest first — what the in-app panel shows. Debug and
    /// verbose lines are skipped (they are payload-sized and the panel does not
    /// render them), so this never feeds a huge line to text layout.
    func tailBasic(maxLines: Int = LogPanelSpec.maxLines) -> [String] {
        queue.sync {
            syncToDisk()   // make buffered lines visible to the read
            var lines: [String] = []
            for category in [LogCategory.server, .inference] {
                guard let text = try? String(contentsOf: url(for: category), encoding: .utf8) else { continue }
                let basic = text.split(separator: "\n", omittingEmptySubsequences: true)
                    .map(String.init)
                    .filter(LogPanelSpec.isBasic(line:))
                lines.append(contentsOf: basic.suffix(maxLines))
            }
            // Every line starts with "[yyyy-MM-dd HH:mm:ss.SSS]", so a string
            // sort is a chronological merge of the two files.
            lines.sort()
            return Array(lines.suffix(maxLines))
        }
    }

    // MARK: - Fetching the files (GET /logs)

    /// Path of the one rolled-over backup kept per category (`rotateIfNeeded`).
    /// `llamaserver.log` → `llamaserver.1.log`.
    private func backupURL(for category: LogCategory) -> URL {
        url(for: category).deletingPathExtension().appendingPathExtension("1.log")
    }

    /// Snapshot the log files for `GET /logs` (see `LogFetchSpec`): for each
    /// requested category, the rolled-over backup (older) followed by the live
    /// file. Buffered lines are flushed first so the caller sees everything that
    /// has been written, not just what the 1 s / 64 KB policy has synced.
    func snapshot(_ plan: LogFetchSpec.Plan) -> [LogFetchSpec.Chunk] {
        queue.sync {
            syncToDisk()
            return plan.categories.map { category in
                let backup = plan.includeBackups
                    ? (try? String(contentsOf: backupURL(for: category), encoding: .utf8)) ?? ""
                    : ""
                let live = (try? String(contentsOf: url(for: category), encoding: .utf8)) ?? ""
                return LogFetchSpec.chunk(name: category.fileName, backupText: backup,
                                          liveText: live, plan: plan)
            }
        }
    }

    func clear() {
        queue.sync {
            syncToDisk()
            for handle in writeHandles.values { try? handle.close() }
            writeHandles = [:]
            pendingSyncBytes = 0
            mirrorLines = []
            for category in LogCategory.allCases {
                try? Data().write(to: url(for: category), options: .atomic)
            }
        }
    }

    /// Keep one rolled-over backup per file so none can grow unbounded.
    private func rotateIfNeeded() {
        let fm = FileManager.default
        for category in LogCategory.allCases {
            let path = url(for: category)
            guard let attrs = try? fm.attributesOfItem(atPath: path.path),
                  let size = (attrs[.size] as? NSNumber)?.intValue,
                  size > maxBytes else { continue }
            let backup = backupURL(for: category)
            try? fm.removeItem(at: backup)
            try? fm.moveItem(at: path, to: backup)
        }
    }
}
