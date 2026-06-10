import Foundation

/// Crash-survivable logger. Appends timestamped lines to a file in the app's
/// Documents directory, flushing each line to disk synchronously so the last
/// breadcrumb before a native abort (GGML_ABORT / jetsam SIGKILL) is preserved.
///
/// The log is readable on-device with NO Mac required:
///   Files app -> On My iPhone -> LlamaServer -> llamaserver.log
/// (the app sets UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace), and
/// via the in-app "Share log" button.
final class FileLogger {
    static let shared = FileLogger()

    /// Public so the UI can offer it to ShareLink / show its path.
    let fileURL: URL

    private let queue = DispatchQueue(label: "llamaserver.filelogger")
    private let maxBytes = 1_000_000          // rotate at ~1 MB
    private let dateFormatter: DateFormatter

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("llamaserver.log")

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        dateFormatter = df

        rotateIfNeeded()
    }

    /// Append a line. Synchronous + fsync so it is on disk before returning —
    /// important when the next thing the caller does may crash the process.
    func log(_ message: String) {
        let line = "[\(dateFormatter.string(from: Date()))] \(message)\n"
        // Also emit to the system log (visible in Console.app when a Mac IS available).
        NSLog("[LlamaServer] %@", message)

        queue.sync {
            guard let data = line.data(using: .utf8) else { return }
            let fm = FileManager.default
            if !fm.fileExists(atPath: fileURL.path) {
                try? data.write(to: fileURL, options: .atomic)
                return
            }
            guard let handle = try? FileHandle(forWritingTo: fileURL) else { return }
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
            try? handle.synchronize()   // fsync — guarantee it survives a crash
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
