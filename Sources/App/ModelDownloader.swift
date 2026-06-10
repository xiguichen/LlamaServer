import Foundation
import Combine

/// Downloads a `.gguf` model from an HTTP(S) URL into the model library, with
/// live progress. Uses a `URLSessionDownloadTask` so large files stream to disk
/// rather than buffering in memory.
final class ModelDownloader: NSObject, ObservableObject {

    @Published private(set) var isDownloading = false
    @Published private(set) var progress: Double = 0      // 0...1 (0 if size unknown)
    @Published private(set) var receivedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var error: String?

    /// Stored by the app delegate when iOS relaunches the app to deliver
    /// background-download completion events.
    static var backgroundCompletionHandler: (() -> Void)?

    private var task: URLSessionDownloadTask?
    private var suggestedName = "model.gguf"
    private var onComplete: ((URL?) -> Void)?
    /// Throttle UI progress publishing. didWriteData fires per network chunk
    /// (hundreds–thousands/sec); publishing each one storms SwiftUI and pegs the
    /// main thread, tripping the iOS CPU watchdog. Updated only on the (serial)
    /// delegate queue.
    private var lastProgressPublish: CFAbsoluteTime = 0
    private let progressMinInterval: CFAbsoluteTime = 0.1   // ~10 updates/sec max

    private lazy var session: URLSession = {
        // A background session lets the download continue when the screen is
        // locked or the app is backgrounded — the system transfer daemon keeps
        // going and wakes the app to finish. Avoids losing the connection on
        // auto-lock/suspend.
        let config = URLSessionConfiguration.background(withIdentifier: "llamaserver.modeldownload")
        config.isDiscretionary = false          // user-initiated: start immediately
        config.sessionSendsLaunchEvents = true  // relaunch app to finish if needed
        // Model downloads can be large/slow; allow generous resource timeout.
        config.timeoutIntervalForResource = 6 * 60 * 60
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    var progressText: String {
        let received = ByteCountFormatter.string(fromByteCount: receivedBytes, countStyle: .file)
        if totalBytes > 0 {
            let total = ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
            return "\(received) / \(total) (\(Int(progress * 100))%)"
        }
        return received
    }

    /// Starts a download. `onComplete` is called on the main thread with the
    /// destination URL on success, or `nil` on failure/cancel.
    func start(urlString: String, onComplete: @escaping (URL?) -> Void) {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            setError("Enter a valid http(s) URL")
            onComplete(nil)
            return
        }

        // Derive a filename from the URL path (query string is ignored by
        // lastPathComponent), defaulting to a .gguf name.
        suggestedName = ModelStore.shared.destinationURL(for: url.lastPathComponent).lastPathComponent
        self.onComplete = onComplete

        DispatchQueue.main.async {
            self.isDownloading = true
            self.progress = 0
            self.receivedBytes = 0
            self.totalBytes = 0
            self.error = nil
        }

        lastProgressPublish = 0   // let the first chunk publish immediately
        let task = session.downloadTask(with: url)
        self.task = task
        task.resume()
    }

    func cancel() {
        task?.cancel()
        task = nil
        let completion = onComplete
        onComplete = nil
        DispatchQueue.main.async {
            self.isDownloading = false
            completion?(nil)
        }
    }

    private func setError(_ message: String) {
        DispatchQueue.main.async {
            self.error = message
            self.isDownloading = false
        }
    }

    private func finish(_ url: URL?) {
        let completion = onComplete
        onComplete = nil
        task = nil
        DispatchQueue.main.async {
            self.isDownloading = false
            if url != nil { self.progress = 1 }
            completion?(url)
        }
    }
}

extension ModelDownloader: URLSessionDownloadDelegate {

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        // Coalesce high-frequency chunk callbacks into at most ~10 UI updates/sec
        // (always emit the final one) to avoid a SwiftUI re-render storm.
        let isFinal = totalBytesExpectedToWrite > 0 && totalBytesWritten >= totalBytesExpectedToWrite
        let now = CFAbsoluteTimeGetCurrent()
        guard isFinal || now - lastProgressPublish >= progressMinInterval else { return }
        lastProgressPublish = now

        let pct = totalBytesExpectedToWrite > 0
            ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            : 0
        DispatchQueue.main.async {
            self.receivedBytes = totalBytesWritten
            self.totalBytes = totalBytesExpectedToWrite
            self.progress = pct
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // `location` is a temp file deleted right after this returns — move it
        // into the library synchronously here.
        if let response = downloadTask.response as? HTTPURLResponse,
           !(200...299).contains(response.statusCode) {
            setError("Server returned HTTP \(response.statusCode)")
            finish(nil)
            return
        }
        do {
            let dest = ModelStore.shared.destinationURL(for: suggestedName)
            let fm = FileManager.default
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: location, to: dest)
            finish(dest)
        } catch {
            setError(error.localizedDescription)
            finish(nil)
        }
    }

    /// Called after all background events for the session have been delivered;
    /// invoke the system-provided completion handler so iOS knows we're done.
    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            let handler = ModelDownloader.backgroundCompletionHandler
            ModelDownloader.backgroundCompletionHandler = nil
            handler?()
        }
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        // Success is handled in didFinishDownloadingTo; only act on errors here.
        guard let error = error else { return }
        // Ignore explicit cancels (already handled in cancel()).
        if (error as NSError).code == NSURLErrorCancelled { return }
        setError(error.localizedDescription)
        finish(nil)
    }
}
