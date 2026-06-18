import Foundation
import Combine
import UIKit

/// Drives the UI: model library (list / import / download), selecting a model,
/// and starting/stopping the inference engine + HTTPS server.
@MainActor
final class ServerViewModel: ObservableObject {

    enum Status: Equatable {
        case stopped
        case loadingModel
        case running
        case error(String)

        var label: String {
            switch self {
            case .stopped:      return "Stopped"
            case .loadingModel: return "Loading model…"
            case .running:      return "Running"
            case .error(let m): return "Error: \(m)"
            }
        }
    }

    // Server state
    @Published private(set) var status: Status = .stopped
    @Published var port: String = "8443"
    @Published var contextSize: String = "4096"
    @Published private(set) var logs: [String] = []
    @Published private(set) var ipAddress: String? = NetworkInfo.wifiIPv4Address()

    /// Controls how much the server records. `.verbose` captures full request and
    /// response payloads so issues can be diagnosed from the server log alone.
    /// Persisted by `FileLogger` across launches.
    @Published var logLevel: LogLevel = FileLogger.shared.minimumLevel {
        didSet { FileLogger.shared.minimumLevel = logLevel }
    }

    // Model library
    @Published private(set) var models: [ModelFile] = []
    @Published var selectedModel: ModelFile?
    @Published var downloadURLString: String = ""

    /// Re-published so the UI updates on download progress.
    let downloader = ModelDownloader()

    private var inference: LlamaInference?
    private var httpServer: LlamaHTTPServer?
    private var cancellables = Set<AnyCancellable>()

    init() {
        // Forward the downloader's changes so views observing this VM refresh.
        downloader.objectWillChange
            .sink { [weak self] in self?.objectWillChange.send() }
            .store(in: &cancellables)
        // Show persisted history (incl. breadcrumbs from before any prior crash).
        logs = FileLogger.shared.tail(maxLines: 500)
        // Mirror EVERY line the server writes (not just UI events) into the panel,
        // so the in-app log matches the on-disk log at the selected level.
        FileLogger.shared.onLine = { [weak self] line in
            Task { @MainActor in self?.appendLine(line) }
        }
        refreshModels()
    }

    /// URL of the on-disk log (for ShareLink / Files app).
    var logFileURL: URL { FileLogger.shared.fileURL }

    /// Reload the persisted log into the panel (e.g. after a crash + relaunch).
    func reloadLogs() { logs = FileLogger.shared.tail(maxLines: 500) }

    func clearLogs() {
        FileLogger.shared.clear()
        logs = []
    }

    var isRunning: Bool {
        if case .running = status { return true }
        return false
    }

    var isBusy: Bool {
        if case .loadingModel = status { return true }
        return false
    }

    var serverURL: String? {
        guard let ip = ipAddress else { return nil }
        return "http://\(ip):\(port)"
    }

    // MARK: - Model library

    func refreshModels() {
        models = ModelStore.shared.list()
        // Drop selection if the file is gone.
        if let selected = selectedModel, !models.contains(selected) {
            selectedModel = models.first(where: { $0.url == selected.url })
        }
    }

    func selectModel(_ model: ModelFile) {
        guard !isRunning, !isBusy else { return }
        selectedModel = model
        log("Selected model: \(model.name)")
    }

    func importModel(from url: URL) {
        log("Importing \(url.lastPathComponent)…")
        Task.detached(priority: .userInitiated) {
            do {
                let dest = try ModelStore.shared.importModel(from: url)
                await MainActor.run {
                    self.refreshModels()
                    self.selectedModel = self.models.first { $0.url == dest }
                    self.log("Imported \(dest.lastPathComponent)")
                }
            } catch {
                await MainActor.run {
                    self.log("Import failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func deleteModels(at offsets: IndexSet) {
        guard !isRunning, !isBusy else { return }
        let targets = offsets.map { models[$0] }
        for model in targets {
            do {
                try ModelStore.shared.delete(model.url)
                log("Deleted \(model.name)")
                if selectedModel == model { selectedModel = nil }
            } catch {
                log("Delete failed: \(error.localizedDescription)")
            }
        }
        refreshModels()
    }

    // MARK: - Download

    func startDownload() {
        let urlString = downloadURLString
        guard !urlString.isEmpty else { return }
        // Keep the screen awake so auto-lock doesn't suspend us mid-download.
        UIApplication.shared.isIdleTimerDisabled = true
        log("Downloading from \(urlString)")
        downloader.start(urlString: urlString) { [weak self] dest in
            Task { @MainActor in
                guard let self else { return }
                UIApplication.shared.isIdleTimerDisabled = false
                if let dest = dest {
                    self.refreshModels()
                    self.selectedModel = self.models.first { $0.url == dest }
                    self.downloadURLString = ""
                    self.log("Downloaded \(dest.lastPathComponent)")
                } else {
                    self.log("Download failed: \(self.downloader.error ?? "cancelled")")
                }
            }
        }
    }

    func cancelDownload() {
        downloader.cancel()
        UIApplication.shared.isIdleTimerDisabled = false
        log("Download cancelled")
    }

    // MARK: - Start / Stop

    func start() {
        guard !isRunning, !isBusy else { return }
        guard let model = selectedModel else {
            status = .error("No model selected")
            return
        }
        guard let portValue = UInt16(port) else {
            status = .error("Invalid port")
            return
        }
        let ctxSize = Int(contextSize) ?? 4096
        let modelPath = model.url.path

        status = .loadingModel
        ipAddress = NetworkInfo.wifiIPv4Address()
        log("Starting… loading \(model.name) into memory")

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                let engine = try LlamaInference(modelPath: modelPath, contextSize: ctxSize)
                let server = LlamaHTTPServer(inference: engine)
                try server.start(port: portValue)

                let effectiveCtx = engine.contextSize
                await MainActor.run {
                    self.inference = engine
                    self.httpServer = server
                    self.status = .running
                    UIApplication.shared.isIdleTimerDisabled = true
                    if effectiveCtx < ctxSize {
                        self.log("Context reduced to \(effectiveCtx) tokens to fit device memory.")
                    }
                    self.log("Model loaded (context \(effectiveCtx)). HTTP server listening on port \(portValue).")
                    if let url = self.serverURL { self.log("Reachable at \(url)") }
                }
            } catch {
                await MainActor.run {
                    self.status = .error(error.localizedDescription)
                    self.log("Failed to start: \(error.localizedDescription)")
                    self.inference = nil
                    self.httpServer = nil
                }
            }
        }
    }

    func stop() {
        httpServer?.stop()
        httpServer = nil
        inference = nil       // deinit frees the llama context/model
        status = .stopped
        UIApplication.shared.isIdleTimerDisabled = false
        log("Server stopped and model unloaded.")
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    // MARK: - Logging

    /// UI-originated events log at `.info`. The `onLine` mirror appends the
    /// formatted line to the panel, so this does NOT append directly (avoids
    /// duplicates and keeps the panel identical to the on-disk log).
    func log(_ message: String) {
        FileLogger.shared.info(message)
    }

    /// Called on the main thread by `FileLogger.onLine` for every emitted line.
    private func appendLine(_ line: String) {
        logs.append(line)
        if logs.count > 500 { logs.removeFirst(logs.count - 500) }
    }
}
