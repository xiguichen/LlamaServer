import Foundation
import Combine

/// Drives the UI: model selection, starting/stopping the inference engine and
/// the HTTPS server, and surfacing status + logs.
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

    @Published private(set) var status: Status = .stopped
    @Published var selectedModelURL: URL?
    @Published var port: String = "8443"
    @Published var contextSize: String = "4096"
    @Published private(set) var logs: [String] = []
    @Published private(set) var ipAddress: String? = NetworkInfo.wifiIPv4Address()

    private var inference: LlamaInference?
    private var httpServer: LlamaHTTPServer?

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
        return "https://\(ip):\(port)"
    }

    // MARK: - Model selection

    func selectModel(url: URL) {
        selectedModelURL = url
        log("Selected model: \(url.lastPathComponent)")
    }

    // MARK: - Start / Stop

    func start() {
        guard !isRunning, !isBusy else { return }
        guard let modelURL = selectedModelURL else {
            status = .error("No model selected")
            return
        }
        guard let portValue = UInt16(port) else {
            status = .error("Invalid port")
            return
        }
        let ctxSize = Int(contextSize) ?? 4096

        status = .loadingModel
        ipAddress = NetworkInfo.wifiIPv4Address()
        log("Starting… loading model into memory")

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            // Access the security-scoped resource for files picked from Files app.
            let needsScopedAccess = modelURL.startAccessingSecurityScopedResource()
            defer { if needsScopedAccess { modelURL.stopAccessingSecurityScopedResource() } }

            do {
                let engine = try LlamaInference(modelPath: modelURL.path,
                                                contextSize: ctxSize)
                let server = LlamaHTTPServer(inference: engine)
                try server.start(port: portValue)

                await MainActor.run {
                    self.inference = engine
                    self.httpServer = server
                    self.status = .running
                    self.log("Model loaded. HTTPS server listening on port \(portValue).")
                    if let url = self.serverURL {
                        self.log("Reachable at \(url)")
                    }
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
        log("Server stopped and model unloaded.")
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    // MARK: - Logging

    func log(_ message: String) {
        let timestamp = Self.formatter.string(from: Date())
        logs.append("[\(timestamp)] \(message)")
        if logs.count > 200 { logs.removeFirst(logs.count - 200) }
    }

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}
