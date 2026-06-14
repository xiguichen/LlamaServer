import Foundation
import Telegraph

// MARK: - StreamableServer

/// Subclass of Telegraph's `Server` that intercepts streaming SSE requests
/// before they reach the synchronous route handler, giving us direct access
/// to the underlying `HTTPConnection` for writing chunked data.
final class StreamableServer: Server {
    /// Return `true` to claim the request (no further processing).
    var interceptHandler: ((HTTPRequest, HTTPConnection) -> Bool)?

    override func handleIncoming(request: HTTPRequest, connection: HTTPConnection, error: Error?) {
        if let handler = interceptHandler, handler(request, connection) {
            return
        }
        super.handleIncoming(request: request, connection: connection, error: error)
    }
}

/// HTTP server exposing an OpenAI-compatible API backed by `LlamaInference`.
///
/// Endpoints:
///   GET  /health                 -> { "status": "ok" }
///   GET  /v1/models              -> OpenAI model list
///   POST /v1/chat/completions    -> OpenAI chat completion (streaming + non-streaming)
///
/// Plain HTTP (no TLS) for friction-free use on a trusted local network — LAN
/// clients just use `http://<device-ip>:<port>` with no certificate to trust.
/// Do not expose this directly to untrusted networks.
final class LlamaHTTPServer {

    enum ServerError: Error, LocalizedError {
        case startFailed(String)

        var errorDescription: String? {
            switch self {
            case .startFailed(let message): return message
            }
        }
    }

    private let inference: LlamaInference
    private var server: StreamableServer?
    /// Serializes access to the (non-thread-safe) llama context.
    private let inferenceQueue = DispatchQueue(label: "llama.inference.serial")
    private var requestCount = 0
    private var heartbeatTimer: Timer?
    private var listenPort: UInt16 = 0

    init(inference: LlamaInference) {
        self.inference = inference
    }

    var isRunning: Bool { server?.isRunning ?? false }
    var port: Int { Int(server?.port ?? 0) }

    // MARK: - Start / Stop

    func start(port: UInt16) throws {
        listenPort = port
        let server = makeServer()
        try server.start(port: Int(port))
        self.server = server
        DispatchQueue.main.async {
            self.heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
                guard let self else { return }
                if let srv = self.server, !srv.isRunning {
                    FileLogger.shared.log("[heartbeat] listener down — restarting")
                    self.restartListener()
                }
                FileLogger.shared.log("[heartbeat] alive")
            }
        }
    }

    func stop() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        server?.stop()
        server = nil
    }

    /// Re-bind the TCP listener without unloading the model (keeps the inference
    /// context live). Called when iOS closes the listener socket during suspension.
    func restartListener() {
        guard listenPort > 0 else { return }
        let oldServer = server
        let newServer = makeServer()
        do {
            try newServer.start(port: Int(listenPort))
            oldServer?.stop()
            self.server = newServer
            FileLogger.shared.log("listener restarted on port \(listenPort)")
        } catch {
            FileLogger.shared.log("listener restart failed: \(error.localizedDescription)")
            self.server = oldServer
        }
    }

    private func makeServer() -> StreamableServer {
        let server = StreamableServer()
        server.interceptHandler = { [weak self] request, connection in
            guard let self = self,
                  request.method == .POST else { return false }
            let path = request.uri.path.lowercased()
            guard path == "/v1/chat/completions" || path.hasSuffix("/v1/chat/completions") else { return false }
            guard request.body.count <= Self.maxRequestBytes,
                  let body = try? JSONDecoder().decode(ChatCompletionRequest.self, from: request.body) else { return false }
            FileLogger.shared.log("chat/completions body.stream=\(body.stream ?? false)")
            guard body.stream == true else { return false }
            self.handleStreamingCompletion(request: request, connection: connection, body: body)
            return true
        }
        configureRoutes(on: server)
        return server
    }

    // MARK: - Routes

    private func configureRoutes(on server: Server) {
        server.route(.GET, "health") { _ in
            var resp = HTTPResponse(.ok, headers: ["Content-Type": "application/json"],
                                    body: Data(#"{"status":"ok"}"#.utf8))
            resp.headers["Connection"] = "close"
            return resp
        }

        server.route(.GET, "v1/models") { [weak self] _ in
            self?.modelsResponse() ?? HTTPResponse(.internalServerError)
        }

        server.route(.POST, "v1/chat/completions") { [weak self] request in
            self?.chatCompletionResponse(request: request) ?? HTTPResponse(.internalServerError)
        }
    }

    private func modelsResponse() -> HTTPResponse {
        let info = ModelInfo(id: inference.modelName, object: "model",
                             created: Int(Date().timeIntervalSince1970),
                             owned_by: "local")
        let list = ModelList(object: "list", data: [info])
        return json(encodable: list, status: .ok)
    }

    /// Reject request bodies larger than this to avoid a memory-spike crash.
    private static let maxRequestBytes = 8 * 1024 * 1024
    /// Upper bound on client-requested completion length (a foot-gun otherwise).
    private static let maxCompletionTokens = 4096

    private var previousFootprint: UInt64 = 0

    private var memoryMB: UInt64 {
        // MACH_TASK_BASIC_INFO = 20
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let r = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, 20, $0, &count)
            }
        }
        return r == KERN_SUCCESS ? info.resident_size / (1024 * 1024) : 0
    }

    private func handleStreamingCompletion(request: HTTPRequest, connection: HTTPConnection, body: ChatCompletionRequest) {
        let currentRequest = requestCount + 1
        requestCount = currentRequest
        let mem = memoryMB
        let delta = previousFootprint == 0 ? 0 : Int64(mem) - Int64(previousFootprint)
        previousFootprint = mem
        FileLogger.shared.log("streaming chat request #\(currentRequest): mem=\(mem)MB\(delta >= 0 ? "+" : "")\(delta)")

        guard !body.messages.isEmpty else {
            connection.send(data: "data: {\"error\":\"messages must not be empty\"}\n\n".data(using: .utf8)!, timeout: 10)
            connection.close(immediately: false)
            return
        }

        let temperature = Float(body.temperature ?? 0.8)
        let topP = Float(body.top_p ?? 0.95)
        let maxTokens = min(max(1, body.max_tokens ?? 512), Self.maxCompletionTokens)
        let chatId = "chatcmpl-\(UUID().uuidString)"
        let modelName = body.model ?? inference.modelName
        let created = Int(Date().timeIntervalSince1970)

        // Send raw HTTP response headers (Connection: keep-alive, no Content-Length)
        var header = "HTTP/1.1 200 OK\r\n"
        header += "Content-Type: text/event-stream\r\n"
        header += "Cache-Control: no-cache\r\n"
        header += "Connection: keep-alive\r\n"
        header += "\r\n"
        connection.send(data: header.data(using: .utf8)!, timeout: 10)
        FileLogger.shared.log("streaming #\(currentRequest): SSE headers sent, token count=\(body.messages.count)")

        // First chunk announces role
        if let roleData = try? JSONEncoder().encode(
            StreamingChunk(id: chatId, created: created, model: modelName,
                           choices: [StreamingChoice(index: 0,
                                                     delta: Delta(role: "assistant", content: nil),
                                                     finish_reason: nil)])
        ), let roleStr = String(data: roleData, encoding: .utf8) {
            connection.send(data: "data: \(roleStr)\n\n".data(using: .utf8)!, timeout: 10)
        }

        // Run generation on the serial queue. Each token is flushed as an SSE event.
        inferenceQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let prompt = self.inference.formatPrompt(messages: body.messages)
                _ = try self.inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                temperature: temperature,
                                                topP: topP) { text in
                    if let chunkData = try? JSONEncoder().encode(
                        StreamingChunk(id: chatId, created: created, model: modelName,
                                       choices: [StreamingChoice(index: 0,
                                                                 delta: Delta(role: nil, content: text),
                                                                 finish_reason: nil)])
                    ), let chunkStr = String(data: chunkData, encoding: .utf8) {
                        connection.send(data: "data: \(chunkStr)\n\n".data(using: .utf8)!, timeout: 10)
                    }
                    return true
                }

                // Final chunk with usage
                if let usageData = try? JSONEncoder().encode(
                    StreamingChunk(id: chatId, created: created, model: modelName,
                                   choices: [StreamingChoice(index: 0,
                                                             delta: Delta(role: nil, content: nil),
                                                             finish_reason: "stop")])
                ), let usageStr = String(data: usageData, encoding: .utf8) {
                    connection.send(data: "data: \(usageStr)\n\n".data(using: .utf8)!, timeout: 10)
                }
            } catch {
                FileLogger.shared.log("streaming generation error: \(error.localizedDescription)")
                if let errData = try? JSONEncoder().encode(
                    StreamingChunk(id: chatId, created: created, model: modelName,
                                   choices: [StreamingChoice(index: 0,
                                                             delta: Delta(role: nil, content: nil),
                                                             finish_reason: "error")])
                ), let errStr = String(data: errData, encoding: .utf8) {
                    connection.send(data: "data: \(errStr)\n\n".data(using: .utf8)!, timeout: 10)
                }
            }
            connection.send(data: "data: [DONE]\n\n".data(using: .utf8)!, timeout: 10)
            FileLogger.shared.log("streaming #\(currentRequest): [DONE] sent, closing connection")
            connection.close(immediately: false)
        }
    }

    private func chatCompletionResponse(request: HTTPRequest) -> HTTPResponse {
        guard request.body.count <= Self.maxRequestBytes else {
            return errorResponse("Request body too large", status: .badRequest)
        }
        let decoder = JSONDecoder()
        guard let body = try? decoder.decode(ChatCompletionRequest.self, from: request.body) else {
            return errorResponse("Invalid request body", status: .badRequest)
        }
        guard !body.messages.isEmpty else {
            return errorResponse("`messages` must not be empty", status: .badRequest)
        }

        requestCount += 1
        let mem = memoryMB
        let delta = previousFootprint == 0 ? 0 : Int64(mem) - Int64(previousFootprint)
        previousFootprint = mem
        FileLogger.shared.log("chat request #\(requestCount): mem=\(mem)MB\(delta >= 0 ? "+" : "")\(delta)")

        let temperature = Float(body.temperature ?? 0.8)
        let topP = Float(body.top_p ?? 0.95)
        let maxTokens = min(max(1, body.max_tokens ?? 512), Self.maxCompletionTokens)

        // All llama API access (formatPrompt + generate) must be serialized —
        // the model/context is NOT thread-safe. Run both inside the serial queue.
        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
                let prompt = inference.formatPrompt(messages: body.messages)
                result = try inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                temperature: temperature,
                                                topP: topP)
            } catch {
                failure = error
            }
        }

        if let failure = failure {
            return errorResponse(failure.localizedDescription, status: .internalServerError)
        }
        guard let result = result else {
            return errorResponse("Generation produced no result", status: .internalServerError)
        }

        // Breadcrumb after generation, before response assembly: lets us tell
        // whether a process death (jetsam SIGKILL leaves no crash report) happens
        // during generation vs. during response encoding/return.
        FileLogger.shared.log("generation OK: #\(requestCount) \(result.text.count) chars, encoding response")

        let response = ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString)",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: body.model ?? inference.modelName,
            choices: [
                ChatCompletionChoice(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: result.text),
                    finish_reason: "stop"
                )
            ],
            usage: Usage(
                prompt_tokens: result.promptTokens,
                completion_tokens: result.completionTokens,
                total_tokens: result.promptTokens + result.completionTokens
            )
        )
        return json(encodable: response, status: .ok)
    }

    // MARK: - Helpers

    private func json<T: Encodable>(encodable: T, status: HTTPStatus) -> HTTPResponse {
        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(encodable) else {
            return HTTPResponse(.internalServerError)
        }
        var resp = HTTPResponse(status, headers: ["Content-Type": "application/json"], body: data)
        resp.headers["Connection"] = "close"
        return resp
    }

    private func errorResponse(_ message: String, status: HTTPStatus) -> HTTPResponse {
        let error = APIError(error: .init(message: message, type: "invalid_request_error"))
        return json(encodable: error, status: status)
    }
}
