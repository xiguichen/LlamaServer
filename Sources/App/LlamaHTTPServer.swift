import Foundation
import Telegraph
import os
import UIKit

/// Proxies `HTTPConnectionDelegate` to detect client disconnect mid-stream.
/// Captures the `didCloseWithError` callback and exposes a thread-safe flag
/// so the generation loop can cancel early instead of writing into a dead
/// connection and wasting inference resources.
final class DisconnectWatcher: HTTPConnectionDelegate {
    weak var server: HTTPConnectionDelegate?
    private let _didDisconnect = OSAllocatedUnfairLock(initialState: false)

    var didDisconnect: Bool { _didDisconnect.withLock { $0 } }

    func connection(_ httpConnection: HTTPConnection, didCloseWithError error: Error?) {
        if let err = error {
            let nsErr = err as NSError
            FileLogger.shared.debug("DisconnectWatcher: didCloseWithError domain=\(nsErr.domain) code=\(nsErr.code) \"\(nsErr.localizedDescription)\"")
        } else {
            FileLogger.shared.debug("DisconnectWatcher: didCloseWithError - no error (clean close)")
        }
        _didDisconnect.withLock { $0 = true }
        server?.connection(httpConnection, didCloseWithError: error)
    }

    func connection(_ httpConnection: HTTPConnection, handleIncomingRequest request: HTTPRequest, error: Error?) {
        server?.connection(httpConnection, handleIncomingRequest: request, error: error)
    }

    func connection(_ httpConnection: HTTPConnection, handleIncomingResponse response: HTTPResponse, error: Error?) {
        server?.connection(httpConnection, handleIncomingResponse: response, error: error)
    }

    func connection(_ httpConnection: HTTPConnection, handleUpgradeByRequest request: HTTPRequest) {
        server?.connection(httpConnection, handleUpgradeByRequest: request)
    }
}

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
    /// Keeps the app alive in the background (via a silent audio session) so the
    /// HTTP server keeps serving requests instead of being suspended by iOS.
    private let keepAlive = BackgroundKeepAlive()
    /// Bonjour service for _http._tcp. Publishing it triggers iOS's local
    /// network permission dialog on first launch. Without this, the dialog may
    /// not appear until the user changes the context size (which reloads the
    /// model and restarts the listener), leaving the server unreachable.
    private var bonjourService: NetService?
    /// Serializes access to the (non-thread-safe) llama context.
    private let inferenceQueue = DispatchQueue(label: "llama.inference.serial")
    /// Counting + cadence for the periodic llama_context recreate (flushes
    /// accumulated GPU/allocator state). The recreate itself runs OFF the
    /// response path — see `runPendingRecreate`.
    private let contextRecreateThreshold = 10
    private lazy var recreateTracker = RecreateTracker(threshold: contextRecreateThreshold)
    private let requestCountLock = OSAllocatedUnfairLock()
    private var _requestCount = 0
    private var requestCount: Int {
        get { requestCountLock.withLock { _requestCount } }
        set { requestCountLock.withLock { _requestCount = newValue } }
    }
    private func nextRequestNumber() -> Int {
        requestCountLock.withLock {
            _requestCount += 1
            return _requestCount
        }
    }

    private var heartbeatTimer: Timer?
    private var listenPort: UInt16 = 0
    /// Tokens for the NotificationCenter observers so we can remove them in
    /// `deinit`. Without this the closures (and the captured `self`) leak, which
    /// keeps every server — and the `LlamaInference` engine it owns — alive for
    /// the lifetime of the process.
    private var observerTokens: [NSObjectProtocol] = []
    private let connectionCountLock = OSAllocatedUnfairLock()
    private var _activeConnections = 0
    private var activeConnections: Int {
        get { connectionCountLock.withLock { _activeConnections } }
        set { connectionCountLock.withLock { _activeConnections = newValue } }
    }
    /// Set true by `stop()` so in-flight streaming callbacks can finish at the
    /// next token boundary instead of blocking the caller (main thread) for the
    /// entire generation. Checked inside the token callback alongside
    /// `disconnectWatcher.didDisconnect`.
    private let _stopping = OSAllocatedUnfairLock(initialState: false)

    init(inference: LlamaInference) {
        self.inference = inference
        // Log iOS memory pressure events so we can correlate them with crashes.
        observerTokens.append(NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main) { [weak self] _ in
                guard let self else { return }
                FileLogger.shared.warn("MEMORY WARNING: \(self.systemMemoryLog), active connections=\(self.activeConnections)")
            })
        // iOS closes the listening socket while the app is suspended, so a server
        // that was "running" before backgrounding can come back with a dead
        // listener (the heartbeat Timer is also paused while suspended). Rebind
        // on foreground so the server is reachable again without a manual restart.
        observerTokens.append(NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil, queue: .main) { [weak self] _ in
                guard let self, self.listenPort > 0, self.server?.isRunning != true else { return }
                FileLogger.shared.info("foreground: listener down — rebinding")
                self.restartListener()
            })
    }

    deinit {
        observerTokens.forEach { NotificationCenter.default.removeObserver($0) }
        observerTokens.removeAll()
    }

    var isRunning: Bool { server?.isRunning ?? false }
    var port: Int { Int(server?.port ?? 0) }

    // MARK: - Start / Stop

    func start(port: UInt16) throws {
        listenPort = port
        let server = makeServer()
        try server.start(port: Int(port))
        self.server = server

        // Publish Bonjour service to trigger iOS local network permission
        // dialog. The service name includes the port so clients can discover
        // multiple instances on the same network.
        let service = NetService(domain: "local.", type: "_http._tcp",
                                 name: "LlamaServer (\(port))", port: Int32(port))
        service.publish()
        self.bonjourService = service

        // Hold the app awake in the background so requests aren't suspended.
        keepAlive.start()
        DispatchQueue.main.async {
            self.heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
                guard let self else { return }
                // `isRunning != true` also covers the case where a prior restart
                // attempt failed and left `server` nil, so the heartbeat keeps
                // retrying until the listener is back up.
                if self.server?.isRunning != true {
                    FileLogger.shared.warn("[heartbeat] listener down — restarting")
                    self.restartListener()
                }
                FileLogger.shared.debug("[heartbeat] \(self.systemMemoryLog)")
            }
        }
    }

    func stop() {
        bonjourService?.stop()
        bonjourService = nil
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        keepAlive.stop()
        FileLogger.shared.debug("stop: active connections=\(activeConnections)")
        // Signal in-flight generations to finish at the next token boundary
        // so the drain below completes quickly instead of blocking the caller
        // (main thread / UI) for potentially minutes.
        _stopping.withLock { $0 = true }
        // Drain the inference queue so in-flight streaming requests can send
        // finish_reason + [DONE] before connections are closed.
        inferenceQueue.sync { }
        server?.stop(immediately: true)
        server = nil
    }

    /// Guards against re-entrant calls to `restartListener()` (e.g. heartbeat
    /// fires while a background restart is already in progress).
    private let restartLock = OSAllocatedUnfairLock(initialState: false)

    /// Re-bind the TCP listener without unloading the model (keeps the inference
    /// context live). Called when iOS closes the listener socket during suspension.
    /// Runs the actual stop/start off the main thread so we can drain the
    /// inference queue without blocking the UI / heartbeat.
    func restartListener() {
        guard listenPort > 0 else { return }
        guard !restartLock.withLock({
            if $0 { return true }
            $0 = true
            return false
        }) else {
            FileLogger.shared.debug("restartListener: already in progress, skipping")
            return
        }
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            let conns = self.activeConnections
            if conns > 0 {
                FileLogger.shared.debug("restartListener: draining \(conns) active connection(s)")
            }
            // Allow in-flight streaming requests to finish before tearing down
            // the listener. Without this, force-closing connections mid-stream
            // causes "Stream ended without finish_reason" errors on the client.
            self.inferenceQueue.sync { }
            DispatchQueue.main.async {
                self.server?.stop(immediately: true)
                self.server = nil
                let newServer = self.makeServer()
                do {
                    try newServer.start(port: Int(self.listenPort))
                    self.server = newServer
                    FileLogger.shared.info("listener restarted on port \(self.listenPort)")
                } catch {
                    FileLogger.shared.error("listener restart failed: \(error.localizedDescription)")
                }
                self.restartLock.withLock { $0 = false }
            }
        }
    }

    private func makeServer() -> StreamableServer {
        let server = StreamableServer()
        // Telegraph starts a socket read after dispatching every request to wait
        // for the next HTTP request on the keep-alive connection. Since our
        // streaming response bypasses Telegraph's normal response flow (writing
        // directly to the socket via interceptHandler), this read would time out
        // during long generations. Disable it entirely (-1 = no timeout).
        server.httpConfig.readTimeout = -1
        server.interceptHandler = { [weak self] request, connection in
            guard let self = self else { return false }
            // CORS preflight: answer any OPTIONS request directly so browser-
            // based OpenAI clients can call the API (mirrors llama-server).
            if request.method == .OPTIONS {
                var header = "HTTP/1.1 204 No Content\r\n"
                header += Self.corsHeaderLines
                header += "Access-Control-Max-Age: 86400\r\n"
                header += "Content-Length: 0\r\n"
                header += "\r\n"
                connection.send(data: header.data(using: .utf8)!, timeout: 10)
                connection.close(immediately: false)
                return true
            }
            guard request.method == .POST else { return false }
            let path = request.uri.path.lowercased()
            guard path == "/v1/chat/completions" || path.hasSuffix("/chat/completions") else { return false }
            guard request.body.count <= Self.maxRequestBytes else {
                FileLogger.shared.warn("intercept: request body too large (\(request.body.count) bytes, max \(Self.maxRequestBytes))")
                return false
            }
            let bodyPreview = String(data: request.body.prefix(500), encoding: .utf8) ?? "<non-utf8>"
            guard let body = try? JSONDecoder().decode(ChatCompletionRequest.self, from: request.body) else {
                FileLogger.shared.error("intercept: JSON decode FAILED for POST \(path). body preview=\(bodyPreview)")
                FileLogger.shared.verbose("intercept: FAILED RAW BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
                return false
            }
            let toolsDesc = body.tools.map { " tools=\($0.count)" } ?? ""
            FileLogger.shared.info("chat/completions body.stream=\(body.stream ?? false)\(toolsDesc)")
            guard body.stream == true else { return false }
            FileLogger.shared.debug("intercept: claiming streaming request")
            self.handleStreamingCompletion(request: request, connection: connection, body: body)
            return true
        }
        configureRoutes(on: server)
        return server
    }

    // MARK: - Routes

    private func configureRoutes(on server: Server) {
        let health: (HTTPRequest) -> HTTPResponse = { _ in
            let body = Data(#"{"status":"ok"}"#.utf8)
            return HTTPResponse(.ok, headers: [
                "Content-Type": "application/json",
                "Content-Length": "\(body.count)",
                "Access-Control-Allow-Origin": "*"
            ], body: body)
        }
        // Register both the bare and /v1-prefixed paths, mirroring llama-server.
        server.route(.GET, "health", health)
        server.route(.GET, "v1/health", health)

        let models: (HTTPRequest) -> HTTPResponse = { [weak self] _ in
            self?.modelsResponse() ?? HTTPResponse(.internalServerError)
        }
        server.route(.GET, "models", models)
        server.route(.GET, "v1/models", models)

        let chat: (HTTPRequest) -> HTTPResponse = { [weak self] request in
            self?.chatCompletionResponse(request: request) ?? HTTPResponse(.internalServerError)
        }
        server.route(.POST, "chat/completions", chat)
        server.route(.POST, "v1/chat/completions", chat)

        let completions: (HTTPRequest) -> HTTPResponse = { [weak self] request in
            self?.completionResponse(request: request) ?? HTTPResponse(.internalServerError)
        }
        server.route(.POST, "completions", completions)
        server.route(.POST, "v1/completions", completions)
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
    /// Also the default when a client omits `max_tokens`: OpenAI semantics treat
    /// an omitted limit as "generate until EOG or context full", and a large
    /// tool call (e.g. a `write` with sizable file content) needs the headroom —
    /// a low default truncates the JSON mid-stream and the tool call fails to
    /// parse. `generate()` still stops cleanly on EOG / context-full.
    private static let maxCompletionTokens = 8192

    /// CORS headers sent on the OPTIONS preflight. The API uses no cookies or
    /// Authorization, so a wildcard origin is appropriate for an open LAN server.
    private static let corsHeaderLines =
        "Access-Control-Allow-Origin: *\r\n" +
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" +
        "Access-Control-Allow-Headers: *\r\n"

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

    /// System memory snapshot: total, this process's RSS, and the raw
    /// `os_proc_available_memory()` value written verbatim (bytes) so every
    /// line doubles as an exact A/B record across toolchain/entitlement
    /// changes, plus Mach `vm_statistics64` system-wide page counts when
    /// available (entitlement-gated on iOS).
    private var systemMemoryLog: String {
        let total = ProcessInfo.processInfo.physicalMemory / (1024 * 1024)
        let pageSize = UInt64(vm_page_size)
        let rss = memoryMB

        // os_proc_available_memory(): bytes this process may allocate before
        // jetsam kills it (iOS 17+, no entitlement needed). Captured raw and
        // included in BOTH branches — previously the Mach-success branch
        // omitted it entirely and the fallback truncated to whole MB.
        let availableRaw = os_proc_available_memory()
        let availableMB = availableRaw / (1024 * 1024)
        let availableField = "available=\(availableRaw)B(\(availableMB)MB)"

        // Try Mach host_statistics (requires entitlement on iOS).
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let active    = UInt64(stats.active_count) * pageSize / (1024 * 1024)
            let wired     = UInt64(stats.wire_count) * pageSize / (1024 * 1024)
            let compressed = UInt64(stats.compressor_page_count) * pageSize / (1024 * 1024)
            let free      = UInt64(stats.free_count) * pageSize / (1024 * 1024)
            return "total=\(total)MB active=\(active)MB wired=\(wired)MB compressed=\(compressed)MB free=\(free)MB rss=\(rss)MB \(availableField)"
        }

        // Mach stats unavailable (host_statistics is entitlement-gated on
        // iOS): report the process-level numbers we always have.
        return "total=\(total)MB \(availableField) rss=\(rss)MB"
    }

    private func handleStreamingCompletion(request: HTTPRequest, connection: HTTPConnection, body: ChatCompletionRequest) {
        let currentRequest = nextRequestNumber()
        activeConnections += 1
        FileLogger.shared.debug("streaming #\(currentRequest): connection OPENED (total active: \(activeConnections))")
        if FileLogger.shared.logPromptContent {
            FileLogger.shared.verbose("streaming #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
        }
        FileLogger.shared.debug("streaming chat request #\(currentRequest): \(systemMemoryLog)")

        guard !body.messages.isEmpty else {
            // No SSE stream has started yet, so return a proper HTTP error
            // response (status line + headers) rather than a bare `data:` line,
            // which clients would reject as a malformed HTTP response.
            let bodyJSON = Data(#"{"error":{"message":"`messages` must not be empty","type":"invalid_request_error"}}"#.utf8)
            var header = "HTTP/1.1 400 Bad Request\r\n"
            header += "Content-Type: application/json\r\n"
            header += "Access-Control-Allow-Origin: *\r\n"
            header += "Content-Length: \(bodyJSON.count)\r\n"
            header += "\r\n"
            var out = header.data(using: .utf8)!
            out.append(bodyJSON)
            connection.send(data: out, timeout: 10)
            connection.close(immediately: false)
            return
        }

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? Self.maxCompletionTokens), Self.maxCompletionTokens)
        let chatId = "chatcmpl-\(UUID().uuidString)"
        let modelName = body.model ?? inference.modelName
        let created = Int(Date().timeIntervalSince1970)
        let samplingParams = Self.samplingParams(from: body)
        let includeUsage = body.stream_options?.include_usage ?? false
        // tool_choice "none" suppresses tools entirely.
        let toolsActive = (body.tools?.isEmpty == false) && !(body.tool_choice?.disablesTools ?? false)

        // Swap in a delegate proxy so we detect when the client disconnects
        // mid-stream and can cancel generation early.
        let disconnectWatcher = DisconnectWatcher()
        disconnectWatcher.server = connection.delegate as? HTTPConnectionDelegate
        connection.delegate = disconnectWatcher

        // Send SSE headers + role chunk immediately so the client gets an HTTP
        // response even when the serial inference queue is busy with a prior
        // request. The queue only guards the llama context, not socket writes.
        var header = "HTTP/1.1 200 OK\r\n"
        header += "Content-Type: text/event-stream\r\n"
        header += "Cache-Control: no-cache\r\n"
        header += "Connection: keep-alive\r\n"
        header += "Access-Control-Allow-Origin: *\r\n"
        header += "\r\n"
        connection.send(data: header.data(using: .utf8)!, timeout: 10)
        FileLogger.shared.debug("streaming #\(currentRequest): SSE headers sent, token count=\(body.messages.count)")

        // First chunk announces role
        if let roleData = try? JSONEncoder().encode(
            StreamingChunk(id: chatId, created: created, model: modelName,
                           choices: [StreamingChoice(index: 0,
                                                      delta: Delta(role: "assistant", content: nil),
                                                      finish_reason: nil)])
        ), let roleStr = String(data: roleData, encoding: .utf8) {
            connection.send(data: "data: \(roleStr)\n\n".data(using: .utf8)!, timeout: 10)
            FileLogger.shared.debug("streaming #\(currentRequest): role chunk sent")
        }

        // Send SSE keep-alive data events every 5 s so clients (including the
        // OpenAI SDK) don't time out during any silent phase: the serial-queue
        // wait, the prompt decode, fully buffered tool generation, AND long
        // text generations that exceed the client's idle timeout.
        // NOTE: SSE comments (`: keepalive`) are ignored by the OpenAI SDK's
        // SSE parser, so we send real data: lines that trigger onBody in
        // undici and yield chunks through the SSE decoder, resetting *both*
        // the HTTP-level body timeout AND the SDK-level stream timeout.
        let keepAliveActive = OSAllocatedUnfairLock(initialState: true)
        let keepAliveSource = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
        keepAliveSource.schedule(deadline: .now() + 2, repeating: .seconds(2), leeway: .seconds(1))
        keepAliveSource.setEventHandler { [weak connection] in
            guard let conn = connection else {
                FileLogger.shared.warn("streaming #\(currentRequest): keepalive SKIPPED — connection deallocated")
                return
            }
            guard keepAliveActive.withLock({ $0 }) else { return }
            if let data = try? JSONEncoder().encode(
                StreamingChunk(id: chatId, created: created, model: modelName,
                               choices: [StreamingChoice(index: 0,
                                                          delta: Delta(role: nil, content: ""),
                                                          finish_reason: nil)])
            ), let str = String(data: data, encoding: .utf8) {
                conn.send(data: "data: \(str)\n\n".data(using: .utf8)!, timeout: 30)
            }
        }
        keepAliveSource.resume()
        FileLogger.shared.debug("streaming #\(currentRequest): keep-alive timer started")

        // Run generation on the serial queue. Each token is flushed as an SSE event.
        inferenceQueue.async { [weak self, disconnectWatcher] in
            guard let self = self else { keepAliveSource.cancel(); return }
            FileLogger.shared.debug("streaming #\(currentRequest): async block started (queue wait ended)")

            // Idempotent: safe to call repeatedly / from any token.
            func stopKeepAlive() {
                keepAliveActive.withLock { $0 = false }
                keepAliveSource.cancel()
            }

            var didStreamContentDuringToolGen = false
            var lastChunkTime = CFAbsoluteTimeGetCurrent()

            func sendChunk(delta: Delta, finishReason: String?) {
                guard !disconnectWatcher.didDisconnect else {
                    if finishReason != nil {
                        FileLogger.shared.warn("streaming #\(currentRequest): finish_reason dropped — client disconnected")
                    }
                    return
                }
                guard let data = try? JSONEncoder().encode(
                    StreamingChunk(id: chatId, created: created, model: modelName,
                                   choices: [StreamingChoice(index: 0, delta: delta,
                                                              finish_reason: finishReason)])
                ), let str = String(data: data, encoding: .utf8),
                      let chunkData = "data: \(str)\n\n".data(using: .utf8) else {
                    FileLogger.shared.error("streaming #\(currentRequest): sendChunk ENCODE FAILED")
                    return
                }
                connection.send(data: chunkData, timeout: 10)
                lastChunkTime = CFAbsoluteTimeGetCurrent()
            }

            func flushKeepAliveIfNeeded() {
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastChunkTime >= 4 {
                    FileLogger.shared.verbose("streaming #\(currentRequest): keepalive from token callback")
                    sendChunk(delta: Delta(role: nil, content: ""), finishReason: nil)
                }
            }

            // Bail early if the client disconnected while we were waiting
            // in the serial queue behind a prior generation.
            guard !disconnectWatcher.didDisconnect else {
                stopKeepAlive()
                FileLogger.shared.log("streaming #\(currentRequest): client disconnected during queue wait")
                connection.close(immediately: false)
                self.activeConnections -= 1
                return
            }

            var generatedTokens = 0
            var generationSucceeded = false
            var lastTokenTime = CFAbsoluteTimeGetCurrent()
            let generationStartTime = CFAbsoluteTimeGetCurrent()
            // Strips <think> reasoning from the streamed content so clients
            // receive only the final answer, not the chain-of-thought. Runs for
            // tool generations too, so a reasoning block emitted on a tool turn
            // is never streamed as visible content.
            let thinkFilter = ReasoningStreamFilter()
            // Buffers <tool_call>…</tool_call> envelopes out of the streamed
            // tool-generation content while prose around them flows through.
            let toolStreamFilter = ToolStreamFilter()
            var finishReasonSent = false

            do {
                let wireFormat = ToolWireFormat.detect(template: self.inference.chatTemplateString)
                let msgs = ToolPrompt.inject(messages: body.messages, tools: body.tools,
                                             toolChoice: body.tool_choice, format: wireFormat)
                let basePrompt = self.inference.formatPrompt(messages: msgs, format: wireFormat)
                let prompt = self.applyThinkingSwitch(prompt: basePrompt, enabled: body.thinkingEnabled, format: wireFormat)
                FileLogger.shared.info("streaming #\(currentRequest): prompt built (\(prompt.count) chars), starting generation")
                FileLogger.shared.debug("streaming #\(currentRequest): thinking=\(body.thinkingEnabled)")
                if FileLogger.shared.logPromptContent {
                    FileLogger.shared.verbose("streaming #\(currentRequest): FULL PROMPT >>>\n\(prompt)\n<<<")
                }
                let genResult = try self.inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams) { text in
                    generatedTokens += 1
                    let now = CFAbsoluteTimeGetCurrent()
                    if now - lastTokenTime >= 10 {
                        FileLogger.shared.debug("streaming #\(currentRequest): generated \(generatedTokens) tokens so far")
                        lastTokenTime = now
                    }
                    // Stop generation early if the client disconnected or the
                    // server is being shut down (prevents blocking the caller).
                    if disconnectWatcher.didDisconnect {
                        FileLogger.shared.debug("streaming #\(currentRequest): token callback aborted — client disconnected")
                        return false
                    }
                    // Generation continues until the model naturally finishes (EOG,
                    // max_tokens, or context full). The client-side httpIdleTimeoutMs
                    // is set to 0 (disabled) in pi's settings, so no artificial
                    // cutoff is needed.
                    if self._stopping.withLock({ $0 }) {
                        FileLogger.shared.debug("streaming #\(currentRequest): token callback aborted — server stopping")
                        return false
                    }
                    // Stream text content during tool generation (up to the first
                    // <tool_call> envelope), then buffer the envelope, then
                    // resume streaming after </tool_call>. This keeps the
                    // client's SSE connection alive during long generations.
                    // The text is run through the reasoning filter first so a
                    // <think> block emitted on a tool turn is stripped rather
                    // than streamed (which would also suppress the cleaned
                    // content below via didStreamContentDuringToolGen).
                    guard !toolsActive else {
                        let visible = thinkFilter.feed(ToolCallParser.stripANSI(text))
                        guard !visible.isEmpty else {
                            flushKeepAliveIfNeeded()
                            return true
                        }
                        let emit = toolStreamFilter.feed(visible)
                        if !emit.isEmpty {
                            didStreamContentDuringToolGen = true
                            FileLogger.shared.verbose("streaming #\(currentRequest): tool-text >>>\(emit)<<<")
                            sendChunk(delta: Delta(role: nil, content: emit), finishReason: nil)
                        }
                        flushKeepAliveIfNeeded()
                        return true
                    }
                    let visible = thinkFilter.feed(ToolCallParser.stripANSI(text))
                    if visible.isEmpty {
                        flushKeepAliveIfNeeded()
                        return true
                    }
                    FileLogger.shared.verbose("streaming #\(currentRequest): chunk >>>\(visible)<<<")
                    sendChunk(delta: Delta(role: nil, content: visible), finishReason: nil)
                    return true
                }
                FileLogger.shared.info("streaming #\(currentRequest): generation done, \(generatedTokens) tokens, finish=\(genResult.finishReason)")
                if FileLogger.shared.logPromptContent {
                    FileLogger.shared.verbose("streaming #\(currentRequest): FULL RAW OUTPUT >>>\n\(genResult.text)\n<<<")
                }
                // Flush any content the reasoning filter held back. On the tool
                // path the flushed text still has to pass the tool filter, so a
                // held-back tail is streamed while an envelope remains buffered.
                let thinkTail = thinkFilter.flush()
                if !thinkTail.isEmpty {
                    if toolsActive {
                        let emit = toolStreamFilter.feed(thinkTail)
                        if !emit.isEmpty {
                            didStreamContentDuringToolGen = true
                            FileLogger.shared.verbose("streaming #\(currentRequest): tool-text >>>\(emit)<<<")
                            sendChunk(delta: Delta(role: nil, content: emit), finishReason: nil)
                        }
                    } else {
                        sendChunk(delta: Delta(role: nil, content: thinkTail), finishReason: nil)
                    }
                }
                // Periodic context flush. The recreate re-allocates the KV
                // cache (~0.3-0.5 s) — run it AFTER finish_reason + [DONE]
                // are delivered (see stream end below), never between the
                // last token and the finish chunk, which made every 10th
                // response visibly stall. Only successful generations count.
                generationSucceeded = true

                // The OpenAI finish reason is derived AFTER tool-call parsing
                // below, from the engine result plus the parse outcome
                // (FinishReasonPolicy in ResponsePolicy.swift).
                var outputUnusable = false
                var toolCallsFound = false

                if toolsActive {
                    FileLogger.shared.debug("streaming #\(currentRequest): tools active, parsing tool calls")
                    // Strip any incomplete <tool_call> (no closing </tool_call>)
                    // so pi never receives truncated JSON inside an envelope.
                    // Safety net for max_tokens, context-full, or edge-case stops.
                    let genText = genResult.text
                    let cleanText: String
                    if genText.contains("<tool_call>") && !genText.contains("</tool_call>") {
                        if let range = genText.range(of: "<tool_call>") {
                            cleanText = String(genText[..<range.lowerBound])
                            FileLogger.shared.warn("streaming #\(currentRequest): stripped incomplete tool call, text before=\(cleanText.suffix(100).replacingOccurrences(of: "\n", with: "\\n"))")
                        } else {
                            cleanText = genText
                        }
                    } else {
                        cleanText = genText
                    }
                    let parsed = ToolCallParser.parse(cleanText, tools: body.tools)
                    // The incomplete tool call stripped everything, or the
                    // parser couldn't salvage an envelope (unparseable JSON /
                    // blank name): the model failed to produce a usable
                    // response. FinishReasonPolicy reports this as an explicit
                    // error unless a real token cutoff caused it.
                    if cleanText != genText && cleanText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        outputUnusable = true
                        FileLogger.shared.warn("streaming #\(currentRequest): tool call unrecoverable after cutoff")
                    }
                    if parsed.hasUnrecoveredEnvelope {
                        outputUnusable = true
                        FileLogger.shared.warn("streaming #\(currentRequest): envelope dropped (unparseable JSON or blank name)")
                    }
                    if !parsed.toolCalls.isEmpty {
                        toolCallsFound = true
                        FileLogger.shared.info("streaming #\(currentRequest): found \(parsed.toolCalls.count) tool calls")
                        for call in parsed.toolCalls {
                            FileLogger.shared.verbose("streaming #\(currentRequest): tool_call name=\(call.function.name) args=\(call.function.arguments)")
                        }
                        // Only send cleaned content if it wasn't already streamed
                        // incrementally during generation.
                        if !parsed.cleanedContent.isEmpty && !didStreamContentDuringToolGen {
                            sendChunk(delta: Delta(role: nil, content: parsed.cleanedContent),
                                      finishReason: nil)
                            FileLogger.shared.debug("streaming #\(currentRequest): sent cleaned content chunk")
                        } else if !parsed.cleanedContent.isEmpty {
                            FileLogger.shared.debug("streaming #\(currentRequest): cleaned content already streamed, skipping")
                        }
                        let streamingCalls = parsed.toolCalls.enumerated().map {
                            StreamingToolCall(index: $0.offset, from: $0.element)
                        }
                        sendChunk(delta: Delta(role: nil, content: nil,
                                               tool_calls: streamingCalls),
                                  finishReason: nil)
                        FileLogger.shared.debug("streaming #\(currentRequest): sent tool_calls chunk")
                    } else if !parsed.cleanedContent.isEmpty {
                        // Envelope fragments are stripped by the parser and
                        // reported via hasUnrecoveredEnvelope (set above), so
                        // what remains is fragment-free prose the model wrote
                        // around the broken envelope — flush it; the finish
                        // reason carries the failure via FinishReasonPolicy.
                        if !didStreamContentDuringToolGen {
                            // No tool call after all — flush the buffered content.
                            FileLogger.shared.debug("streaming #\(currentRequest): no tool calls, flushing content")
                            sendChunk(delta: Delta(role: nil, content: parsed.cleanedContent),
                                      finishReason: nil)
                        } else {
                            FileLogger.shared.debug("streaming #\(currentRequest): content already streamed during generation")
                        }
                    } else {
                        FileLogger.shared.warn("streaming #\(currentRequest): no tool calls and no content")
                    }
                }

                // Derive the OpenAI finish reason from the engine result plus
                // the tool-call outcome — one honest policy for all cases
                // (length only for real max-token cutoffs).
                let streamReason = FinishReasonPolicy.map(engineFinish: genResult.finishReason,
                                                          hasToolCalls: toolCallsFound,
                                                          outputUnusable: outputUnusable)

                // Final chunk carries the finish reason.
                sendChunk(delta: Delta(role: nil, content: nil), finishReason: streamReason)
                finishReasonSent = true
                FileLogger.shared.debug("streaming #\(currentRequest): finish_reason chunk sent (\(streamReason))")

                // Optional usage chunk (OpenAI: empty choices array, usage set).
                if includeUsage {
                    let usage = Usage(prompt_tokens: genResult.promptTokens,
                                      completion_tokens: genResult.completionTokens,
                                      total_tokens: genResult.promptTokens + genResult.completionTokens)
                    if let data = try? JSONEncoder().encode(
                        StreamingChunk(id: chatId, created: created, model: modelName,
                                       choices: [], usage: usage)
                    ), let str = String(data: data, encoding: .utf8) {
                        connection.send(data: "data: \(str)\n\n".data(using: .utf8)!, timeout: 10)
                        FileLogger.shared.debug("streaming #\(currentRequest): usage chunk sent")
                    }
                }
            } catch {
                FileLogger.shared.error("streaming #\(currentRequest): generation error: \(error.localizedDescription)")
                // A headroom failure carries an actionable message the client
                // should see: deliver it as an OpenAI-style stream error event
                // (with an empty choices array so naive clients that iterate
                // choices don't choke) before the error finish chunk.
                if let inferenceError = error as? LlamaInference.InferenceError,
                   case .contextTooFull = inferenceError {
                    let event: [String: Any] = [
                        "error": ["message": error.localizedDescription,
                                  "type": "invalid_request_error"],
                        "choices": []
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: event),
                       let str = String(data: data, encoding: .utf8) {
                        connection.send(data: "data: \(str)\n\n".data(using: .utf8)!, timeout: 10)
                        FileLogger.shared.debug("streaming #\(currentRequest): context-too-full error event sent")
                    }
                }
                sendChunk(delta: Delta(role: nil, content: nil), finishReason: "error")
                finishReasonSent = true
            }
            stopKeepAlive()
            let didDisconnect = disconnectWatcher.didDisconnect
            FileLogger.shared.debug("streaming #\(currentRequest): stream summary — finish_reason_sent=\(finishReasonSent) didDisconnect=\(didDisconnect) genTokens=\(generatedTokens)")
            // Skip [DONE] if the client already disconnected — the finish_reason
            // was already dropped (GCDAsyncSocket queues writes asynchronously
            // and the kernel's TCP buffer doesn't flush after FIN). Sending
            // [DONE] into a dead socket is harmless but adds noise to logs.
            if !didDisconnect {
                connection.send(data: "data: [DONE]\n\n".data(using: .utf8)!, timeout: 10)
                FileLogger.shared.debug("streaming #\(currentRequest): [DONE] sent")
            }
            // Close the connection so the client's URLSession task completes.
            // `close(immediately: false)` calls `disconnectAfterWriting()`, which
            // flushes all queued writes (finish_reason, [DONE]) before sending FIN.
            FileLogger.shared.debug("streaming #\(currentRequest): closing connection")
            connection.close(immediately: false)
            self.activeConnections -= 1
            // Periodic context recreate — now that the client has everything.
            if generationSucceeded {
                self.runPendingRecreate(request: currentRequest, kind: "streaming")
            }
            FileLogger.shared.debug("streaming #\(currentRequest): stream ended (total active: \(self.activeConnections), recreate count: \(self.recreateTracker.count))")
        }
    }

    /// Counts a successful generation and runs the periodic context recreate
    /// if due. MUST be called on the inference queue, after the client's
    /// response has been fully delivered: the recreate re-allocates the KV
    /// cache (~0.3-0.5 s) and used to stall every 10th streamed response
    /// between the last token and finish_reason. A failure only logs — the
    /// engine marks itself unusable and the NEXT request surfaces the error
    /// cleanly (RecreateTracker retries the flush on the next generation).
    private func runPendingRecreate(request: Int, kind: String) {
        guard recreateTracker.record() else { return }
        do {
            try inference.recreateContext()
            recreateTracker.recreateFinished(success: true)
            FileLogger.shared.info("\(kind) request #\(request): periodic context recreate done (every \(contextRecreateThreshold) generations, off response path)")
        } catch {
            recreateTracker.recreateFinished(success: false)
            FileLogger.shared.error("\(kind) request #\(request): periodic recreateContext failed: \(error.localizedDescription)")
        }
    }

    private func chatCompletionResponse(request: HTTPRequest) -> HTTPResponse {
        guard request.body.count <= Self.maxRequestBytes else {
            FileLogger.shared.warn("chat response: body too large (\(request.body.count) bytes)")
            return errorResponse("Request body too large", status: .badRequest)
        }
        let decoder = JSONDecoder()
        guard let body = try? decoder.decode(ChatCompletionRequest.self, from: request.body) else {
            let preview = String(data: request.body.prefix(500), encoding: .utf8) ?? "<non-utf8>"
            FileLogger.shared.error("chat response: JSON decode FAILED. body preview=\(preview)")
            FileLogger.shared.verbose("chat response: FAILED RAW BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
            return errorResponse("Invalid request body", status: .badRequest)
        }
        // OpenAI requires `model`; reject a request that omits it (or leaves
        // messages empty) with a 400 instead of silently completing — see
        // ChatRequestSpec for the testable rule. Caught by llm-comply's
        // `chat-error-handling` test.
        if let rejection = ChatRequestSpec.validationError(model: body.model,
                                                           messageCount: body.messages.count) {
            return errorResponse(rejection, status: .badRequest)
        }

        let currentRequest = nextRequestNumber()
        FileLogger.shared.debug("chat request #\(currentRequest): \(systemMemoryLog), thinking=\(body.thinkingEnabled)")
        if FileLogger.shared.logPromptContent {
            FileLogger.shared.verbose("chat request #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
        }

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? Self.maxCompletionTokens), Self.maxCompletionTokens)
        let samplingParams = Self.samplingParams(from: body)

        // All llama API access (formatPrompt + generate) must be serialized —
        // the model/context is NOT thread-safe. Run both inside the serial queue.
        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
                let wireFormat = ToolWireFormat.detect(template: inference.chatTemplateString)
                let msgs = ToolPrompt.inject(messages: body.messages, tools: body.tools,
                                             toolChoice: body.tool_choice, format: wireFormat)
                let basePrompt = inference.formatPrompt(messages: msgs, format: wireFormat)
                let prompt = applyThinkingSwitch(prompt: basePrompt, enabled: body.thinkingEnabled, format: wireFormat)
                if FileLogger.shared.logPromptContent {
                    FileLogger.shared.verbose("chat request #\(currentRequest): FULL PROMPT >>>\n\(prompt)\n<<<")
                }
                result = try inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams)
            } catch {
                failure = error
            }

            // Periodic recreate bookkeeping moved to runPendingRecreate —
            // scheduled AFTER this response is delivered so the KV-cache
            // re-allocation never delays the client.
        }

        if let failure = failure {
            if let inferenceError = failure as? LlamaInference.InferenceError,
               case .contextTooFull = inferenceError {
                // The prompt leaves no room for a usable response — a client
                // error carrying the actionable message, not a server fault.
                return errorResponse(failure.localizedDescription, status: .badRequest)
            }
            return errorResponse(failure.localizedDescription, status: .internalServerError)
        }
        guard let result = result else {
            return errorResponse("Generation produced no result", status: .internalServerError)
        }
        // Only successful generations count toward (and pay for) the periodic
        // recreate; schedule it on the inference queue so the response below
        // is not held behind the KV-cache re-allocation.
        inferenceQueue.async { [weak self] in
            self?.runPendingRecreate(request: currentRequest, kind: "chat")
        }

        // Breadcrumb after generation, before response assembly: lets us tell
        // whether a process death (jetsam SIGKILL leaves no crash report) happens
        // during generation vs. during response encoding/return.
        FileLogger.shared.info("generation OK: #\(currentRequest) \(result.text.count) chars, encoding response")
        if FileLogger.shared.logPromptContent {
            FileLogger.shared.verbose("chat request #\(currentRequest): FULL RAW OUTPUT >>>\n\(result.text)\n<<<")
        }

        // Parse any <tool_call> envelopes back into structured tool calls and
        // strip ANSI escapes from the human-visible content. When the model
        // requested a tool, OpenAI clients expect `tool_calls` on the message
        // and `finish_reason: "tool_calls"` (mirrors llama-server behavior).
        let parsed = ToolCallParser.parse(result.text, tools: body.tools)
        // One honest policy for all paths: length only for real max-token
        // cutoffs; dropped envelopes / context_full become an explicit error
        // (see FinishReasonPolicy).
        let openAIReason = FinishReasonPolicy.map(engineFinish: result.finishReason,
                                                  hasToolCalls: !parsed.toolCalls.isEmpty,
                                                  outputUnusable: parsed.hasUnrecoveredEnvelope)
        let assistantMessage: ChatMessage
        if parsed.toolCalls.isEmpty {
            assistantMessage = ChatMessage(role: "assistant", content: parsed.cleanedContent)
        } else {
            // OpenAI sends content: null alongside tool_calls; keep any
            // surrounding prose only if the model emitted some.
            let content = parsed.cleanedContent.isEmpty ? nil : parsed.cleanedContent
            assistantMessage = ChatMessage(role: "assistant", content: content,
                                           tool_calls: parsed.toolCalls)
            FileLogger.shared.info("chat request #\(currentRequest): found \(parsed.toolCalls.count) tool calls")
            for call in parsed.toolCalls {
                FileLogger.shared.verbose("chat request #\(currentRequest): tool_call name=\(call.function.name) args=\(call.function.arguments)")
            }
        }

        let response = ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString)",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: body.model ?? inference.modelName,
            choices: [
                ChatCompletionChoice(
                    index: 0,
                    message: assistantMessage,
                    finish_reason: openAIReason
                )
            ],
            usage: Usage(
                prompt_tokens: result.promptTokens,
                completion_tokens: result.completionTokens,
                total_tokens: result.promptTokens + result.completionTokens
            )
        )
        let httpResponse = json(encodable: response, status: .ok)
        let responseSize = httpResponse.body.count
        let preview = String(data: httpResponse.body.prefix(200), encoding: .utf8) ?? ""
        FileLogger.shared.info("chat response #\(currentRequest): \(responseSize) bytes, finish=\(openAIReason), preview=\(preview.prefix(120))")
        return httpResponse
    }

    // MARK: - Helpers

    private func json<T: Encodable>(encodable: T, status: HTTPStatus) -> HTTPResponse {
        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(encodable) else {
            return HTTPResponse(.internalServerError)
        }
        let resp = HTTPResponse(status, headers: [
            "Content-Type": "application/json",
            "Content-Length": "\(data.count)",
            "Access-Control-Allow-Origin": "*"
        ], body: data)
        return resp
    }

    /// Suppresses reasoning by PREFILLING an already-closed, empty think block
    /// onto the assistant turn, so the model resumes generation *after* the
    /// `</think>` and emits only the final answer. This is structural — it does
    /// not depend on the model obeying a `/no_think` text directive (this merge
    /// ignores it) and it does NOT pollute the user's message content (injecting
    /// `/no_think` into the text made the model treat it as content to review).
    /// The format matches what the model emits itself (`<think>\n\n</think>\n\n`).
    /// Without this, short "reply with ONLY ..." / "Return JSON only" calls spend
    /// their whole token budget on reasoning and get truncated (finish=length)
    /// before producing the answer, which breaks strict clients. When the client
    /// opts in (`enable_thinking: true`) the prompt is returned unchanged.
    ///
    /// For the Qwen wire format the prompt already ends with the template's
    /// generation prompt *including* the `<think>` opener (`QwenPrompt` — the
    /// template's `enable_thinking`-undefined branch), so the "off" case only
    /// fills that block in and closes it. Guarded on the suffix, so a prompt
    /// that already carries the switch (e.g. produced by
    /// `llama_chat_apply_template`, which applies it itself) passes through
    /// untouched.
    private func applyThinkingSwitch(prompt: String, enabled: Bool, format: ToolWireFormat) -> String {
        if format == .qwenXML {
            guard prompt.hasSuffix(QwenPrompt.generationPrompt) else { return prompt }
            return enabled ? prompt : prompt + "\n</think>\n\n"
        }
        return enabled ? prompt : prompt + "<think>\n\n</think>\n\n"
    }


    static func samplingParams(from body: ChatCompletionRequest) -> LlamaInference.SamplingParams {
        var p = LlamaInference.SamplingParams()
        if let t = body.temperature { p.temperature = Float(t) }
        if let v = body.top_p { p.topP = Float(v) }
        if let v = body.top_k { p.topK = Int32(v) }
        if let v = body.min_p { p.minP = Float(v) }
        if let v = body.repeat_penalty { p.repeatPenalty = Float(v) }
        if let v = body.repeat_last_n { p.repeatLastN = Int32(v) }
        if let v = body.presence_penalty { p.presencePenalty = Float(v) }
        if let v = body.frequency_penalty { p.frequencyPenalty = Float(v) }
        if let v = body.seed, v >= 0 { p.seed = UInt32(truncatingIfNeeded: v) }
        if let stop = body.stop { p.stop = stop.values }
        return p
    }

    static func samplingParams(from body: CompletionRequest) -> LlamaInference.SamplingParams {
        var p = LlamaInference.SamplingParams()
        if let t = body.temperature { p.temperature = Float(t) }
        if let v = body.top_p { p.topP = Float(v) }
        if let v = body.top_k { p.topK = Int32(v) }
        if let v = body.min_p { p.minP = Float(v) }
        if let v = body.repeat_penalty { p.repeatPenalty = Float(v) }
        if let v = body.presence_penalty { p.presencePenalty = Float(v) }
        if let v = body.frequency_penalty { p.frequencyPenalty = Float(v) }
        if let v = body.seed, v >= 0 { p.seed = UInt32(truncatingIfNeeded: v) }
        if let stop = body.stop { p.stop = stop.values }
        return p
    }

    /// Legacy text-completion endpoint (/v1/completions). Feeds the raw prompt
    /// to the model without a chat template — some clients still use this.
    private func completionResponse(request: HTTPRequest) -> HTTPResponse {
        guard request.body.count <= Self.maxRequestBytes else {
            FileLogger.shared.warn("completion: body too large (\(request.body.count) bytes)")
            return errorResponse("Request body too large", status: .badRequest)
        }
        guard let body = try? JSONDecoder().decode(CompletionRequest.self, from: request.body) else {
            let preview = String(data: request.body.prefix(500), encoding: .utf8) ?? "<non-utf8>"
            FileLogger.shared.error("completion: JSON decode FAILED. body preview=\(preview)")
            FileLogger.shared.verbose("completion: FAILED RAW BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
            return errorResponse("Invalid request body", status: .badRequest)
        }
        let prompt = body.prompt.joined
        guard !prompt.isEmpty else {
            return errorResponse("`prompt` must not be empty", status: .badRequest)
        }

        let currentRequest = nextRequestNumber()
        FileLogger.shared.info("completion request #\(currentRequest): \(prompt.count) prompt chars, \(systemMemoryLog)")
        if FileLogger.shared.logPromptContent {
            FileLogger.shared.verbose("completion request #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
        }

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? Self.maxCompletionTokens), Self.maxCompletionTokens)
        let samplingParams = Self.samplingParams(from: body)

        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
                result = try inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams)
            } catch {
                failure = error
            }
            // Periodic recreate bookkeeping moved to runPendingRecreate —
            // scheduled AFTER this response is delivered (see chat path).
        }

        if let failure = failure {
            if let inferenceError = failure as? LlamaInference.InferenceError,
               case .contextTooFull = inferenceError {
                // The prompt leaves no room for a usable response — a client
                // error carrying the actionable message, not a server fault.
                return errorResponse(failure.localizedDescription, status: .badRequest)
            }
            return errorResponse(failure.localizedDescription, status: .internalServerError)
        }
        guard let result = result else {
            return errorResponse("Generation produced no result", status: .internalServerError)
        }
        inferenceQueue.async { [weak self] in
            self?.runPendingRecreate(request: currentRequest, kind: "completion")
        }

        let reason = FinishReasonPolicy.map(engineFinish: result.finishReason,
                                            hasToolCalls: false,
                                            outputUnusable: false)

        let response = CompletionResponse(
            id: "cmpl-\(UUID().uuidString)",
            object: "text_completion",
            created: Int(Date().timeIntervalSince1970),
            model: body.model ?? inference.modelName,
            choices: [CompletionChoice(index: 0,
                                        text: ToolCallParser.stripReasoning(ToolCallParser.stripANSI(result.text)),
                                        finish_reason: reason)],
            usage: Usage(prompt_tokens: result.promptTokens,
                         completion_tokens: result.completionTokens,
                         total_tokens: result.promptTokens + result.completionTokens))
        let httpResponse = json(encodable: response, status: .ok)
        FileLogger.shared.info("completion response #\(currentRequest): \(httpResponse.body.count) bytes, finish=\(reason)")
        return httpResponse
    }

    private func errorResponse(_ message: String, status: HTTPStatus) -> HTTPResponse {
        let error = APIError(error: .init(message: message, type: "invalid_request_error"))
        return json(encodable: error, status: status)
    }
}
