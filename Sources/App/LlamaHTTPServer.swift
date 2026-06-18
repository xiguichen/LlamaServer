import Foundation
import Telegraph
import os
import UIKit

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
    /// Counter of successful generations. Used to periodically recreate the
    /// llama_context and flush accumulated GPU/allocator state.
    private var generationCount = 0
    /// Recreate the context every N generations to avoid Metal GPU stalls.
    private let contextRecreateThreshold = 10
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
    private let connectionCountLock = OSAllocatedUnfairLock()
    private var _activeConnections = 0
    private var activeConnections: Int {
        get { connectionCountLock.withLock { _activeConnections } }
        set { connectionCountLock.withLock { _activeConnections = newValue } }
    }

    init(inference: LlamaInference) {
        self.inference = inference
        // Log iOS memory pressure events so we can correlate them with crashes.
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main) { _ in
                let mem = self.memoryMB
                FileLogger.shared.warn("MEMORY WARNING: current RSS=\(mem)MB, active connections=\(self.activeConnections)")
            }
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
                    FileLogger.shared.warn("[heartbeat] listener down — restarting")
                    self.restartListener()
                }
                FileLogger.shared.debug("[heartbeat] alive")
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
            FileLogger.shared.info("listener restarted on port \(listenPort)")
        } catch {
            FileLogger.shared.error("listener restart failed: \(error.localizedDescription)")
            self.server = oldServer
        }
    }

    private func makeServer() -> StreamableServer {
        let server = StreamableServer()
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
            FileLogger.shared.debug("intercept: claiming streaming request #\(self.nextRequestNumber())")
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
    private static let maxCompletionTokens = 4096

    /// CORS headers sent on the OPTIONS preflight. The API uses no cookies or
    /// Authorization, so a wildcard origin is appropriate for an open LAN server.
    private static let corsHeaderLines =
        "Access-Control-Allow-Origin: *\r\n" +
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" +
        "Access-Control-Allow-Headers: *\r\n"

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
        let currentRequest = nextRequestNumber()
        activeConnections += 1
        FileLogger.shared.debug("streaming #\(currentRequest): connection OPENED (total active: \(activeConnections))")
        FileLogger.shared.verbose("streaming #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")
        let mem = memoryMB
        let delta = previousFootprint == 0 ? 0 : Int64(mem) - Int64(previousFootprint)
        previousFootprint = mem
        FileLogger.shared.debug("streaming chat request #\(currentRequest): mem=\(mem)MB\(delta >= 0 ? "+" : "")\(delta)")

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

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? 512), Self.maxCompletionTokens)
        let chatId = "chatcmpl-\(UUID().uuidString)"
        let modelName = body.model ?? inference.modelName
        let created = Int(Date().timeIntervalSince1970)
        let samplingParams = Self.samplingParams(from: body)
        let includeUsage = body.stream_options?.include_usage ?? false
        // tool_choice "none" suppresses tools entirely.
        let toolsActive = (body.tools?.isEmpty == false) && !(body.tool_choice?.disablesTools ?? false)

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

        // Send SSE keep-alive comments every 5 s so clients don't time out
        // during ANY silent phase before the first real response bytes: the
        // serial-queue wait, the prompt decode, and — for tool requests — the
        // fully buffered generation (which emits nothing until parsed). The
        // timer keeps running until `stopKeepAlive()` fires at the first real
        // `data:` chunk; cancelling earlier (e.g. when the queue wait ends) left
        // the long decode/buffer window uncovered, so clients saw no response.
        let keepAliveActive = OSAllocatedUnfairLock(initialState: true)
        let keepAliveSource = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
        keepAliveSource.schedule(deadline: .now() + 5, repeating: .seconds(5), leeway: .seconds(1))
        keepAliveSource.setEventHandler { [weak connection] in
            guard let conn = connection else { return }
            guard keepAliveActive.withLock({ $0 }) else { return }
            conn.send(data: ": keepalive\n\n".data(using: .utf8)!, timeout: 5)
        }
        keepAliveSource.resume()
        FileLogger.shared.debug("streaming #\(currentRequest): keep-alive timer started")

        // Run generation on the serial queue. Each token is flushed as an SSE event.
        inferenceQueue.async { [weak self] in
            guard let self = self else { keepAliveSource.cancel(); return }
            FileLogger.shared.debug("streaming #\(currentRequest): async block started (queue wait ended)")

            // Stop the keep-alive comment stream the instant we begin writing
            // real SSE data, so heartbeat comments never interleave with the
            // response. Idempotent: safe to call repeatedly / from any token.
            func stopKeepAlive() {
                keepAliveActive.withLock { $0 = false }
                keepAliveSource.cancel()
            }

            // When tools are active we must buffer the whole response so the
            // <tool_call> envelope can be parsed into structured tool_calls;
            // token-by-token streaming can't reliably do that. Plain chats still
            // stream incrementally for good UX.
            func sendChunk(delta: Delta, finishReason: String?) {
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
            }

            var generatedTokens = 0
            var lastTokenTime = CFAbsoluteTimeGetCurrent()
            // Strips <think> reasoning from the streamed (non-tool) content so
            // clients receive only the final answer, not the chain-of-thought.
            let thinkFilter = ReasoningStreamFilter()

            do {
                let msgs = self.injectToolDefs(messages: body.messages, tools: body.tools,
                                               toolChoice: body.tool_choice)
                let prompt = self.inference.formatPrompt(messages: msgs)
                FileLogger.shared.info("streaming #\(currentRequest): prompt built (\(prompt.count) chars), starting generation")
                FileLogger.shared.verbose("streaming #\(currentRequest): FULL PROMPT >>>\n\(prompt)\n<<<")
                let genResult = try self.inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams) { text in
                    generatedTokens += 1
                    let now = CFAbsoluteTimeGetCurrent()
                    if now - lastTokenTime >= 10 {
                        FileLogger.shared.debug("streaming #\(currentRequest): generated \(generatedTokens) tokens so far")
                        lastTokenTime = now
                    }
                    // Buffer (don't stream) when a tool call may be forming.
                    guard !toolsActive else { return true }
                    let visible = thinkFilter.feed(ToolCallParser.stripANSI(text))
                    guard !visible.isEmpty else { return true }
                    stopKeepAlive()
                    FileLogger.shared.verbose("streaming #\(currentRequest): chunk >>>\(visible)<<<")
                    sendChunk(delta: Delta(role: nil, content: visible), finishReason: nil)
                    return true
                }
                FileLogger.shared.info("streaming #\(currentRequest): generation done, \(generatedTokens) tokens, finish=\(genResult.finishReason)")
                FileLogger.shared.verbose("streaming #\(currentRequest): FULL RAW OUTPUT >>>\n\(genResult.text)\n<<<")
                // Buffered tool requests stream nothing during generation; make
                // sure the heartbeat is stopped before the first real chunk below.
                stopKeepAlive()
                // Flush any content the reasoning filter held back (non-tool path).
                if !toolsActive {
                    let tail = thinkFilter.flush()
                    if !tail.isEmpty {
                        sendChunk(delta: Delta(role: nil, content: tail), finishReason: nil)
                    }
                }
                self.generationCount += 1
                if self.generationCount >= self.contextRecreateThreshold {
                    FileLogger.shared.info("streaming #\(currentRequest): recreating context (threshold reached)")
                    try self.inference.recreateContext()
                    self.generationCount = 0
                }

                // Map the engine's finish reason onto the OpenAI vocabulary.
                var streamReason: String
                switch genResult.finishReason {
                case "length", "context_full": streamReason = "length"
                default:                       streamReason = "stop"
                }

                if toolsActive {
                    FileLogger.shared.debug("streaming #\(currentRequest): tools active, parsing tool calls")
                    let parsed = ToolCallParser.parse(genResult.text)
                    if !parsed.toolCalls.isEmpty {
                        streamReason = "tool_calls"
                        FileLogger.shared.info("streaming #\(currentRequest): found \(parsed.toolCalls.count) tool calls")
                        for call in parsed.toolCalls {
                            FileLogger.shared.verbose("streaming #\(currentRequest): tool_call name=\(call.function.name) args=\(call.function.arguments)")
                        }
                        if !parsed.cleanedContent.isEmpty {
                            sendChunk(delta: Delta(role: nil, content: parsed.cleanedContent),
                                      finishReason: nil)
                            FileLogger.shared.debug("streaming #\(currentRequest): sent cleaned content chunk")
                        }
                        let streamingCalls = parsed.toolCalls.enumerated().map {
                            StreamingToolCall(index: $0.offset, from: $0.element)
                        }
                        sendChunk(delta: Delta(role: nil, content: nil,
                                               tool_calls: streamingCalls),
                                  finishReason: nil)
                        FileLogger.shared.debug("streaming #\(currentRequest): sent tool_calls chunk")
                    } else if !parsed.cleanedContent.isEmpty {
                        // No tool call after all — flush the buffered content.
                        FileLogger.shared.debug("streaming #\(currentRequest): no tool calls, flushing content")
                        sendChunk(delta: Delta(role: nil, content: parsed.cleanedContent),
                                  finishReason: nil)
                    } else {
                        FileLogger.shared.warn("streaming #\(currentRequest): no tool calls and no content")
                    }
                }

                // Final chunk carries the finish reason.
                sendChunk(delta: Delta(role: nil, content: nil), finishReason: streamReason)
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
                stopKeepAlive()
                FileLogger.shared.error("streaming #\(currentRequest): generation error: \(error.localizedDescription)")
                sendChunk(delta: Delta(role: nil, content: nil), finishReason: "error")
            }
            connection.send(data: "data: [DONE]\n\n".data(using: .utf8)!, timeout: 10)
            FileLogger.shared.debug("streaming #\(currentRequest): [DONE] sent")
            connection.close(immediately: false)
            self.activeConnections -= 1
            FileLogger.shared.debug("streaming #\(currentRequest): connection CLOSED (total active: \(self.activeConnections), genCount: \(self.generationCount))")
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
        guard !body.messages.isEmpty else {
            return errorResponse("`messages` must not be empty", status: .badRequest)
        }

        let currentRequest = nextRequestNumber()
        let mem = memoryMB
        let delta = previousFootprint == 0 ? 0 : Int64(mem) - Int64(previousFootprint)
        previousFootprint = mem
        FileLogger.shared.debug("chat request #\(currentRequest): mem=\(mem)MB\(delta >= 0 ? "+" : "")\(delta)")
        FileLogger.shared.verbose("chat request #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? 512), Self.maxCompletionTokens)
        let samplingParams = Self.samplingParams(from: body)

        // All llama API access (formatPrompt + generate) must be serialized —
        // the model/context is NOT thread-safe. Run both inside the serial queue.
        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
                let msgs = injectToolDefs(messages: body.messages, tools: body.tools,
                                          toolChoice: body.tool_choice)
                let prompt = inference.formatPrompt(messages: msgs)
                FileLogger.shared.verbose("chat request #\(currentRequest): FULL PROMPT >>>\n\(prompt)\n<<<")
                result = try inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams)

                // Periodically recreate the context to flush accumulated
                // GPU/allocator state that could otherwise cause Metal to hang.
                generationCount += 1
                if generationCount >= contextRecreateThreshold {
                    try inference.recreateContext()
                    generationCount = 0
                }
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
        FileLogger.shared.info("generation OK: #\(currentRequest) \(result.text.count) chars, encoding response")
        FileLogger.shared.verbose("chat request #\(currentRequest): FULL RAW OUTPUT >>>\n\(result.text)\n<<<")

        var openAIReason: String
        switch result.finishReason {
        case "eog", "eog_immediate", "stopped":
            openAIReason = "stop"
        case "length", "context_full":
            openAIReason = "length"
        default:
            openAIReason = "stop"
        }

        // Parse any <tool_call> envelopes back into structured tool calls and
        // strip ANSI escapes from the human-visible content. When the model
        // requested a tool, OpenAI clients expect `tool_calls` on the message
        // and `finish_reason: "tool_calls"` (mirrors llama-server behavior).
        let parsed = ToolCallParser.parse(result.text)
        let assistantMessage: ChatMessage
        if parsed.toolCalls.isEmpty {
            assistantMessage = ChatMessage(role: "assistant", content: parsed.cleanedContent)
        } else {
            openAIReason = "tool_calls"
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
        let responseSize = httpResponse.body?.count ?? 0
        let preview = String(data: httpResponse.body?.prefix(200) ?? Data(), encoding: .utf8) ?? ""
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

    /// Serializes tool definitions into the system message so the model knows
    /// which tools are available and how to call them via <tool_call> tags.
    /// Honors `tool_choice`: "none" skips injection entirely; "required" / a
    /// named function strengthens the instruction so the model must emit a call.
    private func injectToolDefs(messages: [ChatMessage],
                               tools: [ToolDefinition]?,
                               toolChoice: ToolChoiceField?) -> [ChatMessage] {
        guard let tools = tools, !tools.isEmpty else { return messages }
        if toolChoice?.disablesTools == true { return messages }

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let toolsData = try? encoder.encode(tools),
              let toolsStr = String(data: toolsData, encoding: .utf8) else {
            return messages
        }

        var instruction = """
        You have access to the following tools. When you want to use a tool, respond with EXACTLY this format (no markdown, no extra text around it):

        <tool_call>{"name": "tool_name", "arguments": {"param1": "val1", "param2": "val2"}}</tool_call>

        IMPORTANT: The `arguments` object MUST include ALL required parameters listed in the tool's JSON schema. Never omit a required parameter.
        """

        if let forced = toolChoice?.forcedFunctionName {
            instruction += "\n\nYou MUST call the function named \"\(forced)\" with a <tool_call> and nothing else."
        } else if toolChoice?.requiresTool == true {
            instruction += "\n\nYou MUST respond with a <tool_call> and nothing else."
        }

        instruction += "\n\nAvailable tools:\n\(toolsStr)"

        var modified = messages
        if let idx = modified.firstIndex(where: { $0.role == "system" }) {
            let existing = modified[idx].content?.textValue ?? ""
            modified[idx] = ChatMessage(role: "system", content: instruction + "\n\n" + existing)
        } else {
            modified.insert(ChatMessage(role: "system", content: instruction), at: 0)
        }
        return modified
    }

    /// Builds the engine sampling parameters from an OpenAI chat request,
    /// clamping to safe ranges. Unspecified fields fall back to llama.cpp's
    /// defaults so behavior is unchanged unless a client opts in.
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
        FileLogger.shared.info("completion request #\(currentRequest): \(prompt.count) prompt chars")
        FileLogger.shared.verbose("completion request #\(currentRequest): RAW REQUEST BODY >>>\(String(data: request.body, encoding: .utf8) ?? "<non-utf8>")<<<")

        let maxTokens = min(max(1, body.resolvedMaxTokens ?? 512), Self.maxCompletionTokens)
        let samplingParams = Self.samplingParams(from: body)

        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
                result = try inference.generate(prompt: prompt,
                                                maxTokens: maxTokens,
                                                params: samplingParams)
                generationCount += 1
                if generationCount >= contextRecreateThreshold {
                    try inference.recreateContext()
                    generationCount = 0
                }
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

        let reason: String
        switch result.finishReason {
        case "length", "context_full": reason = "length"
        default:                       reason = "stop"
        }

        let response = CompletionResponse(
            id: "cmpl-\(UUID().uuidString)",
            object: "text_completion",
            created: Int(Date().timeIntervalSince1970),
            model: body.model ?? inference.modelName,
            choices: [CompletionChoice(index: 0,
                                        text: ToolCallParser.stripANSI(result.text),
                                        finish_reason: reason)],
            usage: Usage(prompt_tokens: result.promptTokens,
                         completion_tokens: result.completionTokens,
                         total_tokens: result.promptTokens + result.completionTokens))
        let httpResponse = json(encodable: response, status: .ok)
        FileLogger.shared.info("completion response #\(currentRequest): \(httpResponse.body?.count ?? 0) bytes, finish=\(reason)")
        return httpResponse
    }

    private func errorResponse(_ message: String, status: HTTPStatus) -> HTTPResponse {
        let error = APIError(error: .init(message: message, type: "invalid_request_error"))
        return json(encodable: error, status: status)
    }
}
