import Foundation
import Telegraph

/// HTTPS server exposing an OpenAI-compatible API backed by `LlamaInference`.
///
/// Endpoints:
///   GET  /health                 -> { "status": "ok" }
///   GET  /v1/models              -> OpenAI model list
///   POST /v1/chat/completions    -> OpenAI chat completion (non-streaming)
///
/// TLS identity is loaded from the bundled `server.p12` (self-signed, generated
/// by scripts/generate-cert.sh). LAN clients must trust / ignore the self-signed
/// cert (e.g. curl --insecure, or OpenAI clients with verification disabled).
final class LlamaHTTPServer {

    enum ServerError: Error, LocalizedError {
        case missingCertificate
        case identityLoad

        var errorDescription: String? {
            switch self {
            case .missingCertificate: return "server.p12 not found in app bundle."
            case .identityLoad:       return "Could not load TLS identity from server.p12."
            }
        }
    }

    private let inference: LlamaInference
    private let server = Server()
    /// Serializes access to the (non-thread-safe) llama context.
    private let inferenceQueue = DispatchQueue(label: "llama.inference.serial")
    private let p12Password: String

    init(inference: LlamaInference, p12Password: String = "llamaserver") {
        self.inference = inference
        self.p12Password = p12Password
        configureRoutes()
    }

    var isRunning: Bool { server.isRunning }
    var port: Int { Int(server.port) }

    // MARK: - Start / Stop

    func start(port: UInt16) throws {
        guard let p12URL = Bundle.main.url(forResource: "server", withExtension: "p12") else {
            throw ServerError.missingCertificate
        }
        guard let identity = CertificateIdentity(p12URL: p12URL, passphrase: p12Password) else {
            throw ServerError.identityLoad
        }
        // Enable HTTPS with our self-signed identity.
        server.tlsConfig = TLSConfig(identity: identity)
        try server.start(port: Int(port))
    }

    func stop() {
        server.stop()
    }

    // MARK: - Routes

    private func configureRoutes() {
        server.route(.GET, "health") { _ in
            HTTPResponse(.ok, headers: ["Content-Type": "application/json"],
                         body: #"{"status":"ok"}"#.data(using: .utf8)!)
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

    private func chatCompletionResponse(request: HTTPRequest) -> HTTPResponse {
        let decoder = JSONDecoder()
        guard let body = try? decoder.decode(ChatCompletionRequest.self, from: request.body) else {
            return errorResponse("Invalid request body", status: .badRequest)
        }
        guard !body.messages.isEmpty else {
            return errorResponse("`messages` must not be empty", status: .badRequest)
        }

        let prompt = inference.formatPrompt(messages: body.messages)
        let temperature = Float(body.temperature ?? 0.8)
        let topP = Float(body.top_p ?? 0.95)
        let maxTokens = body.max_tokens ?? 512

        // Run inference synchronously on the serial queue (one request at a time).
        var result: LlamaInference.GenerationResult?
        var failure: Error?
        inferenceQueue.sync {
            do {
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
        return HTTPResponse(status, headers: ["Content-Type": "application/json"], body: data)
    }

    private func errorResponse(_ message: String, status: HTTPStatus) -> HTTPResponse {
        let error = APIError(error: .init(message: message, type: "invalid_request_error"))
        return json(encodable: error, status: status)
    }
}
