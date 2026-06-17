import Foundation
import llama   // module provided by vendor/llama.xcframework

/// Thin Swift wrapper around the llama.cpp C API.
///
/// Mirrors the flow of llama.cpp's `examples/simple/simple.cpp`:
///   load model -> create context -> tokenize prompt -> decode loop -> sample.
///
/// NOTE: a single `LlamaInference` / `llama_context` is NOT thread-safe. Callers
/// must serialize access (the HTTP server does this with a serial queue).
///
/// This file is pinned to a recent llama.cpp API (functions such as
/// `llama_model_load_from_file`, `llama_init_from_model`, `llama_model_get_vocab`,
/// the `llama_sampler_*` chain API, and `llama_vocab_is_eog`). If you bump the
/// pinned llama.cpp tag in scripts/build-llama-xcframework.sh and the build fails,
/// this is the file to adjust.
final class LlamaInference {

    enum InferenceError: Error, LocalizedError {
        case backendInit
        case modelLoad(String)
        case contextInit
        case tokenize
        case decode
        case insufficientMemory(String)

        var errorDescription: String? {
            switch self {
            case .backendInit:           return "Failed to initialize llama backend."
            case .modelLoad(let p):      return "Failed to load model \((p as NSString).lastPathComponent). The architecture may be unsupported by this llama.cpp build."
            case .contextInit:           return "Failed to create llama context."
            case .tokenize:              return "Failed to tokenize input."
            case .decode:                return "llama_decode failed (context may be full)."
            case .insufficientMemory(let m): return m
            }
        }
    }

    private let model: OpaquePointer
    private var context: OpaquePointer
    private let vocab: OpaquePointer
    private let threadCount: Int32

    /// The effective context size actually used (may be smaller than requested
    /// after clamping to the device's memory budget and the model's train window).
    let contextSize: Int

    /// Logical max tokens per llama_decode call. A single decode must NOT exceed
    /// this or llama.cpp aborts (ggml_abort) — the prompt is decoded in chunks.
    let batchSize: Int

    let modelName: String

    // MARK: - Lifecycle

    init(modelPath: String, contextSize requestedContext: Int = 4096, threads: Int32 = 0) throws {
        // Backend init is idempotent across instances within a process.
        llama_backend_init()

        self.modelName = (modelPath as NSString).lastPathComponent

        // Memory budget. iOS jetsam-kills (SIGKILL — uncatchable) apps well below
        // total RAM, so be conservative. We use this to fail *gracefully* with a
        // message instead of letting a too-big load/KV-cache crash the app.
        let totalRAM = Int(ProcessInfo.processInfo.physicalMemory)
        let budget = Int(Double(totalRAM) * 0.55)
        let computeReserve = 320 * 1024 * 1024   // activation/compute buffers headroom

        // Pre-check the model file size before attempting the load that would
        // otherwise OOM-kill the whole app.
        if let attrs = try? FileManager.default.attributesOfItem(atPath: modelPath),
           let fileSize = (attrs[.size] as? NSNumber)?.intValue,
           fileSize + computeReserve > budget {
            let f = ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file)
            let b = ByteCountFormatter.string(fromByteCount: Int64(budget), countStyle: .file)
            throw InferenceError.insufficientMemory(
                "Model is \(f) but only ~\(b) is usable on this device. Try a smaller model (1–3B, Q4).")
        }

        var modelParams = llama_model_default_params()
        // Offload everything to Metal GPU when available.
        modelParams.n_gpu_layers = 999

        // Durable breadcrumb flushed to disk: if llama.cpp hard-aborts
        // (GGML_ABORT/SIGABRT) on an unsupported architecture or OOMs during
        // load, this line survives the crash in llamaserver.log — so a crash is
        // never "without any log", and it's readable on-device (no Mac needed).
        FileLogger.shared.log("loading model '\(self.modelName)' (requested ctx \(requestedContext), budget \(budget / (1024 * 1024)) MB)")

        guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
            FileLogger.shared.log("model load returned NULL for '\(self.modelName)'")
            throw InferenceError.modelLoad(modelPath)
        }
        FileLogger.shared.log("model loaded OK: '\(self.modelName)'")
        self.model = loadedModel

        guard let loadedVocab = llama_model_get_vocab(loadedModel) else {
            llama_model_free(loadedModel)
            throw InferenceError.contextInit
        }
        self.vocab = loadedVocab

        // Clamp the context size to (a) the model's trained window, (b) a hard
        // ceiling, and (c) what the KV cache can fit in the memory budget.
        // Without this, a large value (e.g. 140000) allocates a multi-GB KV cache
        // and the app is instantly OOM-killed.
        let nLayer    = Int(llama_model_n_layer(loadedModel))
        let nHead     = max(1, Int(llama_model_n_head(loadedModel)))
        let nHeadKV   = max(1, Int(llama_model_n_head_kv(loadedModel)))
        let nEmbd     = Int(llama_model_n_embd(loadedModel))
        let nCtxTrain = Int(llama_model_n_ctx_train(loadedModel))
        let modelSize = Int(llama_model_size(loadedModel))

        let headDim = max(1, nEmbd / nHead)
        // K + V, f16 (2 bytes), across all layers, per token.
        let kvBytesPerToken = max(1, 2 * nLayer * (headDim * nHeadKV) * 2)

        var effective = min(max(256, requestedContext), 32768)
        if nCtxTrain > 0 { effective = min(effective, nCtxTrain) }

        let kvBudget = budget - modelSize - computeReserve
        if kvBudget < kvBytesPerToken * 256 {
            llama_model_free(loadedModel)
            throw InferenceError.insufficientMemory(
                "Not enough free memory for a usable KV cache after loading this model. Try a smaller model.")
        }
        let maxCtxByMemory = (kvBudget / kvBytesPerToken / 256) * 256   // round down to 256
        effective = min(effective, maxCtxByMemory)
        self.contextSize = effective
        self.batchSize = min(effective, 512)

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(effective)
        ctxParams.n_batch = UInt32(batchSize)
        let hwThreads = Int32(ProcessInfo.processInfo.activeProcessorCount)
        let useThreads = threads > 0 ? threads : max(1, hwThreads)
        ctxParams.n_threads = useThreads
        ctxParams.n_threads_batch = useThreads
        self.threadCount = useThreads

        FileLogger.shared.log("creating context (\(effective) tokens) for '\(self.modelName)'")
        guard let ctx = llama_init_from_model(loadedModel, ctxParams) else {
            FileLogger.shared.log("context creation returned NULL")
            llama_model_free(loadedModel)
            throw InferenceError.contextInit
        }
        self.context = ctx
        FileLogger.shared.log("context ready (\(effective) tokens)")
    }

    /// Frees the old llama_context and creates a fresh one. This is called after
    /// many generations to flush any accumulated GPU/allocator state that could
    /// cause Metal to hang on `llama_sampler_sample`.
    func recreateContext() throws {
        llama_free(context)

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)
        ctxParams.n_batch = UInt32(batchSize)
        ctxParams.n_threads = threadCount
        ctxParams.n_threads_batch = threadCount

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            throw InferenceError.contextInit
        }
        context = ctx
        cachedTokenCount = 0
        lastPromptText = nil
        cachedTokenIds = []
        FileLogger.shared.log("context recreated (\(contextSize) tokens)")
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
        // Intentionally do NOT call llama_backend_free() here: other instances
        // in the same process may still need the backend. It is freed on exit.
    }

    // MARK: - Prompt formatting (chat template)

    /// Applies the model's built-in chat template (when present) to a list of
    /// messages, falling back to a generic ChatML-style layout.
    func formatPrompt(messages: [ChatMessage]) -> String {
        // Build C array of llama_chat_message with strdup'd strings.
        var cMessages: [llama_chat_message] = []
        var allocated: [UnsafeMutablePointer<CChar>] = []
        cMessages.reserveCapacity(messages.count)
        // Free everything on every exit path (including early returns below).
        defer { allocated.forEach { free($0) } }

        for message in messages {
            // strdup returns nil only on allocation failure — fall back safely
            // instead of force-unwrapping (which would crash).
            guard let rolePtr = strdup(message.role) else {
                return fallbackPrompt(messages: messages)
            }
            allocated.append(rolePtr)
            // content may be nil (e.g. an assistant turn that only has
            // tool_calls). strdup(nil) returns nil → never pass nil to the
            // template engine; reconstruct text from tool_calls instead.
            guard let contentPtr = strdup(effectiveContent(of: message)) else {
                return fallbackPrompt(messages: messages)
            }
            allocated.append(contentPtr)
            cMessages.append(
                llama_chat_message(role: UnsafePointer(rolePtr),
                                   content: UnsafePointer(contentPtr))
            )
        }

        let tmpl = llama_model_chat_template(model, nil) // default template, if any

        // First call to size the buffer.
        var bufferSize = 0
        for message in messages {
            let contentText = effectiveContent(of: message)
            bufferSize += message.role.utf8.count + contentText.utf8.count + 16
        }
        bufferSize = max(bufferSize * 2, 1024)

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let written = cMessages.withUnsafeBufferPointer { ptr -> Int32 in
            llama_chat_apply_template(
                tmpl,
                ptr.baseAddress,
                ptr.count,
                true,            // add assistant generation prompt
                &buffer,
                Int32(buffer.count)
            )
        }

        if written < 0 {
            return fallbackPrompt(messages: messages)
        }

        if Int(written) >= buffer.count {
            // Grow and retry once. Use `>=` because when `written == buffer.count`
            // the buffer is completely filled with no room for the trailing NUL,
            // and `String(cString:)` below would then read past the end of the
            // array (out-of-bounds read / garbage / crash).
            buffer = [CChar](repeating: 0, count: Int(written) + 1)
            let retry = cMessages.withUnsafeBufferPointer { ptr -> Int32 in
                llama_chat_apply_template(tmpl, ptr.baseAddress, ptr.count, true,
                                          &buffer, Int32(buffer.count))
            }
            if retry < 0 { return fallbackPrompt(messages: messages) }
            return String(cString: buffer)
        }

        // `written` is the length; the buffer is already NUL padded.
        return String(cString: buffer)
    }

    private func fallbackPrompt(messages: [ChatMessage]) -> String {
        var out = ""
        for message in messages {
            out += "<|im_start|>\(message.role)\n\(effectiveContent(of: message))<|im_end|>\n"
        }
        out += "<|im_start|>assistant\n"
        return out
    }

    /// The text to feed the chat template for a message. Normally this is the
    /// message content, but an assistant turn may carry only `tool_calls` with
    /// `content: nil`; reconstruct the `<tool_call>` envelope so multi-turn tool
    /// conversations present a consistent history to the model.
    private func effectiveContent(of message: ChatMessage) -> String {
        if let content = message.content?.textValue, !content.isEmpty {
            return content
        }
        guard let toolCalls = message.tool_calls, !toolCalls.isEmpty else {
            return message.content?.textValue ?? ""
        }
        return toolCalls.map { call in
            "<tool_call>{\"name\": \"\(call.function.name)\", \"arguments\": \(call.function.arguments)}</tool_call>"
        }.joined(separator: "\n")
    }

    // MARK: - Tokenization

    private func tokenize(_ text: String, addBOS: Bool) throws -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        let utf8Count = Int32(text.utf8.count)

        // Negative return = required token count.
        let needed = -llama_tokenize(vocab, utf8, utf8Count, nil, 0, addBOS, true)
        guard needed > 0 else {
            // Empty input still tokenizes to BOS in some models; allow 0.
            if needed == 0 { return [] }
            throw InferenceError.tokenize
        }

        var tokens = [llama_token](repeating: 0, count: Int(needed))
        let count = llama_tokenize(vocab, utf8, utf8Count, &tokens, needed, addBOS, true)
        guard count >= 0 else { throw InferenceError.tokenize }
        return Array(tokens.prefix(Int(count)))
    }

    /// Returns the raw bytes for a token's piece. Multi-byte UTF-8 characters
    /// can be split across two tokens, so callers must accumulate bytes and only
    /// decode complete UTF-8 sequences (see `flushDecodableUTF8`). Decoding each
    /// token's bytes individually drops any character whose bytes straddle a
    /// token boundary — which silently corrupts CJK, emoji, and accented text.
    private func pieceBytes(for token: llama_token) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 256)
        let n = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, true)
        if n < 0 {
            var bigger = [CChar](repeating: 0, count: Int(-n) + 1)
            let m = llama_token_to_piece(vocab, token, &bigger, Int32(bigger.count), 0, true)
            guard m > 0 else { return [] }
            return bigger.prefix(Int(m)).map { UInt8(bitPattern: $0) }
        }
        guard n > 0 else { return [] }
        return buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }
    }

    /// Pulls every complete UTF-8 character out of `pending`, returning the
    /// decoded string and leaving any trailing incomplete sequence buffered.
    /// A well-formed UTF-8 continuation is at most 4 bytes, so if the buffer
    /// grows past that and still won't decode the leading bytes are genuine
    /// garbage (not a split char) and are flushed lossily to avoid a stall.
    private func flushDecodableUTF8(_ pending: inout [UInt8]) -> String {
        guard !pending.isEmpty else { return "" }
        var tryLen = pending.count
        let minLen = max(0, pending.count - 3)
        while tryLen > minLen {
            if let s = String(bytes: pending[0..<tryLen], encoding: .utf8) {
                pending.removeFirst(tryLen)
                return s
            }
            tryLen -= 1
        }
        // Leading bytes are not the tail of an incomplete char — flush lossily.
        if pending.count > 4 {
            let s = String(decoding: pending, as: UTF8.self)
            pending.removeAll(keepingCapacity: true)
            return s
        }
        return ""
    }

    /// Returns the range of the earliest stop string found in `text`, if any.
    private func firstStop(in text: String, stops: [String]) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for s in stops {
            if let r = text.range(of: s) {
                if earliest == nil || r.lowerBound < earliest!.lowerBound {
                    earliest = r
                }
            }
        }
        return earliest
    }

    /// Length (in characters) of the longest suffix of `text` that is a proper
    /// prefix of some stop string. While streaming, this many trailing chars
    /// must be withheld in case the next token completes a stop sequence.
    private func partialStopSuffixLength(of text: String, stops: [String]) -> Int {
        var maxLen = 0
        for s in stops {
            let upper = min(s.count - 1, text.count)
            var k = upper
            while k > 0 {
                if s.hasPrefix(String(text.suffix(k))) {
                    maxLen = max(maxLen, k)
                    break
                }
                k -= 1
            }
        }
        return maxLen
    }

    // MARK: - Generation

    /// Full set of sampling controls honored by `generate`, mirroring the
    /// subset of llama-server's parameters that matter on-device. Defaults match
    /// llama.cpp's defaults so an unspecified field is a no-op.
    struct SamplingParams {
        var temperature: Float = 0.8
        var topP: Float = 0.95
        var topK: Int32 = 40
        var minP: Float = 0.05
        var repeatPenalty: Float = 1.0
        var repeatLastN: Int32 = 64
        var frequencyPenalty: Float = 0.0
        var presencePenalty: Float = 0.0
        var seed: UInt32 = 0xFFFFFFFF
        /// Stop strings: generation halts (finish_reason "stop") when any appears.
        var stop: [String] = []
    }

    struct GenerationResult {
        let text: String
        let promptTokens: Int
        let completionTokens: Int
        let finishReason: String
    }

    /// Tracks the total number of tokens in the KV cache after the last generate()
    /// call (prompt + completion). Used to reuse cache when consecutive requests
    /// share a prefix (e.g. the same conversation history).
    private var cachedTokenCount: Int = 0
    /// The text of the prompt from the last generate() call, for prefix matching.
    private var lastPromptText: String?
    /// The actual token IDs from the last generate() call (prompt + completion),
    /// stored for token-level prefix verification during cache reuse.
    private var cachedTokenIds: [llama_token] = []

    /// Generates a completion for the given (already chat-formatted) prompt.
    /// `onToken` is called for each decoded piece; return `false` to stop early.
    func generate(prompt: String,
                  maxTokens: Int,
                  params: SamplingParams,
                  onToken: ((String) -> Bool)? = nil) throws -> GenerationResult {

        let promptTokens = try tokenize(prompt, addBOS: true)
        guard !promptTokens.isEmpty else { throw InferenceError.tokenize }

        // Reuse KV cache when the new prompt extends the previous one and has
        // at least as many tokens. This avoids re-decoding the conversation
        // prefix on every request — a big win for multi-turn dialogues.
        //
        // Both string-level (hasPrefix) and token-level (cachedTokenIds)
        // checks are required: the string check ensures the text matches at
        // the application level, while the token check verifies that the old
        // token IDs from the cache match the re-tokenized prefix. BPE is
        // deterministic for the same byte sequence, so the token check should
        // always pass when the string check does — but we verify anyway to
        // guard against edge cases (e.g. lossy pieces, tokenizer quirks).
        let prefixMatches = cachedTokenCount > 0
            && promptTokens.count >= cachedTokenCount
            && cachedTokenIds.count >= cachedTokenCount
            && cachedTokenIds.prefix(cachedTokenCount).elementsEqual(promptTokens.prefix(cachedTokenCount))
        let reuseCache = prefixMatches
            && lastPromptText != nil && prompt.hasPrefix(lastPromptText!)
        if reuseCache {
            FileLogger.shared.log("KV cache reused (\(cachedTokenCount) cached, \(promptTokens.count) total prompt tokens)")
        } else {
            cachedTokenIds = []
            if let memory = llama_get_memory(context) {
                llama_memory_clear(memory, true)
                FileLogger.shared.log("KV cache cleared\(cachedTokenCount > 0 ? " (last cached \(cachedTokenCount))" : "")")
            } else {
                FileLogger.shared.log("llama_get_memory returned nil — KV cache NOT cleared, invalidating tracking")
                cachedTokenCount = 0
                lastPromptText = nil
                cachedTokenIds = []
            }
        }
        lastPromptText = prompt

        // Build the sampler chain. Order mirrors llama.cpp's default pipeline:
        // penalties -> top_k -> top_p -> min_p -> temperature -> distribution.
        var samplerParams = llama_sampler_chain_default_params()
        samplerParams.no_perf = true
        let sampler = llama_sampler_chain_init(samplerParams)
        defer { llama_sampler_free(sampler) }

        // Repetition penalties apply regardless of temperature.
        if params.repeatPenalty != 1.0 || params.frequencyPenalty != 0.0 || params.presencePenalty != 0.0 {
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                params.repeatLastN, params.repeatPenalty,
                params.frequencyPenalty, params.presencePenalty))
        }

        if params.temperature <= 0 {
            // Greedy: deterministic argmax, ignores top_k/top_p/min_p.
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        } else {
            if params.topK > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.topK))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.topP, 1))
            if params.minP > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_min_p(params.minP, 1))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed))
        }

        // Cache and error recovery: if anything below fails, the llama context's
        // internal position counter and KV cache may be in an undefined state
        // (partial prompt decode and/or partial generation).  Invalidate the
        // tracking so the next call always clears the cache and starts fresh.
        var generationSucceeded = false
        defer {
            if !generationSucceeded {
                if cachedTokenCount > 0 || lastPromptText != nil {
                    FileLogger.shared.log("generation failed — cache tracking invalidated")
                }
                cachedTokenCount = 0
                lastPromptText = nil
                cachedTokenIds = []
            }
        }

        // Decode the prompt. Breadcrumb first: a too-long prompt or a native
        // abort here would otherwise kill the process with no trace.
        FileLogger.shared.log("decoding prompt (\(promptTokens.count) tokens, ctx \(contextSize), batch \(batchSize))")
        if promptTokens.count >= contextSize {
            throw InferenceError.decode
        }
        // A single llama_decode must NOT exceed n_batch or llama.cpp aborts
        // (ggml_abort/SIGABRT), so feed the prompt in batchSize-sized chunks.
        // Positions auto-continue across calls from the context's KV state.
        // The batch from llama_batch_get_one points INTO this buffer, so the
        // decode must happen while the pointer is valid (inside the closure).
        var tokens = promptTokens
        let total = tokens.count
        var offset = reuseCache ? min(cachedTokenCount, total) : 0
        while offset < total {
            let count = min(batchSize, total - offset)
            let status = tokens.withUnsafeMutableBufferPointer { buf -> Int32 in
                let chunk = buf.baseAddress!.advanced(by: offset)
                let batch = llama_batch_get_one(chunk, Int32(count))
                return llama_decode(context, batch)
            }
            guard status == 0 else { throw InferenceError.decode }
            offset += count
        }

        // Diagnostic: how does the formatted prompt end? It should include the
        // assistant generation header (e.g. "<|im_start|>assistant").
        let promptTail = String(prompt.suffix(180)).replacingOccurrences(of: "\n", with: "\\n")
        FileLogger.shared.log("prompt tail: …\(promptTail)")

        var output = ""
        var pending: [UInt8] = []        // incomplete trailing UTF-8 bytes
        var emittedCount = 0             // chars already handed to onToken
        var generated = 0
        var generatedTokenIds: [llama_token] = []
        var finishReason = "length"
        let limit = max(1, maxTokens)
        let stops = params.stop.filter { !$0.isEmpty }
        var earlyStop = false            // a stop string or client cancel ended it

        // Emits any text that is safe to stream now — i.e. everything past the
        // longest tail that could still be the start of a stop string.
        func emitSafe() -> Bool {
            guard let onToken = onToken else { emittedCount = output.count; return true }
            let hold = stops.isEmpty ? 0 : partialStopSuffixLength(of: output, stops: stops)
            let safeEnd = output.count - hold
            guard safeEnd > emittedCount else { return true }
            let lo = output.index(output.startIndex, offsetBy: emittedCount)
            let hi = output.index(output.startIndex, offsetBy: safeEnd)
            let slice = String(output[lo..<hi])
            emittedCount = safeEnd
            return onToken(slice)
        }

        while generated < limit {
            let nCtxUsed = promptTokens.count + generated
            if nCtxUsed >= contextSize { finishReason = "context_full"; break }

            let newToken = llama_sampler_sample(sampler, context, -1)
            guard newToken != -1 else {
                finishReason = "sampler_error"
                FileLogger.shared.log("llama_sampler_sample returned -1 (invalid token)")
                break
            }
            if llama_vocab_is_eog(vocab, newToken) {
                FileLogger.shared.log("eog token id=\(newToken) after \(generated) generated tokens")
                finishReason = generated == 0 ? "eog_immediate" : "eog"
                break
            }
            llama_sampler_accept(sampler, newToken)
            generatedTokenIds.append(newToken)

            // Accumulate bytes and decode only complete UTF-8 characters.
            pending.append(contentsOf: pieceBytes(for: newToken))
            let decoded = flushDecodableUTF8(&pending)
            if !decoded.isEmpty { output += decoded }
            generated += 1

            // Complete stop string? Truncate at it, emit the safe remainder, stop.
            if !stops.isEmpty, let r = firstStop(in: output, stops: stops) {
                let cut = output.distance(from: output.startIndex, to: r.lowerBound)
                output = String(output[..<r.lowerBound])
                finishReason = "stop"
                earlyStop = true
                if let onToken = onToken, emittedCount < cut {
                    let lo = output.index(output.startIndex, offsetBy: emittedCount)
                    _ = onToken(String(output[lo...]))
                    emittedCount = cut
                }
                break
            }

            if emitSafe() == false {
                finishReason = "stopped"
                earlyStop = true
                break
            }

            var next = [newToken]
            let stepDecode = next.withUnsafeMutableBufferPointer { buf -> Int32 in
                let batch = llama_batch_get_one(buf.baseAddress, Int32(buf.count))
                return llama_decode(context, batch)
            }
            guard stepDecode == 0 else { throw InferenceError.decode }
        }

        // Flush any trailing bytes (lossy) and any held-back text, unless a stop
        // string / client cancel already finalized the output.
        if !earlyStop {
            if !pending.isEmpty {
                output += String(decoding: pending, as: UTF8.self)
                pending.removeAll()
            }
            if let onToken = onToken, emittedCount < output.count {
                let lo = output.index(output.startIndex, offsetBy: emittedCount)
                _ = onToken(String(output[lo...]))
                emittedCount = output.count
            }
        }

        generationSucceeded = true

        // Update cache tracking. Next call can skip the cached prefix.
        cachedTokenCount = promptTokens.count + generated
        cachedTokenIds = promptTokens + generatedTokenIds

        let preview = output.prefix(200).replacingOccurrences(of: "\n", with: "\\n")
        FileLogger.shared.log("generated \(generated) tokens (finish=\(finishReason), cache now \(cachedTokenCount)) preview='\(preview)'")

        return GenerationResult(text: output,
                                promptTokens: promptTokens.count,
                                completionTokens: generated,
                                finishReason: finishReason)
    }
}
