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
/// All mutable state is guarded by the serial `inferenceQueue` in
/// `LlamaHTTPServer`, so cross-thread access is safe despite the non-Sendable
/// C pointers (OpaquePointer) this class holds.
final class LlamaInference: @unchecked Sendable {

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

    /// Whether to use Multi-Token Prediction (MTP). When enabled the model
    /// predicts multiple future tokens per forward pass, which can increase
    /// throughput on supported models.
    private let useMtp: Bool

    /// How many MTP heads to use (n_outputs_max). Only meaningful when useMtp is
    /// true. 0 means "let llama.cpp pick" (defaults to the model's MTP head count).
    private let mtpHeads: Int

    /// Computed number of MTP output slots (main head + MTP heads) derived from
    /// the model's GGUF metadata or, as a fallback, the user-provided mtpHeads.
    /// Always 1 when useMtp is false.
    private let mtpOutputCount: Int

    /// Whether this model's KV cache supports partial suffix removal
    /// (`llama_memory_seq_rm`). Hybrid attention/SSM and recurrent models keep a
    /// recurrent state that cannot be partially trimmed, so seq_rm always fails
    /// for them — we must clear the whole cache instead of attempting (and
    /// logging a warning for) a removal that can never succeed.
    private var supportsPartialCacheReuse = true

    /// The effective context size actually used (may be smaller than requested
    /// after clamping to the device's memory budget and the model's train window).
    let contextSize: Int

    /// Logical max tokens per llama_decode call. A single decode must NOT exceed
    /// this or llama.cpp aborts (ggml_abort) — the prompt is decoded in chunks.
    let batchSize: Int

    let modelName: String

    // MARK: - Lifecycle

    init(modelPath: String, contextSize requestedContext: Int = 32768, threads: Int32 = 0, useMtp: Bool = false, mtpHeads: Int = 0) throws {
        self.useMtp = useMtp
        self.mtpHeads = mtpHeads
        // Backend init is idempotent across instances within a process.
        llama_backend_init()

        // Route llama.cpp's own internal log (including the real reason a
        // `llama_decode` fails) into our file log. Without this we only see the
        // non-zero return code, never the native explanation. The closure
        // captures nothing (FileLogger.shared is a global), so it converts to a
        // C function pointer. Routed at debug level since model load is verbose.
        llama_log_set({ _, text, _ in
            guard let text else { return }
            let msg = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !msg.isEmpty else { return }
            FileLogger.shared.debug("[llama] \(msg)")
        }, nil)

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
        FileLogger.shared.info("loading model '\(self.modelName)' (requested ctx \(requestedContext), budget \(budget / (1024 * 1024)) MB)")

        guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
            FileLogger.shared.error("model load returned NULL for '\(self.modelName)'")
            throw InferenceError.modelLoad(modelPath)
        }
        FileLogger.shared.info("model loaded OK: '\(self.modelName)'")
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
        let kvTokenSize = ByteCountFormatter.string(fromByteCount: Int64(kvBytesPerToken), countStyle: .memory)
        let kvBudgetStr = ByteCountFormatter.string(fromByteCount: Int64(kvBudget), countStyle: .memory)
        if kvBudget < kvBytesPerToken * 256 {
            llama_model_free(loadedModel)
            throw InferenceError.insufficientMemory(
                "Not enough memory for \(requestedContext)-token context (KV cache needs ~\(kvTokenSize)/token, only \(kvBudgetStr) available). Close other apps or reduce context size.")
        }
        let maxCtxByMemory = (kvBudget / kvBytesPerToken / 256) * 256   // round down to 256
        if effective > maxCtxByMemory {
            FileLogger.shared.warn("requested ctx \(effective) exceeds memory budget (\(kvBudgetStr), \(kvTokenSize)/token); reducing to \(maxCtxByMemory)")
        }
        effective = min(effective, maxCtxByMemory)
        self.contextSize = effective

        // Physical micro-batch (n_ubatch) drives prefill GPU parallelism.
        // Using 512 everywhere: 1024 can crash with a null-pointer dereference
        // inside ggml_metal_buffer_is_shared when the Metal backend hasn't
        // finished its async shader-set initialization, which happens regularly
        // on fresh app launches with large contexts (>= 32768). 512 keeps the
        // compute graphs small enough that sched_reserve completes before Metal
        // readiness becomes a problem, at a modest TTFT cost (~170 vs ~220 tok/s
        // on A19 Pro for a 5k-token prompt — ~5 s difference).
        let baseUBatch = 512
        let fastUBatch = 512   // was 1024, but see crash note above
        let kvSlack = kvBudget - kvBytesPerToken * effective
        let extraUBatchReserve = 200 * 1024 * 1024
        let fastPrefill = kvSlack > extraUBatchReserve
        self.batchSize = min(effective, fastPrefill ? fastUBatch : baseUBatch)

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(effective)
        ctxParams.n_batch = UInt32(batchSize)
        // n_ubatch is the physical sub-batch llama.cpp actually decodes on the
        // GPU; match it to our logical batch so a full chunk is prefilled in one
        // pass. n_batch must be >= n_ubatch, which holds since both equal batchSize.
        ctxParams.n_ubatch = UInt32(batchSize)
        let hwThreads = Int32(ProcessInfo.processInfo.activeProcessorCount)
        let useThreads = threads > 0 ? threads : max(1, hwThreads)
        ctxParams.n_threads = useThreads
        ctxParams.n_threads_batch = useThreads
        self.threadCount = useThreads
        // Sliding-window-attention (SWA) models otherwise keep only a partial KV
        // cache, which makes `llama_memory_seq_rm` fail on any prefix that reaches
        // before the window — defeating all prefix reuse. Keeping the full SWA
        // cache lets seq_rm trim a divergent suffix so the shared prefix is reused.
        // The cost is the full per-token KV our memory budget already assumes.
        ctxParams.swa_full = true

        // Determine the number of MTP output slots. When mtpHeads is explicitly
        // set (non-zero), honour that. Otherwise query the model's GGUF metadata
        // for nextn_predict_layers (the number of MTP heads) and add 1 for the
        // main head. Falls back to 3 when the metadata is unavailable.
        if useMtp && mtpHeads > 0 {
            mtpOutputCount = mtpHeads
        } else if useMtp {
            var layers = 0
            var found = false
            let metaCount = Int(llama_model_meta_count(loadedModel))
            for i in 0..<metaCount {
                var keyBuf = [CChar](repeating: 0, count: 128)
                guard llama_model_meta_key_by_index(loadedModel, Int32(i), &keyBuf, keyBuf.count) > 0 else { continue }
                let key = String(cString: keyBuf)
                guard key.hasSuffix("nextn_predict_layers") || key.hasSuffix("n_predict_layers") else { continue }
                var valBuf = [CChar](repeating: 0, count: 16)
                guard llama_model_meta_val_str_by_index(loadedModel, Int32(i), &valBuf, valBuf.count) > 0 else { continue }
                layers = Int(String(cString: valBuf)) ?? 0
                found = true
                break
            }
            mtpOutputCount = found ? max(1, layers + 1) : 3
        } else {
            mtpOutputCount = 1
        }
        if useMtp {
            ctxParams.ctx_type = LLAMA_CONTEXT_TYPE_MTP
            ctxParams.n_outputs_max = UInt32(mtpOutputCount)
        }

        let nSwa = Int(llama_model_n_swa(loadedModel))
        FileLogger.shared.info("creating context (\(effective) tokens, batch \(batchSize) [fast prefill \(fastPrefill), kv slack \(kvSlack / (1024 * 1024)) MB], swa_window \(nSwa), swa_full on, mtp=\(useMtp)) for '\(self.modelName)'")
        guard let ctx = llama_init_from_model(loadedModel, ctxParams) else {
            FileLogger.shared.error("context creation returned NULL (ctx=\(effective), batch=\(batchSize))")
            let ctxMem = ByteCountFormatter.string(fromByteCount: Int64(kvBytesPerToken * effective), countStyle: .memory)
            let compMem = ByteCountFormatter.string(fromByteCount: Int64(computeReserve), countStyle: .memory)
            let avail = ByteCountFormatter.string(fromByteCount: Int64(kvBudget), countStyle: .memory)
            llama_model_free(loadedModel)
            throw InferenceError.insufficientMemory(
                "Failed to create context: need ~\(ctxMem) KV + ~\(compMem) compute, only \(avail) available. Close other apps or reduce context size.")
        }
        self.context = ctx
        // One-time ground-truth diagnostic for the persistent `seq_rm` failure.
        // swa_full proved ineffective (swa_window 0), so log the cache traits that
        // actually decide whether a partial suffix removal is supported.
        let isRecurrent = llama_model_is_recurrent(loadedModel)
        let isHybrid = llama_model_is_hybrid(loadedModel)
        let canShift = (llama_get_memory(ctx).map { llama_memory_can_shift($0) }) ?? false
        FileLogger.shared.info("context ready (\(effective) tokens) [recurrent \(isRecurrent), hybrid \(isHybrid), can_shift \(canShift)]")
        // Use the authoritative can_shift flag instead of inferring from model
        // type: hybrid/recurrent caches can't partially drop a suffix, and even
        // some pure-attention architectures may have immovable cache state.
        self.supportsPartialCacheReuse = canShift
    }

    /// Frees the old llama_context and creates a fresh one. This is called after
    /// many generations to flush any accumulated GPU/allocator state that could
    /// cause Metal to hang on `llama_sampler_sample`.
    func recreateContext() throws {
        llama_free(context)
        // The old context is now gone; until the new one is created the pointer
        // is dangling. Mark it invalid so a failure below can't be used.
        contextValid = false

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)
        ctxParams.n_batch = UInt32(batchSize)
        ctxParams.n_ubatch = UInt32(batchSize)
        ctxParams.n_threads = threadCount
        ctxParams.n_threads_batch = threadCount
        // Keep the full SWA cache so prefix reuse (seq_rm) works — see init().
        ctxParams.swa_full = true
        if useMtp {
            ctxParams.ctx_type = LLAMA_CONTEXT_TYPE_MTP
            ctxParams.n_outputs_max = UInt32(mtpOutputCount)
        }

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            // `context` still holds the freed pointer; leave `contextValid` false
            // so generate()/unload() never touch it. The engine is now unusable
            // and the next request will surface a clear error.
            FileLogger.shared.error("recreateContext: llama_init_from_model returned NULL — engine unusable")
            throw InferenceError.contextInit
        }
        context = ctx
        contextValid = true
        cachedTokenCount = 0
        cachedTokenIds = []
        FileLogger.shared.info("context recreated (\(contextSize) tokens)")
    }

    /// Recovers after a `llama_decode` returns non-zero. On hybrid attention/SSM
    /// and recurrent models (can_shift == false) a single failed decode leaves
    /// the context's recurrent state permanently wedged, so EVERY later decode
    /// also fails — misleadingly reported as "context may be full" even though
    /// the context is nearly empty. Recreating the context discards that broken
    /// state so the next request can succeed. Best-effort: logs if recreation
    /// itself fails (a genuinely unrecoverable condition).
    private func recoverFromDecodeFailure() {
        FileLogger.shared.error("llama_decode failed — recreating context to recover")
        do {
            try recreateContext()
        } catch {
            FileLogger.shared.error("context recreation after decode failure FAILED: \(error.localizedDescription)")
        }
    }

    /// Tracks whether MTP head outputs have ever been confirmed available
    /// during the current generate() call. When false, the probe skips
    /// out-of-bounds indices to avoid llama.cpp error logs.
    private var mtpHeadsActive = false
    /// Whether the MTP availability probe has been performed (after the first
    /// generation decode, which is the earliest MTP heads could appear).
    private var mtpAvailabilityChecked = false

    /// Tracks whether the native model/context have already been freed so we
    /// never double-free (which would crash). Set by `unload()`.
    private var isUnloaded = false
    /// Whether `context` currently points at a live llama_context. Set false the
    /// instant `recreateContext()` frees the old context, and back to true only
    /// after a new one is successfully created. If re-creation fails, this stays
    /// false so `generate()` fails fast (instead of using a dangling pointer —
    /// use-after-free) and `unload()` skips the free (instead of double-freeing).
    private(set) var contextValid = true

    /// Deterministically frees the llama context and model RIGHT NOW, releasing
    /// the (often multi-GB) Metal/CPU allocation instead of waiting for ARC to
    /// run `deinit`. Call this before discarding the instance so a subsequent
    /// model load starts with a clean memory budget — otherwise the old model is
    /// still resident when the next one loads, the real allocation fails, and
    /// `llama_model_load_from_file` returns NULL (surfaced as a misleading
    /// "architecture may be unsupported" error). Idempotent.
    ///
    /// The caller MUST ensure no generation is in flight on the inference queue
    /// before calling this (the HTTP server drains its queue in `stop()`).
    func unload() {
        guard !isUnloaded else { return }
        isUnloaded = true
        // Only free the context if it's still live: a failed recreateContext()
        // may have already freed it (contextValid == false), and freeing again
        // would be a double-free.
        if contextValid { llama_free(context) }
        llama_model_free(model)
        // Drop the on-disk prompt-state snapshot (if any) so tmp doesn't accrue
        // stale multi-MB state files across model reloads.
        if hasSnapshot {
            try? FileManager.default.removeItem(atPath: snapshotPath)
            hasSnapshot = false
        }
        FileLogger.shared.info("inference unloaded (model + context freed)")
    }

    deinit {
        // Free deterministically if the owner already called `unload()`; the
        // guard makes this a no-op in that case. We intentionally do NOT call
        // llama_backend_free() here: other instances in the same process may
        // still need the backend. It is freed on process exit.
        unload()
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
    /// The exact token IDs currently resident in the KV cache (prompt + the
    /// generated tokens that were actually decoded). Used to compute the longest
    /// shared prefix with the next prompt so only the divergent suffix is
    /// re-decoded. Must mirror the cache precisely — see the end of `generate`.
    private var cachedTokenIds: [llama_token] = []

    // --- Hybrid/recurrent prompt-state snapshot ----------------------------
    // Hybrid attention/SSM and recurrent models cannot trim a KV suffix
    // (`llama_memory_seq_rm` fails), so the prefix-reuse path above is unusable
    // for them and every turn would otherwise re-decode the entire prompt. The
    // only available reuse mechanism is to serialize the sequence state of the
    // prompt to a file and reload it on the next turn when the new prompt is a
    // strict extension of it. This trades a (large) re-decode for a disk write +
    // read, which is far cheaper for the multi-thousand-token agent prompts.
    /// On-disk path for the serialized prompt sequence state. Unique per
    /// instance so concurrent model loads never collide.
    private lazy var snapshotPath: String = {
        let name = "llama_seqstate_\(UInt(bitPattern: ObjectIdentifier(self).hashValue)).bin"
        return (NSTemporaryDirectory() as NSString).appendingPathComponent(name)
    }()
    /// The exact prompt token IDs captured in the snapshot file (the previous
    /// turn's prompt). Cheap to hold in RAM (4 bytes/token) and used to test
    /// whether the next prompt extends it.
    private var snapshotTokens: [llama_token] = []
    /// Whether `snapshotPath` currently holds a valid, loadable snapshot.
    private var hasSnapshot = false

    /// Serializes the current sequence-0 state (which must equal `promptTokens`)
    /// to `snapshotPath` so a later turn can restore this prompt prefix without
    /// re-decoding. Best-effort; a failure just disables reuse next turn.
    private func savePromptSnapshot(_ promptTokens: [llama_token]) {
        let n = promptTokens.count
        let written = promptTokens.withUnsafeBufferPointer { buf in
            llama_state_seq_save_file(context, snapshotPath, 0, buf.baseAddress, n)
        }
        if written > 0 {
            snapshotTokens = promptTokens
            hasSnapshot = true
            FileLogger.shared.debug("KV prompt-state snapshot saved (\(n) tokens, \(written) bytes)")
        } else {
            hasSnapshot = false
            FileLogger.shared.warn("KV prompt-state snapshot save failed")
        }
    }

    /// Restores the prompt-state snapshot when `promptTokens` is a strict
    /// extension of the snapshotted prompt, returning the number of leading
    /// tokens now resident in the cache (so the caller decodes only the suffix).
    /// Returns 0 (after clearing the cache) when the snapshot can't be applied.
    /// Only valid for hybrid/recurrent models, which have no seq_rm trim.
    private func restorePromptSnapshot(for promptTokens: [llama_token],
                                       memory: OpaquePointer) -> Int {
        guard hasSnapshot, !snapshotTokens.isEmpty else {
            llama_memory_clear(memory, true)
            return 0
        }
        // Longest common token prefix of the snapshot and the new prompt.
        let maxLCP = min(snapshotTokens.count, promptTokens.count)
        var lcp = 0
        while lcp < maxLCP && snapshotTokens[lcp] == promptTokens[lcp] { lcp += 1 }
        // The snapshot is one indivisible state blob at position
        // `snapshotTokens.count`; it can only be restored whole. Require the new
        // prompt to contain it entirely AND leave >= 1 token to decode for fresh
        // logits. Anything less can't use the snapshot — re-decode from scratch.
        guard lcp == snapshotTokens.count, snapshotTokens.count < promptTokens.count else {
            llama_memory_clear(memory, true)
            FileLogger.shared.debug("KV snapshot unusable (lcp \(lcp)/\(snapshotTokens.count), prompt \(promptTokens.count)); full decode")
            return 0
        }
        // Start from a clean cache, then load the saved prompt state into seq 0.
        llama_memory_clear(memory, true)
        var loaded = [llama_token](repeating: 0, count: snapshotTokens.count)
        var nOut = 0
        let nread = llama_state_seq_load_file(context, snapshotPath, 0,
                                              &loaded, snapshotTokens.count, &nOut)
        guard nread > 0, nOut == snapshotTokens.count else {
            FileLogger.shared.warn("KV snapshot load failed (nread \(nread), nOut \(nOut)); full decode")
            llama_memory_clear(memory, true)
            hasSnapshot = false
            return 0
        }
        FileLogger.shared.debug("KV snapshot restored (\(snapshotTokens.count) tokens, prompt \(promptTokens.count)); decoding \(promptTokens.count - snapshotTokens.count) new")
        return snapshotTokens.count
    }

    /// Token length of the prompt's "stable prefix" — everything up to (but not
    /// including) the final assistant generation primer the chat template
    /// appends. Excluding the primer from the hybrid/recurrent snapshot is what
    /// makes that snapshot a strict prefix of the next turn's prompt (the primer
    /// is replaced by the real assistant message next turn). Returns the full
    /// length when no primer boundary is found (snapshot then falls back to the
    /// whole prompt). Re-tokenizes the prefix and takes the common-prefix length
    /// with `promptTokens` to stay robust against BPE boundary drift.
    private func stablePrefixLength(prompt: String, promptTokens: [llama_token]) -> Int {
        guard let r = prompt.range(of: "<|im_start|>assistant", options: .backwards) else {
            return promptTokens.count
        }
        let prefixStr = String(prompt[prompt.startIndex..<r.lowerBound])
        guard let prefixTokens = try? tokenize(prefixStr, addBOS: true), !prefixTokens.isEmpty else {
            return promptTokens.count
        }
        let maxN = min(prefixTokens.count, promptTokens.count)
        var n = 0
        while n < maxN && prefixTokens[n] == promptTokens[n] { n += 1 }
        return n
    }

    /// Generates a completion for the given (already chat-formatted) prompt.
    /// `onToken` is called for each decoded piece; return `false` to stop early.
    func generate(prompt: String,
                  maxTokens: Int,
                  params: SamplingParams,
                  onToken: ((String) -> Bool)? = nil) throws -> GenerationResult {

        // A prior recreateContext() may have failed and left no live context.
        // Fail fast with a clear error instead of decoding into a freed pointer.
        guard contextValid else {
            FileLogger.shared.error("generate called with no valid context (recreate previously failed)")
            throw InferenceError.contextInit
        }

        let promptTokens = try tokenize(prompt, addBOS: true)
        guard !promptTokens.isEmpty else { throw InferenceError.tokenize }

        // --- KV cache prefix reuse --------------------------------------------
        // Reuse the largest shared *token* prefix between this prompt and what is
        // already in the KV cache, then re-decode only the divergent suffix. The
        // KV cache is keyed by token position, so token IDs are authoritative —
        // a text-prefix check is both weaker and forces all-or-nothing reuse.
        //
        // For an agent loop (e.g. OpenCode) that resends a growing
        // system+history prefix on every turn, this keeps the long shared prefix
        // (often thousands of tokens) cached and only decodes the new tail —
        // turning a multi-second prompt decode into a near-instant one. Upstream
        // llama-server uses the same common-prefix + seq_rm strategy.
        var reuseLen = 0
        if let memory = llama_get_memory(context) {
            if !supportsPartialCacheReuse {
                // Hybrid/recurrent model: a KV suffix can't be trimmed, so prefix
                // reuse via seq_rm is impossible. Reuse the previous prompt's
                // sequence state from a saved snapshot when the new prompt extends
                // it; otherwise fall back to a full clear + re-decode.
                reuseLen = restorePromptSnapshot(for: promptTokens, memory: memory)
            } else if cachedTokenIds.isEmpty {
                llama_memory_clear(memory, true)
            } else {
                // Longest common prefix of the cached tokens and the new prompt.
                let maxLCP = min(cachedTokenIds.count, promptTokens.count)
                var lcp = 0
                while lcp < maxLCP && cachedTokenIds[lcp] == promptTokens[lcp] { lcp += 1 }
                // Always leave at least one prompt token to decode so the forward
                // pass produces fresh logits to sample the first new token from.
                if lcp >= promptTokens.count { lcp = promptTokens.count - 1 }

                if lcp <= 0 {
                    llama_memory_clear(memory, true)
                    FileLogger.shared.debug("KV cache cleared (no shared prefix; last cached \(cachedTokenIds.count))")
                } else if lcp == cachedTokenIds.count {
                    // The whole cache is a prefix of the new prompt — keep it all.
                    reuseLen = lcp
                    FileLogger.shared.debug("KV cache reused fully (\(reuseLen) tokens, prompt \(promptTokens.count))")
                } else if !supportsPartialCacheReuse {
                    // Hybrid/recurrent model: partial suffix removal is impossible,
                    // so don't attempt seq_rm (it would fail every time). Clear the
                    // whole cache and re-decode the prompt from scratch.
                    llama_memory_clear(memory, true)
                    FileLogger.shared.debug("KV cache cleared (model has no partial-reuse; last cached \(cachedTokenIds.count))")
                } else if llama_memory_seq_rm(memory, 0, llama_pos(lcp), -1) {
                    // Drop the diverging suffix [lcp, end); keep prefix [0, lcp).
                    reuseLen = lcp
                    FileLogger.shared.debug("KV cache reused (\(reuseLen)/\(cachedTokenIds.count) tokens, prompt \(promptTokens.count))")
                } else {
                    // Partial removal unsupported by this cache type — start fresh.
                    // Log the sequence bounds first: seq_pos_min > 0 means the cache
                    // already evicted the prefix we tried to keep (SWA/recurrent),
                    // which is the only documented reason seq_rm returns false here.
                    let posMin = Int(llama_memory_seq_pos_min(memory, 0))
                    let posMax = Int(llama_memory_seq_pos_max(memory, 0))
                    llama_memory_clear(memory, true)
                    FileLogger.shared.warn("KV cache partial-rm failed at lcp \(lcp) [seq_pos_min \(posMin), seq_pos_max \(posMax)]; cleared (last cached \(cachedTokenIds.count))")
                }
            }
        } else {
            FileLogger.shared.warn("llama_get_memory returned nil — KV cache NOT cleared")
        }

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
                if cachedTokenCount > 0 || !cachedTokenIds.isEmpty {
                    FileLogger.shared.error("generation failed — cache tracking invalidated")
                }
                cachedTokenCount = 0
                cachedTokenIds = []
            }
        }

        let genStart = CFAbsoluteTimeGetCurrent()
        // Decode the prompt. Breadcrumb first: a too-long prompt or a native
        // abort here would otherwise kill the process with no trace.
        FileLogger.shared.debug("decoding prompt (\(promptTokens.count) tokens, ctx \(contextSize), batch \(batchSize))")
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

        // Decodes the token range [lo, hi) in n_batch-sized chunks. Throws (after
        // recovery) on a decode failure. Generation always decodes through to
        // `total`, so this is split into two phases only to land a snapshot
        // boundary at `stableLen` — never affecting what the model finally sees.
        func decodeRange(_ lo: Int, _ hi: Int) throws {
            var off = lo
            while off < hi {
                let count = min(batchSize, hi - off)
                let status = tokens.withUnsafeMutableBufferPointer { buf -> Int32 in
                    let chunk = buf.baseAddress!.advanced(by: off)
                    var logits = [Int8](repeating: 0, count: count)
                    logits[count - 1] = 1
                    return logits.withUnsafeMutableBufferPointer { logitsBuf in
                        let batch = llama_batch(
                            n_tokens: Int32(count),
                            token: chunk,
                            embd: nil,
                            pos: nil,
                            n_seq_id: nil,
                            seq_id: nil,
                            logits: logitsBuf.baseAddress
                        )
                        return llama_decode(context, batch)
                    }
                }
                guard status == 0 else {
                    recoverFromDecodeFailure()
                    throw InferenceError.decode
                }
                off += count
            }
        }

        if !supportsPartialCacheReuse {
            // Hybrid/recurrent model: the snapshot can only be restored whole, so
            // it must EXCLUDE the trailing generation primer (the assistant turn
            // the chat template appends, e.g. "<|im_start|>assistant\n<think>…").
            // The next turn replaces that primer with the real assistant message,
            // so a snapshot that includes it is never a prefix of the next prompt
            // (its last few tokens always diverge) and is thus useless. Snapshot
            // the stable prefix instead, BEFORE generation advances the state.
            let stableLen = max(reuseLen, min(total, stablePrefixLength(prompt: prompt, promptTokens: promptTokens)))
            try decodeRange(reuseLen, stableLen)
            if stableLen > 0 && stableLen < total {
                savePromptSnapshot(Array(promptTokens[0..<stableLen]))
            }
            try decodeRange(stableLen, total)
            if stableLen >= total {
                // No primer boundary found — fall back to snapshotting the whole
                // prompt (behaviour as before; may simply be unusable next turn).
                savePromptSnapshot(promptTokens)
            }
        } else {
            try decodeRange(reuseLen, total)
        }

        // Diagnostic: how does the formatted prompt end? It should include the
        // assistant generation header (e.g. "<|im_start|>assistant").
        let promptTail = String(prompt.suffix(180)).replacingOccurrences(of: "\n", with: "\\n")
        FileLogger.shared.verbose("prompt tail: …\(promptTail)")

        let prefillEnd = CFAbsoluteTimeGetCurrent()
        // Reset MTP tracking for a fresh generation (closed over by the
        // while-loop below, which has closure capture for its mutable state).
        mtpHeadsActive = false
        mtpAvailabilityChecked = false
        var genTokens = 0  // separate counter for generation timing
        var output = ""
        var pending: [UInt8] = []        // incomplete trailing UTF-8 bytes
        var emittedCount = 0             // chars already handed to onToken
        var generated = 0
        var decodedGenerated = 0         // generated tokens actually fed to the KV cache
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

        let mtpCount = mtpOutputCount
        let firstTokenTimerStart = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        while generated < limit {
            let nCtxUsed = promptTokens.count + generated
            if nCtxUsed >= contextSize { finishReason = "context_full"; break }

            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
            genTokens += 1

            // Probe available output rows. After prompt decode only the main
            // head produces output (n_outputs=1); MTP heads — if the model
            // supports them — become available after the first single-token
            // generation decode. To avoid llama.cpp error logs for
            // out-of-bounds indices we:
            //   1. Skip probing MTP heads on the first iteration (always 1).
            //   2. On the second iteration, check -2 once and cache result.
            //   3. After that, either all slots or just slot -1 — no repeat
            //      probes that would hit the native error path.
            var availableOutputs = MtpProbe.computeOutputs(
                mtpCount: mtpCount,
                genTokens: genTokens,
                mtpAvailabilityChecked: &mtpAvailabilityChecked,
                mtpHeadsActive: &mtpHeadsActive,
                checkLogits: { llama_get_logits_ith(context, $0) != nil }
            )

            // Sample from available output slots using reverse indices.
            // Slot 0 (main head) → -availableOutputs, slot 1 → -(availableOutputs-1), etc.
            var batchTokens: [llama_token] = []
            for i in 0..<min(mtpCount, availableOutputs) {
                if generated + batchTokens.count >= limit { break }
                if nCtxUsed + batchTokens.count >= contextSize { break }

                let logitsIdx = Int32(i - availableOutputs)
                guard llama_get_logits_ith(context, logitsIdx) != nil else { break }

                let token = llama_sampler_sample(sampler, context, logitsIdx)
                guard token != -1 else {
                    if batchTokens.isEmpty {
                        finishReason = "sampler_error"
                        FileLogger.shared.error("llama_sampler_sample returned -1 (invalid token)")
                    }
                    break
                }
                if llama_vocab_is_eog(vocab, token) {
                    if batchTokens.isEmpty {
                        FileLogger.shared.debug("eog token id=\(token) after \(generated) generated tokens")
                        finishReason = generated == 0 ? "eog_immediate" : "eog"
                    }
                    break
                }
                llama_sampler_accept(sampler, token)
                batchTokens.append(token)
            }

            if batchTokens.isEmpty { break }

            for token in batchTokens {
                generatedTokenIds.append(token)

                pending.append(contentsOf: pieceBytes(for: token))
                let decoded = flushDecodableUTF8(&pending)
                if !decoded.isEmpty { output += decoded }
                generated += 1

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
            }
            if earlyStop { break }

            // Decode all batch tokens in a single forward pass. Must set logits
            // explicitly (not via llama_batch_get_one) for MTP context type.
            var next = batchTokens
            let stepDecode = next.withUnsafeMutableBufferPointer { buf -> Int32 in
                let count = buf.count
                var logits = [Int8](repeating: 0, count: count)
                if count > 0 { logits[count - 1] = 1 }
                return logits.withUnsafeMutableBufferPointer { logitsBuf in
                    let batch = llama_batch(
                        n_tokens: Int32(count),
                        token: buf.baseAddress,
                        embd: nil,
                        pos: nil,
                        n_seq_id: nil,
                        seq_id: nil,
                        logits: logitsBuf.baseAddress
                    )
                    return llama_decode(context, batch)
                }
            }
            guard stepDecode == 0 else {
                recoverFromDecodeFailure()
                throw InferenceError.decode
            }
            // All decoded tokens are now resident in the KV cache.
            decodedGenerated = generated
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

        // Update cache tracking so the next call can reuse the shared prefix.
        // Only tokens actually decoded into the KV cache are recorded —
        // `decodedGenerated` excludes any final token from a stop-string or
        // client-cancel break that was sampled but never fed back to the cache.
        cachedTokenCount = promptTokens.count + decodedGenerated
        cachedTokenIds = promptTokens + generatedTokenIds.prefix(decodedGenerated)

        let genEnd = CFAbsoluteTimeGetCurrent()
        let prefillSecs = prefillEnd - genStart
        let genSecs = genEnd - (firstTokenTime ?? genEnd)
        let prefillTokPerSec = prefillSecs > 0 ? Double(promptTokens.count) / prefillSecs : 0
        let genTokPerSec = genSecs > 0 ? Double(generated) / genSecs : 0
        let tpms = genTokPerSec * 60
        let preview = output.prefix(200).replacingOccurrences(of: "\n", with: "\\n")
        FileLogger.shared.info("generated \(generated) tokens (prefill=\(Int(prefillSecs))s \(Int(prefillTokPerSec))t/s, gen=\(Int(genSecs))s \(Int(genTokPerSec))t/s, \(Int(tpms))t/min, finish=\(finishReason), cache now \(cachedTokenCount)) preview='\(preview)'")

        return GenerationResult(text: output,
                                promptTokens: promptTokens.count,
                                completionTokens: generated,
                                finishReason: finishReason)
    }
}
