import Foundation

/// KV-cache element type — mirrors the ggml types accepted by llama.cpp's
/// `llama_context_params.type_k` / `type_v` (b9553). Raw values match the
/// ggml type names for logging and the UI picker.
enum KVCacheType: String, CaseIterable {
    case f16
    case q8_0
    case q4_0
}

/// Pure decisions for KV-cache sizing, quantization capability, and the
/// context-window clamp. No llama.cpp types here — everything is testable
/// in the local harness and on CI's simulator (`KVCacheSpecTests`).
///
/// Background: the pre-KVCacheSpec clamp derived head dim as
/// `n_embd / n_head`, which is wrong for models that store an explicit
/// `attention.key_length` in the GGUF (Qwen3-4B: 128, not 80) and therefore
/// underestimated f16 KV cost by 1.6x. Quantized KV (q8_0/q4_0) trades a
/// small accuracy loss for 1.8x–3.6x smaller caches but is only accepted by
/// llama.cpp when `head_dim % 32 == 0`.
enum KVCacheSpec {

    /// Hard ceiling on context size. Raised from 131,072 to 262,144 (32768 × 8)
    /// so a 262,144 request can take effect; the memory clamp is what decides
    /// whether a context this size can actually be backed (see
    /// `chargedKvBytesPerToken` for what every context on the model costs).
    static let hardMaxContext = 262_144

    // MARK: YaRN long-context recipe

    /// Qwen's YaRN base window: `original_max_position_embeddings = 32768`.
    /// Applied at runtime via context params — no GGUF regeneration needed.
    static let yarnOrigCtx: Int = 32_768

    /// YaRN extension factor for a target context.
    ///
    /// The recipe's factor *is* the extension ratio (target / original), so it
    /// has to be derived from the context actually being built rather than
    /// pinned to one value. A fixed factor 4 was right at 131,072 (Qwen's
    /// validated --rope-scale 4 --yarn-orig-ctx 32768) but wrong either side of
    /// it: at 65,536 it stretched positions past the trained window for nothing,
    /// and at 262,144 it left everything beyond 131,072 unscaled.
    ///
    /// Rounded **up**: a factor below the true ratio leaves the far positions
    /// under-scaled, which is the failure mode that matters.
    static func yarnFactor(forContext context: Int) -> Float {
        guard context > yarnOrigCtx else { return 1.0 }
        return Float((context + yarnOrigCtx - 1) / yarnOrigCtx)
    }

    /// llama.cpp computes `factor = 1 / rope_freq_scale`
    /// (llama-context.cpp rope-init), matching common's `--rope-scale N`.
    static func yarnRopeFreqScale(forContext context: Int) -> Float {
        1.0 / yarnFactor(forContext: context)
    }

    /// KV layers the MTP draft context keeps.
    ///
    /// Every context on this model holds its own cache, but they are not the
    /// same size. The draft context is created with
    /// `ctx_type = LLAMA_CONTEXT_TYPE_MTP` (`LlamaInference.setupDraftContext`),
    /// so llama.cpp builds only the model's nextn blocks and commits KV for
    /// those alone — the main layers' attention is never run there (the MTP
    /// block takes the target's hidden state as input).
    ///
    /// This replaced a plain "two caches" count, which charged the draft a
    /// *second full cache*. The device log disproved that (2026-10-18, build
    /// `698dbbf`): at 80,128 tokens the target context allocated 1,330.25 MiB
    /// (80,128 × 17,408 B/token = 8 of 32 layers) and the draft context
    /// 166.28 MiB (80,128 × 2,176 B/token = exactly **one** layer, this model's
    /// `nextn_predict_layers = 1`). The 8x over-charge clamped a viable
    /// 142,336-token context down to 80,128.
    ///
    /// The floor of one layer: `nextnLayers` comes from
    /// `llama_model_n_layer_nextn`, and a model reporting none cannot be drafted
    /// at all (`load_mtp` finds no `nextn.eh_proj` and the MTP graph aborts), so
    /// a zero here is a mis-report and must never charge the draft cache nothing.
    static func draftKVLayers(nextnLayers: Int, useMtp: Bool) -> Int {
        guard useMtp else { return 0 }
        return max(1, nextnLayers)
    }

    /// Total KV bytes per token the budget must back across **every** context on
    /// the model: the target's own cache plus the draft context's nextn-layer
    /// cache (`draftKVLayers`). Naming this separately is what keeps the charge
    /// honest: budgeting one cache approved contexts the device could not back,
    /// and budgeting two charged the draft eight times what it stores.
    static func chargedKvBytesPerToken(targetKvBytesPerToken: Int,
                                       draftKvBytesPerToken: Int) -> Int {
        max(1, targetKvBytesPerToken) + max(0, draftKvBytesPerToken)
    }

    /// Largest 256-aligned context the KV budget can back at
    /// `chargedBytesPerToken` — the sum across every context the engine
    /// allocates, not one cache's cost and not a whole number of caches.
    static func maxContextByMemory(kvBudgetBytes: Int, chargedBytesPerToken: Int) -> Int {
        (kvBudgetBytes / max(1, chargedBytesPerToken) / 256) * 256
    }

    // MARK: Sizing

    /// Bytes per stored element: f16 = 2; q8_0 = (32 payload + 2 scale)/32;
    /// q4_0 = (16 payload + 2 scale)/32.
    static func bytesPerElement(_ type: KVCacheType) -> Double {
        switch type {
        case .f16:  return 2.0
        case .q8_0: return 34.0 / 32.0
        case .q4_0: return 18.0 / 32.0
        }
    }

    /// K + V cache bytes per token across all layers, for a cache of the
    /// given type. `headDimK`/`headDimV` come from the GGUF's
    /// `attention.key_length` / `attention.value_length` (see
    /// `resolveHeadDim`), NOT from `n_embd / n_head`.
    static func kvBytesPerToken(nLayer: Int, headDimK: Int, headDimV: Int,
                                nHeadKV: Int, type: KVCacheType) -> Int {
        // Elements per token: every layer stores nHeadKV heads of K plus the
        // same for V. Quantized types lay out blocks of 32 elements, so the
        // byte count is exact when supports() admitted the dimensions.
        let elements = nLayer * nHeadKV * (headDimK + headDimV)
        switch type {
        case .f16:  return elements * 2
        case .q8_0: return elements * 34 / 32
        case .q4_0: return elements * 18 / 32
        }
    }

    // MARK: Micro-batch (ubatch) policy

    /// Contexts at/above this size get the 1024 micro-batch (measured ~29%
    /// faster prefill — 220 vs 170 tok/s on a ~5k prompt) — but only via the
    /// Metal pre-warm path in `LlamaInference`, because a >=32k context built
    /// on a cold Metal backend crashes (async shader-set init still running).
    static let prewarmUbatchContextSize = 32_768

    /// KV-budget headroom required to double the micro-batch: activation
    /// buffers draw from the same budget as the cache.
    static let ubatchSlackBytes = 200 * 1024 * 1024

    /// The micro-batch (`n_ubatch`) for a context of `contextSize` with
    /// `kvSlackBytes` of KV-budget headroom. Callers then clamp to the
    /// context size itself (a batch can't exceed the context).
    ///
    /// - Tight slack (<= `ubatchSlackBytes`): the conservative 512, since the
    ///   larger batch's activation buffers would contend with the cache.
    /// - Large, pre-warmed context: 1024 for faster prefill.
    /// - Smaller contexts stayed crash-free at 512, so behavior is unchanged.
    static func ubatchSize(contextSize: Int, kvSlackBytes: Int) -> Int {
        guard kvSlackBytes > ubatchSlackBytes else { return 512 }
        return contextSize >= prewarmUbatchContextSize ? 1024 : 512
    }

    /// Whether a context of `contextSize` tokens needs the Metal pre-warm
    /// before it is built. Gated on the CONTEXT size, never the micro-batch:
    /// `ubatchSize` only ever returns 512 or 1024, so a batch compared with
    /// this 32k threshold can never pass and the pre-warm never runs.
    static func needsMetalPrewarm(contextSize: Int) -> Bool {
        contextSize >= prewarmUbatchContextSize
    }

    // MARK: Hybrid-model layer counts

    /// The number of layers that actually keep a KV cache.
    ///
    /// Dense models: every layer is full attention → `nLayer`.
    ///
    /// Hybrid models (e.g. Qwen3.5 SSM+attention): the GGUF stores
    /// `<arch>.full_attention_interval` — only every `interval`-th layer
    /// (index+1 divisible by interval) is full attention; the recurrent
    /// layers hold a fixed-size state that does not grow with context.
    /// Counting all `nLayer` over-estimates KV cost by `interval`-x and
    /// clamps the context unnecessarily: observed on a 32-layer /
    /// interval-4 model, 36,864 B/token estimated vs llama.cpp's actual
    /// 9,216 B/token (8 KV layers — 738 MiB / 83,968 tokens in its log),
    /// which cut a viable 131,072 context down to 83,968.
    ///
    /// `fullAttentionInterval` is the raw GGUF value (as returned by
    /// `ggufMetaValue`); absent, unparsable, or <= 1 means dense.
    static func kvLayerCount(nLayer: Int, fullAttentionInterval: String?) -> Int {
        guard let raw = fullAttentionInterval?.trimmingCharacters(in: .whitespacesAndNewlines),
              let interval = Int(raw), interval > 1 else {
            return max(0, nLayer)
        }
        // Attention layers are those with (index + 1) % interval == 0, so
        // the count is floor(nLayer / interval); keep at least one KV layer.
        return max(1, min(nLayer, nLayer / interval))
    }

    // MARK: Head dimension resolution

    /// Prefer the GGUF's explicit head length (`…attention.key_length` /
    /// `…attention.value_length`); fall back to the derived
    /// `n_embd / n_head` when the key is absent, unparsable, or non-positive.
    static func resolveHeadDim(ggufValue: String?, fallback: Int) -> Int {
        guard let raw = ggufValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              let value = Int(raw), value > 0 else {
            return fallback
        }
        return value
    }

    // MARK: Quantization capability

    /// llama.cpp (b9553) rejects a quantized K or V cache when any layer's
    /// per-head dimension is not a multiple of the 32-element quant block
    /// (llama_init_from_model validation) — context creation would return
    /// NULL. f16 has no such constraint.
    static func supports(_ type: KVCacheType, headDimK: Int, headDimV: Int) -> Bool {
        switch type {
        case .f16:
            return true
        case .q8_0, .q4_0:
            return headDimK % 32 == 0 && headDimV % 32 == 0
        }
    }

    /// The cache type to actually create: the user's selection if the model
    /// supports it, otherwise f16 with `fellBack == true` (caller logs).
    static func resolveCacheType(selected: KVCacheType,
                                 headDimK: Int, headDimV: Int) -> (type: KVCacheType, fellBack: Bool) {
        if supports(selected, headDimK: headDimK, headDimV: headDimV) {
            return (selected, false)
        }
        return (.f16, true)
    }

    // MARK: Context window

    /// The context size after clamping to the model's trained window —
    /// unless `longContext` (YaRN) is enabled, in which case the trained
    /// window is lifted (the hard 131072 ceiling and the 256 floor still
    /// apply). `nCtxTrain <= 0` means unknown: no window clamp.
    static func contextCap(requested: Int, nCtxTrain: Int, longContext: Bool) -> Int {
        var cap = min(max(256, requested), hardMaxContext)
        if nCtxTrain > 0 && !longContext {
            cap = min(cap, nCtxTrain)
        }
        return cap
    }

    /// Which cap decided the effective context, when it came out below the
    /// request. The distinction is operator-facing: a *window* cap is a
    /// property of the model (raise it with long context or a bigger-window
    /// model), a *memory* cap is a property of this device and the KV/model
    /// footprint (shrink the KV type, context, or model).
    ///
    /// Why this exists: the app used to print "Context reduced to N tokens to
    /// fit device memory" for *any* reduction. With the default request at
    /// 262,144 and `hardMaxContext` at 131,072 that message printed on every
    /// start while the KV cache was in fact far from its limit — blaming device
    /// memory for a hard cap.
    enum ContextClamp: Equatable {
        case notReduced
        case modelWindow(cap: Int)
        case memoryBudget(cap: Int)
    }

    /// Classifies the binding clamp: `windowCap` is `contextCap(...)` (trained
    /// window or the hard ceiling), `memoryCap` is what the KV budget can hold.
    /// Ties go to `.memoryBudget`: it is the actionable one.
    static func contextClamp(requested: Int, windowCap: Int, memoryCap: Int) -> ContextClamp {
        let effective = min(requested, windowCap, memoryCap)
        if effective >= requested { return .notReduced }
        return memoryCap <= windowCap ? .memoryBudget(cap: effective) : .modelWindow(cap: effective)
    }
}
