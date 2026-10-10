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

    /// Hard ceiling on context size, mirroring the pre-existing clamp.
    static let hardMaxContext = 131_072

    // MARK: YaRN long-context recipe

    /// Qwen's validated YaRN recipe (README): factor 4.0 with
    /// `original_max_position_embeddings = 32768` extends the native 32K
    /// window to 131,072. Applied at runtime via context params — no GGUF
    /// regeneration needed.
    static let yarnFactor: Float = 4.0
    static let yarnOrigCtx: Int = 32_768
    /// llama.cpp computes `factor = 1 / rope_freq_scale`
    /// (llama-context.cpp rope-init), matching common's `--rope-scale 4`
    /// → `rope_freq_scale = 1/4`.
    static let yarnRopeFreqScale: Float = 0.25

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
}
