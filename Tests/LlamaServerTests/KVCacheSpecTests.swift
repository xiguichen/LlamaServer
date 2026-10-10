import XCTest

/// Tests for the pure KV-cache sizing / capability decisions in
/// `KVCacheSpec.swift`.
///
/// Why this exists (two bugs/goals it pins down):
///
/// 1. The old clamp computed `headDim = n_embd / n_head` — 80 for Qwen3-4B,
///    whose GGUF actually stores `attention.key_length = 128`. That
///    underestimated f16 KV cost by 1.6x (92,160 vs the true 147,456 B/token)
///    and let the clamp approve contexts the device cannot back.
///
/// 2. Quantized KV caches (q8_0 / q4_0) trade precision for memory:
///    llama.cpp b9553 only accepts them when `head_dim % 32 == 0`
///    (per-head validation in llama_init_from_model) — otherwise context
///    creation returns NULL. `resolveCacheType` must fall back to f16.
///
/// 3. The trained-window clamp (`min(effective, n_ctx_train)`) must be
///    liftable by the YaRN long-context toggle so Qwen3-4B (n_ctx_train
///    40,960) can reach its YaRN-validated 131,072.
final class KVCacheSpecTests: XCTestCase {

    // MARK: - bytesPerElement

    func testBytesPerElementMatchesQuantBlockSizes() {
        // f16: 2 bytes/element.
        XCTAssertEqual(KVCacheSpec.bytesPerElement(.f16), 2.0, accuracy: 1e-9)
        // q8_0: 32-byte payload + 2-byte fp16 scale per 32 values = 34/32.
        XCTAssertEqual(KVCacheSpec.bytesPerElement(.q8_0), 34.0 / 32.0, accuracy: 1e-9)
        // q4_0: 16-byte payload + 2-byte fp16 scale per 32 values = 18/32.
        XCTAssertEqual(KVCacheSpec.bytesPerElement(.q4_0), 18.0 / 32.0, accuracy: 1e-9)
    }

    // MARK: - kvBytesPerToken

    func testKVBytesPerTokenQwen34BFullPrecision() {
        // Qwen3-4B: 36 layers, head_dim 128 (key_length), 8 KV heads.
        let f16 = KVCacheSpec.kvBytesPerToken(
            nLayer: 36, headDimK: 128, headDimV: 128, nHeadKV: 8, type: .f16)
        XCTAssertEqual(f16, 147_456,
                       "true f16 cost: 2(K+V) x 36 x (128 x 8) x 2B")

        let q8 = KVCacheSpec.kvBytesPerToken(
            nLayer: 36, headDimK: 128, headDimV: 128, nHeadKV: 8, type: .q8_0)
        XCTAssertEqual(q8, 78_336, "q8_0 = 34/32 bytes per element")

        let q4 = KVCacheSpec.kvBytesPerToken(
            nLayer: 36, headDimK: 128, headDimV: 128, nHeadKV: 8, type: .q4_0)
        XCTAssertEqual(q4, 41_472, "q4_0 = 18/32 bytes per element")
    }

    func testKVBytesPerTokenMatchesLegacyFormulaForDerivedHeadDim() {
        // Regression: the pre-KVCacheSpec clamp used
        // 2 * nLayer * (headDim * nHeadKV) * 2 with headDim = n_embd/n_head.
        // For headDim 80 / 36 layers / 8 KV heads that is 92,160 B/token —
        // the new formula must agree when given the same (wrong) head dim,
        // so the fallback path doesn't change any numbers silently.
        let legacy = 2 * 36 * (80 * 8) * 2
        let spec = KVCacheSpec.kvBytesPerToken(
            nLayer: 36, headDimK: 80, headDimV: 80, nHeadKV: 8, type: .f16)
        XCTAssertEqual(spec, legacy)
        XCTAssertEqual(spec, 92_160)
    }

    // MARK: - kvLayerCount (hybrid models)

    func testKVLayerCountHybridFullAttentionInterval() {
        // Qwen3.5 hybrid (observed in device log): n_layer 32,
        // `qwen35.full_attention_interval = 4` — only every 4th layer is
        // full attention and keeps a KV cache; the rest are recurrent
        // (Gated Delta Net) layers with a fixed-size state. llama.cpp's
        // own log showed the KV cache at "8 layers" with the others
        // "filtered", and allocated exactly 738 MiB for 83,968 tokens —
        // i.e. 9,216 B/token, a quarter of the all-layers estimate.
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "4"), 8)
        // Uneven division: attention layers are the ones where
        // (index + 1) % interval == 0, so 30 layers / interval 4 → 7.
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 30, fullAttentionInterval: "4"), 7)
        // Interval larger than the model: at least one KV layer must remain.
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 3, fullAttentionInterval: "4"), 1)
    }

    func testKVLayerCountDefaultsToAllLayersForDenseModels() {
        // Every dense (non-hybrid) model: key absent, unparsable, or <= 1
        // → all layers keep a KV cache (the behavior every previous
        // release relied on).
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: nil), 32)
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: ""), 32)
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "abc"), 32)
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "0"), 32)
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "1"), 32)
        XCTAssertEqual(KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "-4"), 32)
    }

    func testHybridKVCostMatchesLlamaActualAllocation() {
        // End-to-end pin of the device-log bug: with the corrected layer
        // count the q4_0 estimate must equal llama.cpp's real allocation
        // (738 MiB / 83,968 tokens = 9,216 B/token) instead of the
        // 36,864 B/token all-layers over-count that clamped the context
        // to 83,968 despite ~2.89 GB of KV budget.
        let kvLayers = KVCacheSpec.kvLayerCount(nLayer: 32, fullAttentionInterval: "4")
        let perToken = KVCacheSpec.kvBytesPerToken(
            nLayer: kvLayers, headDimK: 256, headDimV: 256, nHeadKV: 4, type: .q4_0)
        XCTAssertEqual(perToken, 9_216, "matches llama's 738 MiB / 83,968 tokens")
        // 131,072 tokens then need 1.13 GB — inside the observed
        // 3,100,752,896 B kvBudget, so no clamp should fire.
        XCTAssertLessThanOrEqual(131_072 * perToken, 3_100_752_896)
    }

    // MARK: - resolveHeadDim

    func testResolveHeadDimPrefersGGUFKeyLength() {
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: "128", fallback: 80), 128)
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: " 64 ", fallback: 80), 64)
    }

    func testResolveHeadDimFallsBackOnMissingOrBogusValue() {
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: nil, fallback: 80), 80)
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: "", fallback: 80), 80)
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: "abc", fallback: 80), 80)
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: "0", fallback: 80), 80)
        XCTAssertEqual(KVCacheSpec.resolveHeadDim(ggufValue: "-32", fallback: 80), 80)
    }

    // MARK: - supports (quantization capability)

    func testQuantizationRequiresHeadDimDivisibleBy32() {
        // Qwen3-4B head_dim 128: full quantization allowed.
        XCTAssertTrue(KVCacheSpec.supports(.q4_0, headDimK: 128, headDimV: 128))
        XCTAssertTrue(KVCacheSpec.supports(.q8_0, headDimK: 128, headDimV: 128))
        // head_dim 80 (n_embd/n_head-derived): rejected by llama.cpp's
        // per-head % 32 check when flash attention is on (required for V).
        XCTAssertFalse(KVCacheSpec.supports(.q4_0, headDimK: 80, headDimV: 80))
        XCTAssertFalse(KVCacheSpec.supports(.q8_0, headDimK: 80, headDimV: 80))
        // Mixed dims: any non-divisible side poisons the K+V pair.
        XCTAssertFalse(KVCacheSpec.supports(.q4_0, headDimK: 128, headDimV: 80))
        XCTAssertTrue(KVCacheSpec.supports(.q4_0, headDimK: 96, headDimV: 128))
        // f16 always supported.
        XCTAssertTrue(KVCacheSpec.supports(.f16, headDimK: 80, headDimV: 80))
    }

    func testResolveCacheTypeFallsBackToFullPrecision() {
        let (type, fellBack) = KVCacheSpec.resolveCacheType(
            selected: .q4_0, headDimK: 80, headDimV: 80)
        XCTAssertEqual(type, .f16)
        XCTAssertTrue(fellBack, "quantized selection on head_dim 80 must fall back")

        let (same, noFallback) = KVCacheSpec.resolveCacheType(
            selected: .q4_0, headDimK: 128, headDimV: 128)
        XCTAssertEqual(same, .q4_0)
        XCTAssertFalse(noFallback)

        let (plain, plainFallback) = KVCacheSpec.resolveCacheType(
            selected: .f16, headDimK: 80, headDimV: 80)
        XCTAssertEqual(plain, .f16)
        XCTAssertFalse(plainFallback, "selecting f16 is never a fallback")
    }

    // MARK: - contextCap

    func testContextCapClampsToTrainedWindowByDefault() {
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 131_072, nCtxTrain: 40_960, longContext: false),
            40_960)
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 20_000, nCtxTrain: 40_960, longContext: false),
            20_000)
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 131_072, nCtxTrain: 0, longContext: false),
            131_072, "n_ctx_train 0 means unknown — no window clamp")
    }

    func testContextCapLiftsTrainedWindowWithLongContextEnabled() {
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 131_072, nCtxTrain: 40_960, longContext: true),
            131_072)
    }

    func testContextCapEnforcesHardBounds() {
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 999_999, nCtxTrain: 40_960, longContext: true),
            131_072, "hard ceiling 131072 applies even with YaRN")
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 100, nCtxTrain: 40_960, longContext: false),
            256, "floor 256 applies")
    }

    // MARK: - YaRN constants

    func testYarnRecipeMatchesQwenValidatedScaling() {
        // Qwen recipe: --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768
        // llama.cpp computes factor = 1 / rope_freq_scale, so freq scale is 1/4.
        XCTAssertEqual(KVCacheSpec.yarnOrigCtx, 32_768)
        XCTAssertEqual(KVCacheSpec.yarnFactor, 4.0, accuracy: 1e-6)
        XCTAssertEqual(KVCacheSpec.yarnRopeFreqScale, 0.25, accuracy: 1e-6)
        XCTAssertEqual(
            KVCacheSpec.yarnRopeFreqScale * KVCacheSpec.yarnFactor, 1.0, accuracy: 1e-6)
    }
}
