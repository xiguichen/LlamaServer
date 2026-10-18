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
            262_144, "hard ceiling 262144 applies even with YaRN")
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 100, nCtxTrain: 40_960, longContext: false),
            256, "floor 256 applies")
    }

    // MARK: - ubatch policy

    func testUbatchSizeScalesWithHeadroomAndContext() {
        let slack = 1024 * 1024 * 1024          // 1 GB of KV-budget slack
        let tight = 100 * 1024 * 1024           // under the 200 MB reserve
        // Large contexts (Metal pre-warmed at init) get the 1024 micro-batch:
        // measured ~29% faster prefill (220 vs 170 tok/s on a 5k prompt).
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 131_072, kvSlackBytes: slack), 1024)
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 32_768, kvSlackBytes: slack), 1024,
                       "the pre-warm threshold itself qualifies for 1024")
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 32_767, kvSlackBytes: slack), 512,
                       "below the pre-warm threshold stay on the conservative batch")
        // Tight KV slack keeps the smaller batch regardless of size
        // (activation buffers share the budget with the cache).
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 131_072, kvSlackBytes: tight), 512)
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 131_072, kvSlackBytes: 0), 512)
        // Small contexts always stayed crash-free — unchanged behavior.
        XCTAssertEqual(KVCacheSpec.ubatchSize(contextSize: 8_192, kvSlackBytes: slack), 512)
    }

    func testMetalPrewarmIsGatedOnContextSize() {
        XCTAssertTrue(KVCacheSpec.needsMetalPrewarm(contextSize: 131_072))
        XCTAssertTrue(KVCacheSpec.needsMetalPrewarm(contextSize: 32_768),
                      "the threshold itself is pre-warmed")
        XCTAssertFalse(KVCacheSpec.needsMetalPrewarm(contextSize: 32_767))
        // Regression: the gate once compared the micro-batch, which never
        // exceeds 1024, so the pre-warm never ran for any context.
        let slack = 1024 * 1024 * 1024
        let batch = KVCacheSpec.ubatchSize(contextSize: 131_072, kvSlackBytes: slack)
        XCTAssertFalse(KVCacheSpec.needsMetalPrewarm(contextSize: batch))
    }

    // MARK: - Which clamp decided the context

    func testContextClampNamesTheBindingLimit() {
        // The device's own case (2026-10-18): requested 262,144, hard ceiling
        // 131,072, memory room for hundreds of thousands of tokens. The
        // *ceiling* decided — yet the on-device message said "reduced to fit
        // device memory", blaming memory for a request that was simply above
        // the hard cap (the field default is 262,144, so it printed on every
        // start).
        XCTAssertEqual(
            KVCacheSpec.contextClamp(requested: 262_144, windowCap: 131_072, memoryCap: 400_000),
            .modelWindow(cap: 131_072))
        // Memory binds when it is the smaller cap.
        XCTAssertEqual(
            KVCacheSpec.contextClamp(requested: 262_144, windowCap: 131_072, memoryCap: 83_968),
            .memoryBudget(cap: 83_968))
        // Request small enough that nothing clamps.
        XCTAssertEqual(
            KVCacheSpec.contextClamp(requested: 4_096, windowCap: 131_072, memoryCap: 90_000),
            .notReduced)
        // A window below the request wins even with roomy memory.
        XCTAssertEqual(
            KVCacheSpec.contextClamp(requested: 8_192, windowCap: 4_096, memoryCap: 90_000),
            .modelWindow(cap: 4_096))
        // Ties go to memory: that is the message worth printing, because it is
        // the one the user can act on (KV type, context, model size).
        XCTAssertEqual(
            KVCacheSpec.contextClamp(requested: 9_000, windowCap: 8_000, memoryCap: 8_000),
            .memoryBudget(cap: 8_000))
    }

    // MARK: - YaRN scaling is derived from the target context

    func testYarnFactorMatchesTheValidatedRecipeAt131k() {
        // Qwen's validated recipe: --rope-scaling yarn --rope-scale 4
        // --yarn-orig-ctx 32768 extends 32K to 131,072. The factor IS the
        // extension ratio, so the same recipe must produce it at 131,072.
        XCTAssertEqual(KVCacheSpec.yarnOrigCtx, 32_768)
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 131_072), 4.0, accuracy: 1e-6)
        XCTAssertEqual(KVCacheSpec.yarnRopeFreqScale(forContext: 131_072), 0.25, accuracy: 1e-6)
        // llama.cpp computes factor = 1 / rope_freq_scale.
        XCTAssertEqual(
            KVCacheSpec.yarnRopeFreqScale(forContext: 131_072)
                * KVCacheSpec.yarnFactor(forContext: 131_072), 1.0, accuracy: 1e-6)
    }

    func testYarnFactorTracksTheContextNotAConstant() {
        // The whole point of deriving it: 262,144 needs twice the extension.
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 262_144), 8.0, accuracy: 1e-6)
        XCTAssertEqual(KVCacheSpec.yarnRopeFreqScale(forContext: 262_144), 0.125, accuracy: 1e-6)
        // At or below the original window there is nothing to extend — a fixed
        // factor 4 used to over-stretch these positions for no reason.
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 32_768), 1.0, accuracy: 1e-6)
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 16_000), 1.0, accuracy: 1e-6)
        // Rounded UP: a factor below the ratio would leave positions past the
        // trained window unscaled.
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 32_769), 2.0, accuracy: 1e-6)
        XCTAssertEqual(KVCacheSpec.yarnFactor(forContext: 100_000), 4.0, accuracy: 1e-6)
    }

    func testHardCeilingAllows256kWithYarn() {
        // Raised from 131,072 so a 262,144 request can actually take effect
        // (32768 × 8). Without long context the trained window still caps it.
        XCTAssertEqual(KVCacheSpec.hardMaxContext, 262_144)
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 262_144, nCtxTrain: 32_768, longContext: true),
            262_144)
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 262_144, nCtxTrain: 32_768, longContext: false),
            32_768)
        XCTAssertEqual(
            KVCacheSpec.contextCap(requested: 524_288, nCtxTrain: 32_768, longContext: true),
            262_144)
    }

    // MARK: - Every KV cache the engine allocates counts against the budget

    func testKvCacheCountFollowsMtp() {
        // The MTP draft context is a second context on the same model with the
        // same n_ctx (setupDraftContext), and llama.cpp commits its cache up
        // front. Counting one approved contexts that cannot fit.
        XCTAssertEqual(KVCacheSpec.kvCacheCount(useMtp: false), 1)
        XCTAssertEqual(KVCacheSpec.kvCacheCount(useMtp: true), 2)
    }

    func testMaxContextByMemoryCountsEveryCache() {
        // The device's post-fix numbers: budget 8,721,686,208B − 2.78GB model −
        // 320MB compute = 5,606,141,888B of KV budget at 17,408 B/token.
        let kvBudget = 5_606_141_888
        let perToken = 17_408
        // One cache: 262,144 fits comfortably (321,792 possible).
        XCTAssertEqual(
            KVCacheSpec.maxContextByMemory(kvBudgetBytes: kvBudget, kvBytesPerToken: perToken, cacheCount: 1),
            321_792)
        XCTAssertGreaterThanOrEqual(321_792, 262_144)
        // Two caches (MTP): 262,144 cannot fit — 160,768 is the ceiling, so the
        // request must be clamped and reported as memory-limited.
        XCTAssertEqual(
            KVCacheSpec.maxContextByMemory(kvBudgetBytes: kvBudget, kvBytesPerToken: perToken, cacheCount: 2),
            160_768)
        XCTAssertLessThan(160_768, 262_144)
        // Alignment and degenerate inputs.
        XCTAssertEqual(
            KVCacheSpec.maxContextByMemory(kvBudgetBytes: 300, kvBytesPerToken: 1, cacheCount: 1), 256)
        XCTAssertEqual(
            KVCacheSpec.maxContextByMemory(kvBudgetBytes: 100, kvBytesPerToken: 1, cacheCount: 1), 0)
    }
}
