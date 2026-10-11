import Foundation
import XCTest

/// EngineConfigSpec is the cache-identity rule for the kept-alive inference
/// engine. The server reuses a loaded engine across stop/start and only
/// reloads when the `modelKey` changes. This pins the fix for a real bug:
/// `useMtp` was omitted from the key, so toggling MTP and restarting reused
/// the MTP-off engine and the toggle did nothing — which would have made an
/// on-device MTP A/B measure MTP-off twice.
final class EngineConfigSpecTests: XCTestCase {

    private func key(_ mtp: Bool,
                     ctx: Int = 131072,
                     kv: String = "q8_0",
                     yarn: Bool = true,
                     path: String = "m.gguf") -> String {
        EngineConfigSpec.cacheKey(modelPath: path, contextSize: ctx,
                                  kvCacheType: kv, yarnLongContext: yarn, useMtp: mtp)
    }

    func testTogglingMtpChangesCacheKey() {
        XCTAssertFalse(key(false) == key(true),
                       "toggling MTP must change the cache key, else the toggle is ignored on restart")
    }

    func testSameConfigProducesSameKey() {
        // Unchanged config must still hit the fast path (no reload).
        XCTAssertEqual(key(false), key(false))
    }

    func testEachVaryingSettingChangesKey() {
        let base = key(false)
        XCTAssertFalse(base == key(false, ctx: 4096), "context size must be part of the key")
        XCTAssertFalse(base == key(false, kv: "f16"), "KV cache type must be part of the key")
        XCTAssertFalse(base == key(false, yarn: false), "YaRN must be part of the key")
        XCTAssertFalse(base == key(false, path: "other.gguf"), "model path must be part of the key")
    }

    func testModelPathStaysKeyPrefix() {
        // ServerViewModel does loadedModelKey.hasPrefix(path + "|") to detect
        // "is this file loaded" — the path must remain the key's prefix.
        let k = key(true, path: "/docs/m.gguf")
        XCTAssertTrue(k.hasPrefix("/docs/m.gguf|"))
    }
}
