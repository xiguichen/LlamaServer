import Foundation

/// Pure cache-identity rules for the kept-alive inference engine.
///
/// The server reuses a loaded engine across stop/start for a fast restart and
/// only reloads when the `modelKey` changes. That key must capture EVERY
/// `LlamaInference` constructor input the UI can vary — omitting one means
/// toggling that setting is silently ignored on restart (the old engine is
/// reused and the new setting never takes effect).
///
/// This happened with MTP: `useMtp` was left out, so flipping the MTP toggle
/// and restarting reused the MTP-off engine — an on-device MTP A/B would then
/// measure MTP-off twice and conclude (wrongly) that MTP gives no speedup.
enum EngineConfigSpec {

    /// Cache key identifying a loaded engine. The model path stays FIRST so a
    /// `hasPrefix(path + "|")` check (used to answer "is this file loaded?")
    /// keeps working.
    static func cacheKey(modelPath: String,
                         contextSize: Int,
                         kvCacheType: String,
                         yarnLongContext: Bool,
                         useMtp: Bool) -> String {
        "\(modelPath)|\(contextSize)|\(kvCacheType)|\(yarnLongContext)|\(useMtp)"
    }
}
