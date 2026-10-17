import Foundation

/// Which on-disk file a log line belongs to.
///
/// The log is split by subsystem so the readable file stays readable and the
/// multi-kilobyte payload dumps can never crowd it out (or reach the UI):
///
///   - `server`    HTTP serving + app lifecycle: connections, heartbeat, memory
///                 warnings, listener restarts, background keep-alive, UI events.
///   - `inference` Engine internals: model load, context create/recreate, KV
///                 cache decisions, MTP/plain stats, per-generation timing.
///   - `payloads`  The bulky dumps captured at `.verbose`: raw request bodies,
///                 full prompts, full responses, streamed token text, tool args.
enum LogCategory: String, CaseIterable {
    case server
    case inference
    case payloads

    /// File name in the app's Documents directory. `server` keeps the historical
    /// `llamaserver.log` so the documented Files-app path still resolves.
    var fileName: String {
        switch self {
        case .server:    return "llamaserver.log"
        case .inference: return "llamaserver-inference.log"
        case .payloads:  return "llamaserver-payloads.log"
        }
    }
}

/// Pure routing policy: which file a line lands in.
///
/// Routing keys off the source file (`#file`) so every call site is classified
/// without threading a category argument through ~150 of them, and off the level
/// for the payload channel — `.verbose` *is* the payload channel by definition
/// (`LogLevel` documents it as "full payloads"), so it always gets its own file.
enum LogRouter {
    static func category(level: LogLevel, sourceFile: String) -> LogCategory {
        if level == .verbose { return .payloads }
        if sourceFile.hasSuffix("LlamaInference.swift") { return .inference }
        return .server
    }
}
