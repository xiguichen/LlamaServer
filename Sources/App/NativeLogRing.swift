import Foundation

/// Bounded buffer of llama.cpp's own log lines, so a failure can say *why*.
///
/// Why this exists: llama.cpp's internal log is the only place that explains a
/// failed decode (`ggml_metal_synchronize: error: command buffer 0 failed with
/// status 5`, `backend is in error state from a previous command buffer failure`,
/// `llama_decode: failed to decode, ret = -3`). `LlamaInference.init` routes it
/// into `FileLogger` at `.debug` level, which the default log level filters out.
///
/// The result, in the 2026-10-17 incident: the device log showed only
/// `llama_decode failed — recreating context to recover` plus our own guess
/// ("context may be full"), and the actual reason — a GPU command-buffer fault —
/// was invisible without a verbose-level log. Keeping the last few native lines
/// lets `recoverFromDecodeFailure()` dump them at error level (always written, and
/// mirrored to the in-app panel) at the moment they matter.
enum NativeLogRing {
    /// Native lines kept. The observed failure emitted 14 discarded-command-buffer
    /// lines plus the backend-error chain before `llama_decode` returned, so the
    /// tail has to be at least that long to contain the cause.
    static let capacity = 16

    /// Appends `message`, dropping the oldest lines past `capacity`.
    static func append(_ message: String, to lines: inout [String]) {
        lines.append(message)
        if lines.count > capacity {
            lines.removeFirst(lines.count - capacity)
        }
    }

    /// Oldest-first lines, prefixed for the log, ready to hand to `FileLogger`.
    static func dump(_ lines: [String]) -> [String] {
        lines.map { "  [llama] \($0)" }
    }
}
