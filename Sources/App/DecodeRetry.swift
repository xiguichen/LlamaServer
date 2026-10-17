import Foundation

/// Whether a failed generation may be retried transparently.
///
/// Why this exists: `llama_decode` returns non-zero when the GPU discards our
/// command buffers — observed on device (2026-10-17 20:39:13, prompt prefill of
/// request #2) as `kIOGPUCommandBufferCallbackErrorInnocentVictim`, "victim of
/// GPU error/recovery": the GPU was reset by something outside this process and
/// our command buffers were collateral damage. llama.cpp then reports
/// `ggml_metal_graph_compute: backend is in error state … recreate the backend to
/// recover` and `llama_decode: failed to decode, ret = -3`.
///
/// The decode sites hand that to `LlamaInference.recoverFromDecodeFailure()`,
/// which rebuilds the llama_context (and with it the Metal backend) — the engine
/// is healthy again ~0.7 s later. The request, however, used to die with a 500
/// even when the failure happened during prefill and not a byte had reached the
/// client, so a transient fault outside our control cost the user a whole turn.
///
/// The fault itself is not preventable; hiding it is. Retrying is transparent
/// only while nothing has been delivered — once output is on the wire it cannot
/// be taken back — and retrying more than once is pointless: a second failure
/// means the backend is not recovering, so surface it instead of looping.
enum DecodeRetry {
    /// Retries allowed per generation. The observed fault cleared within one
    /// context rebuild (the client's own retry of the same prompt succeeded).
    static let maxRetries = 1

    /// - Parameters:
    ///   - isDecodeFailure: the thrown error is the recoverable backend decode
    ///     failure (`InferenceError.decode`), not a client error such as
    ///     `.contextTooFull`.
    ///   - retriesSoFar: retries this generation has already used.
    ///   - tokensEmitted: pieces already delivered to the caller. Anything above
    ///     zero makes a retry visible (the caller would see the text twice), so
    ///     it is never retried.
    static func shouldRetry(isDecodeFailure: Bool,
                            retriesSoFar: Int,
                            tokensEmitted: Int) -> Bool {
        guard isDecodeFailure else { return false }
        guard retriesSoFar < maxRetries else { return false }
        return tokensEmitted == 0
    }
}
