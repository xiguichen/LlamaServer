import Foundation

/// Decisions about carrying sequence-state reuse across a context recreate.
///
/// `recreateContext()` builds a brand-new `llama_context` to flush accumulated
/// Metal GPU/allocator state that can hang `llama_sampler_sample`. The recreate
/// used to drop prefix reuse unconditionally (`cachedTokenIds = []`), which
/// forced the next turn to re-decode the whole prompt.
///
/// That cost is unnecessary. The reuse snapshot is an on-disk blob of the seq_0
/// sequence state (`llama_state_seq_save_file`) plus the token ids of the prompt
/// prefix it represents. `llama_state_seq_load_file` restores the entire
/// sequence into any compatible context on the same model — the fresh context
/// created by the recreate is just as valid a destination as the original. So
/// for BOTH dense (partial-reuse via `seq_rm`) and hybrid (snapshot-only) models
/// the snapshot survives a recreate and the next turn restores it.
///
/// This function exists so that decision is explicit and testable in isolation,
/// instead of being an implicit invariant of the call sites.
enum ReuseSurvival {

    /// Whether the sequence-state snapshot taken before a recreate can still be
    /// trusted and restored after the new context is built.
    ///
    /// Both cache strategies keep their snapshot across a recreate:
    ///  - Dense (`supportsPartialReuse == true`): the on-disk seq_0 blob is
    ///    restorable into the fresh context exactly like the live cache was.
    ///  - Hybrid (`supportsPartialReuse == false`): already relied on the
    ///    snapshot for every turn; the recreate must not break that.
    static func snapshotSurvivesRecreate(supportsPartialReuse: Bool) -> Bool {
        // The snapshot file and `snapshotTokens` are instance state that persist
        // across `recreateContext()`; nothing in the recreate invalidates them.
        _ = supportsPartialReuse
        return true
    }

    /// Whether `recreateContext()` must serialize the live KV cache to disk
    /// before freeing it.
    ///
    /// Dense models normally reuse their live KV cache via `seq_rm` and never
    /// write a snapshot during generation. A recreate destroys that cache, so
    /// the only way to keep prefix reuse (and to avoid treating a now-empty
    /// cache as resident) is to save the current sequence state first. Hybrid
    /// models already write their snapshot every turn, so they need no extra
    /// capture.
    static func denseSnapshotCaptureNeeded(supportsPartialReuse: Bool,
                                           hasCachedTokens: Bool) -> Bool {
        supportsPartialReuse && hasCachedTokens
    }

    /// For hybrid/recurrent models (`supportsPartialReuse == false`), the live
    /// KV cache is still reusable without any trim when it is already a strict
    /// prefix of the new prompt. The snapshot path is the general fallback, but
    /// it serializes and reloads a multi-megabyte blob even when the live cache
    /// already holds exactly the right tokens (for example a client that resends
    /// a growing prompt, or an assistant-continuation request).
    ///
    /// The live cache is trusted only when its sequence bounds prove the whole
    /// tracked prefix is resident: `seq_pos_min == 0` (nothing evicted from the
    /// front) and `seq_pos_max + 1 == cachedTokens.count` (the cache ends where
    /// `cachedTokenIds` says it does). At least one prompt token must remain to
    /// decode so the forward pass yields fresh logits.
    ///
    /// Token ids are `Int32` (llama_token's underlying type) so this file stays
    /// free of the llama module, like the other pure specs compiled into tests.
    ///
    /// - Returns: the reusable prefix length, or nil when the live cache cannot
    ///   be used whole and the caller must fall back to the snapshot / full
    ///   decode.
    static func liveHybridReuseLen(cachedTokens: [Int32],
                                   promptTokens: [Int32],
                                   seqPosMin: Int,
                                   seqPosMax: Int) -> Int? {
        guard !cachedTokens.isEmpty, cachedTokens.count < promptTokens.count else { return nil }
        guard seqPosMin == 0, seqPosMax + 1 == cachedTokens.count else { return nil }
        guard zip(cachedTokens, promptTokens).allSatisfy({ $0.0 == $0.1 }) else { return nil }
        return cachedTokens.count
    }
}
