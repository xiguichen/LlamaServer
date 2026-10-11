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
}
