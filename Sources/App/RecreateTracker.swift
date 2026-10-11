import Foundation

/// Bookkeeping for the periodic context recreate (Metal allocator-state
/// flush), shared by the streaming / chat / completions handlers.
///
/// The recreate re-allocates the KV cache (~0.3–0.5 s at long contexts), so
/// the handlers run it only AFTER the client's response has been fully
/// delivered — never between the last token and finish_reason, which made
/// every 10th streamed response visibly stall.
///
/// Semantics (mirroring the original inline logic, now pinned by tests):
///   - count successful generations; at `threshold` a recreate is due;
///   - a FAILED recreate attempt does not reset the counter, so the next
///     generation retries immediately (the engine needs the flush);
///   - a successful attempt resets the cadence.
///
/// All mutations happen on the server's serial inference queue.
struct RecreateTracker {
    let threshold: Int
    private(set) var count = 0

    /// Call once per completed (successful) generation.
    /// Returns true when a periodic recreate is now due; the counter is held
    /// until `recreateFinished` confirms the attempt.
    mutating func record() -> Bool {
        guard count + 1 >= threshold else {
            count += 1
            return false
        }
        return true
    }

    /// Report the outcome of the recreate attempt that `record()` made due.
    mutating func recreateFinished(success: Bool) {
        if success { count = 0 }
    }
}
