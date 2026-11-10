import Foundation

/// Incremental stop-string matcher for the streaming generation loops.
///
/// Generation appends one decoded piece at a time and must answer two
/// questions on every token:
///   1. Has a stop string appeared? (truncate the response there.)
///   2. How many trailing characters must be withheld because they could be
///      the start of a stop string split across the next token?
///
/// Re-scanning the whole accumulated output for (1) and rebuilding suffixes of
/// the whole output for (2) makes both answers O(output) per token — quadratic
/// over a generation. A stop string of length L can only be completed by text
/// in the last L-1 characters plus the new piece, so this type retains only
/// that tail. Each answer is then bounded by the longest stop string, never by
/// the response length.
struct StopScanner {
    /// Non-empty stop strings. Ties are resolved by earliest position, matching
    /// the old whole-output scan (`firstStop`).
    let stops: [String]
    /// Longest stop string in characters (0 when there are no stops).
    let maxStopLength: Int

    /// The last `maxStopLength - 1` characters of the stream. A new match can
    /// only complete within these plus the next piece.
    private var tail = ""
    /// Number of leading stream characters that have already fallen out of
    /// `tail`, so a match offset can still be reported against the full output.
    private var tailStart = 0

    init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
        self.maxStopLength = self.stops.map(\.count).max() ?? 0
    }

    var isEmpty: Bool { stops.isEmpty }

    /// Characters currently retained for match detection. Bounded by
    /// `maxStopLength - 1`; exposed so a regression test can prove the buffer
    /// does not accumulate the whole response.
    var retainedCharacterCount: Int { tail.count }

    /// Appends `piece` and returns the absolute character offset (from the
    /// first piece ever appended) where the earliest stop begins, or nil.
    ///
    /// Callers that stop on a match never append again; updating the tail even
    /// on a match keeps the offset bookkeeping honest.
    mutating func append(_ piece: String) -> Int? {
        guard !stops.isEmpty else { return nil }
        let window = tail + piece
        let windowStart = tailStart

        var earliest: Range<String.Index>?
        for stop in stops {
            guard let r = window.range(of: stop) else { continue }
            if earliest == nil || r.lowerBound < earliest!.lowerBound {
                earliest = r
            }
        }

        let keep = min(maxStopLength - 1, window.count)
        tailStart = windowStart + window.count - keep
        tail = keep > 0 ? String(window.suffix(keep)) : ""

        guard let r = earliest else { return nil }
        return windowStart + window.distance(from: window.startIndex, to: r.lowerBound)
    }

    /// Length of the longest suffix of the stream that is a *proper* prefix of
    /// some stop string. Withhold this many trailing characters from the
    /// client so a stop split across the next token is never leaked.
    func heldSuffixLength() -> Int {
        guard maxStopLength > 1 else { return 0 }
        var hold = 0
        for stop in stops {
            let upper = min(stop.count - 1, tail.count)
            var k = upper
            while k > hold {
                if tail.suffix(k) == stop.prefix(k) {
                    hold = k
                    break
                }
                k -= 1
            }
        }
        return hold
    }
}
