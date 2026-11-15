/// Why an in-flight generation must stop at its next safe boundary.
///
/// The prompt loop always observed server shutdown between prefill chunks, but
/// an HTTP client that closed the socket was only noticed in the token callback
/// — i.e. after the whole prompt had been decoded. In the 2026-11-15 device log
/// a client left 1.5 s into a 4,451-token prefill and the serial queue stayed
/// busy for the remaining ~19 s.
///
/// One decision for both call sites keeps the precedence explicit: shutdown wins,
/// because `stop()` drains the queue and its cache-tracking breadcrumbs are keyed
/// on that path.
enum GenerationCancel: Equatable {
    case serverStopping
    case clientDisconnected

    static func cause(isShuttingDown: Bool, isClientDisconnected: Bool) -> GenerationCancel? {
        if isShuttingDown { return .serverStopping }
        if isClientDisconnected { return .clientDisconnected }
        return nil
    }

    var logDescription: String {
        switch self {
        case .serverStopping: return "server stopping"
        case .clientDisconnected: return "client disconnected"
        }
    }
}
