import Foundation

/// Pure presentation rules for the Context size field.
///
/// The engine clamps the requested window to the device's measured memory budget
/// and the model's trained window — the 2026-11-15 log loaded a 9B at 48,640
/// tokens while the field still showed the requested 262,144 (and the field is
/// disabled while running, so the user could not correct it). The field now
/// shows the *effective* size while the server runs, and the request again when
/// stopped (the field is editable then, and that request is the next Start's
/// input).
///
/// The requested value is never overwritten: it is both the next Start's input
/// and part of the engine cache key (`EngineConfigSpec`), so replacing it would
/// make a stop/start reload the multi-GB model instead of reusing it.
enum ContextDisplay {

    static func fieldText(requested: String, effective: Int?, isRunning: Bool) -> String {
        guard isRunning, let effective else { return requested }
        return String(effective)
    }

    static func limitNote(requested: Int,
                          effective: Int?,
                          clamp: KVCacheSpec.ContextClamp,
                          isRunning: Bool) -> String? {
        guard isRunning, let effective else { return nil }
        switch clamp {
        case .notReduced:
            return nil
        case .memoryBudget:
            return "Using \(effective) tokens — device memory budget (requested \(requested))."
        case .modelWindow:
            return "Using \(effective) tokens — the model's trained window (requested \(requested); enable long context for more)."
        }
    }
}
