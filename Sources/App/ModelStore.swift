import Foundation

/// A `.gguf` model file stored in the app's Documents/models directory.
struct ModelFile: Identifiable, Hashable {
    let url: URL
    let size: Int64

    var id: URL { url }
    var name: String { url.lastPathComponent }
    var sizeText: String {
        ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
    }
}

/// Owns the on-device model library: a `models/` folder inside the app's
/// Documents directory. Files here are fully owned by the app, so llama.cpp can
/// mmap them without security-scoped-resource lifetime issues.
final class ModelStore {
    static let shared = ModelStore()

    private let fm = FileManager.default

    /// Documents/models (created on first access).
    var directory: URL {
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("models", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// All `.gguf` files currently in the library, sorted by name.
    func list() -> [ModelFile] {
        let keys: [URLResourceKey] = [.fileSizeKey]
        guard let items = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return items
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map { url in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                    .flatMap { Int64($0) } ?? 0
                return ModelFile(url: url, size: size)
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// Destination URL for a given file name inside the library, ensuring a safe,
    /// non-empty `.gguf` name. Sanitizes against path traversal — a name from an
    /// untrusted download URL must not be able to escape the models directory.
    func destinationURL(for fileName: String) -> URL {
        // Keep only the final path component, then strip separators / leading
        // dots so values like "..", "../x", or "a/b" can't traverse out.
        var name = (fileName as NSString).lastPathComponent
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        while name.hasPrefix(".") { name.removeFirst() }
        if name.isEmpty { name = "model.gguf" }
        if !name.lowercased().hasSuffix(".gguf") { name += ".gguf" }

        let dest = directory.appendingPathComponent(name)
        // Defense in depth: confirm the result is still directly inside the dir.
        guard dest.deletingLastPathComponent().standardizedFileURL.path
                == directory.standardizedFileURL.path else {
            return directory.appendingPathComponent("model.gguf")
        }
        return dest
    }

    /// Copies a (possibly security-scoped) picked file into the library.
    /// Runs synchronously — call off the main thread for large files.
    @discardableResult
    func importModel(from source: URL) throws -> URL {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }

        let dest = destinationURL(for: source.lastPathComponent)
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.copyItem(at: source, to: dest)
        return dest
    }

    func delete(_ url: URL) throws {
        try fm.removeItem(at: url)
    }
}
