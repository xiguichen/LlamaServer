import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: ServerViewModel
    @State private var showingImporter = false

    /// .gguf has no registered UTType; treat it as raw data so the importer
    /// shows all files and we accept whatever the user picks.
    private var allowedTypes: [UTType] { [.data, .item] }

    var body: some View {
        NavigationView {
            Form {
                modelSection
                serverSection
                statusSection
                logSection
            }
            .navigationTitle("LlamaServer")
            .fileImporter(isPresented: $showingImporter,
                          allowedContentTypes: allowedTypes,
                          allowsMultipleSelection: false) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first { viewModel.selectModel(url: url) }
                case .failure(let error):
                    viewModel.log("File pick failed: \(error.localizedDescription)")
                }
            }
        }
        .navigationViewStyle(.stack)
    }

    // MARK: - Sections

    private var modelSection: some View {
        Section("Model") {
            HStack {
                Text(viewModel.selectedModelURL?.lastPathComponent ?? "No model selected")
                    .foregroundColor(viewModel.selectedModelURL == nil ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button("Choose .gguf") { showingImporter = true }
                    .disabled(viewModel.isRunning || viewModel.isBusy)
            }
        }
    }

    private var serverSection: some View {
        Section("Server") {
            HStack {
                Text("Port")
                Spacer()
                TextField("8443", text: $viewModel.port)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
                    .disabled(viewModel.isRunning || viewModel.isBusy)
            }
            HStack {
                Text("Context size")
                Spacer()
                TextField("4096", text: $viewModel.contextSize)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
                    .disabled(viewModel.isRunning || viewModel.isBusy)
            }
        }
    }

    private var statusSection: some View {
        Section("Status") {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(viewModel.status.label)
                Spacer()
                if viewModel.isBusy { ProgressView() }
            }

            if let url = viewModel.serverURL, viewModel.isRunning {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Endpoint").font(.caption).foregroundColor(.secondary)
                    Text(url).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                    Text("POST \(url)/v1/chat/completions")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                }
            }

            Button(action: { viewModel.toggle() }) {
                Text(viewModel.isRunning ? "Stop Server" : "Start Server")
                    .frame(maxWidth: .infinity)
                    .fontWeight(.semibold)
            }
            .buttonStyle(.borderedProminent)
            .tint(viewModel.isRunning ? .red : .green)
            .disabled(viewModel.isBusy || (viewModel.selectedModelURL == nil && !viewModel.isRunning))
        }
    }

    private var logSection: some View {
        Section("Logs") {
            if viewModel.logs.isEmpty {
                Text("No activity yet.").foregroundColor(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(viewModel.logs.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(.caption2, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .frame(height: 160)
            }
        }
    }

    private var statusColor: Color {
        switch viewModel.status {
        case .running:      return .green
        case .loadingModel: return .orange
        case .error:        return .red
        case .stopped:      return .gray
        }
    }
}
