import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: ServerViewModel
    @State private var showingImporter = false

    /// .gguf has no registered UTType; accept any file and let the user pick.
    private var allowedTypes: [UTType] { [.data, .item] }

    var body: some View {
        NavigationView {
            Form {
                modelsSection
                downloadSection
                serverSection
                statusSection
                logSection
            }
            .navigationTitle("LlamaServer")
            .onAppear {
                viewModel.refreshModels()
                viewModel.reloadLogs()
            }
            .fileImporter(isPresented: $showingImporter,
                          allowedContentTypes: allowedTypes,
                          allowsMultipleSelection: false) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first { viewModel.importModel(from: url) }
                case .failure(let error):
                    viewModel.log("File pick failed: \(error.localizedDescription)")
                }
            }
        }
        .navigationViewStyle(.stack)
    }

    // MARK: - Models

    private var modelsSection: some View {
        Section {
            if viewModel.models.isEmpty {
                Text("No models yet. Import a .gguf or download one below.")
                    .foregroundColor(.secondary)
            } else {
                ForEach(viewModel.models) { model in
                    Button { viewModel.selectModel(model) } label: {
                        HStack(spacing: 12) {
                            Image(systemName: viewModel.selectedModel == model
                                  ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(viewModel.selectedModel == model
                                                 ? .accentColor : .secondary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.name).lineLimit(1).truncationMode(.middle)
                                Text(model.sizeText).font(.caption).foregroundColor(.secondary)
                            }
                            Spacer()
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isRunning || viewModel.isBusy)
                }
                .onDelete { viewModel.deleteModels(at: $0) }
            }
        } header: {
            HStack {
                Text("Models")
                Spacer()
                Button { showingImporter = true } label: {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                .disabled(viewModel.isRunning || viewModel.isBusy)
            }
        }
    }

    // MARK: - Download

    private var downloadSection: some View {
        Section {
            TextField("https://…/model.gguf", text: $viewModel.downloadURLString)
                .textInputAutocapitalization(.never)
                .disableAutocorrection(true)
                .keyboardType(.URL)
                .disabled(viewModel.downloader.isDownloading)

            if viewModel.downloader.isDownloading {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: viewModel.downloader.progress)
                    HStack {
                        Text(viewModel.downloader.progressText)
                            .font(.caption).foregroundColor(.secondary)
                        Spacer()
                        Button("Cancel", role: .destructive) { viewModel.cancelDownload() }
                    }
                }
            } else {
                Button { viewModel.startDownload() } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
                .disabled(viewModel.downloadURLString.isEmpty || viewModel.isRunning)
            }

            if let err = viewModel.downloader.error, !viewModel.downloader.isDownloading {
                Text(err).font(.caption).foregroundColor(.red)
            }
        } header: {
            Text("Download from URL")
        } footer: {
            Text("Downloads continue when the screen locks or the app is backgrounded — you don't need to keep the screen on.")
        }
    }

    // MARK: - Server config

    private var serverSection: some View {
        Section {
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
            Toggle("Multi-Token Prediction (MTP)", isOn: $viewModel.useMtp)
                .disabled(viewModel.isRunning || viewModel.isBusy)
                .font(.caption)
        } header: {
            Text("Server")
        } footer: {
            Text("Context is automatically capped to fit this device's memory (and the model's trained window). Large values like 140000 aren't possible on-device.")
        }
    }

    // MARK: - Status / control

    private var statusSection: some View {
        Section("Status") {
            HStack {
                Circle().fill(statusColor).frame(width: 10, height: 10)
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
            .disabled(viewModel.isBusy || (viewModel.selectedModel == nil && !viewModel.isRunning))
        }
    }

    private var logSection: some View {
        Section {
            Picker("Detail level", selection: $viewModel.logLevel) {
                ForEach(LogLevel.allCases) { level in
                    Text(level.label).tag(level)
                }
            }
            Text(viewModel.logLevel.detail)
                .font(.caption)
                .foregroundColor(.secondary)

            Toggle("Log prompt & response content", isOn: $viewModel.logPromptContent)
                .font(.caption)
                .disabled(viewModel.logLevel != .verbose)

            if viewModel.logs.isEmpty {
                Text("No activity yet.").foregroundColor(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(viewModel.logs.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(.caption2, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                    }
                }
                .frame(height: 180)
            }
        } header: {
            HStack {
                Text("Logs")
                Spacer()
                Button { viewModel.reloadLogs() } label: {
                    Image(systemName: "arrow.clockwise")
                }
                ShareLink(item: viewModel.logFileURL) {
                    Image(systemName: "square.and.arrow.up")
                }
                Button(role: .destructive) { viewModel.clearLogs() } label: {
                    Image(systemName: "trash")
                }
            }
        } footer: {
            Text("Logs persist to llamaserver.log and survive a crash. Read them on-device via Files → On My iPhone → LlamaServer, or tap Share above. After a crash, reopen the app and tap Reload — the last line shows the step that crashed. Set the level to Verbose to capture full request and response payloads for troubleshooting.")
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
