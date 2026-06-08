import SwiftUI

@main
struct LlamaServerApp: App {
    @StateObject private var viewModel = ServerViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
    }
}
