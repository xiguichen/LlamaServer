# LlamaServer (iOS)

Run a local LLM on your iPhone and serve it to your LAN over an **OpenAI-compatible
HTTP API**. Built around [llama.cpp](https://github.com/ggml-org/llama.cpp).

- **No Mac required to build.** The app is built in **GitHub Actions** on a macOS
  runner and produces an **unsigned `.ipa`**.
- **No Apple Developer account / no code signing in CI.** Re-signing happens later,
  on-device, with a sideloading tool (AltStore, Sideloadly, ESign, etc.).
- **UI** to pick a `.gguf` model, set port/context size, and **Start/Stop** the server.
- **HTTP** server reachable from other machines on the same network (no cert hassle).

> Status: **builds in CI** and produces a valid unsigned IPA. Runtime (loading a
> model + serving inference) is unverified — that needs a real device. See
> [Caveats](#caveats).

---

## How it works

iOS sandboxes apps and does **not** allow launching a separate server *binary*
(no `fork`/`exec` of arbitrary executables). So instead of running the `llama-server`
executable, this app:

1. **Links llama.cpp as a framework** (`llama.xcframework`) directly into the app.
   The framework is the **official prebuilt** artifact downloaded from llama.cpp's
   GitHub releases (no local compilation).
2. Runs an **in-process HTTP server** ([Telegraph](https://github.com/Building42/Telegraph))
   that wraps llama inference and exposes the OpenAI API.

The UI "Start/Stop" controls loading the model + starting/stopping that embedded server.

### API endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | `/health`               | `{"status":"ok"}`                    |
| GET    | `/v1/models`            | Lists the loaded model               |
| POST   | `/v1/chat/completions`  | OpenAI chat completion (non-streaming) |

Example from another machine on the network:

```bash
curl http://<iphone-ip>:8443/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

---

## Repository layout

```
project.yml                       XcodeGen spec (the .xcodeproj is generated in CI)
Sources/App/
  LlamaServerApp.swift            App entry point
  ContentView.swift               SwiftUI UI (models list, import, download, start/stop)
  ServerViewModel.swift           UI <-> engine/server glue
  ModelStore.swift                Model library (list/import/delete in Documents/models)
  ModelDownloader.swift           URL download to disk with live progress
  LlamaInference.swift            llama.cpp C API wrapper (load/tokenize/generate)
  LlamaHTTPServer.swift           HTTP server + OpenAI routes (Telegraph)
  OpenAIModels.swift              Codable request/response types
  NetworkInfo.swift               LAN IPv4 discovery
  Info.plist                      App metadata + file sharing + local-network usage
scripts/
  fetch-llama-xcframework.sh      Downloads official prebuilt llama.xcframework -> vendor/
  package-ipa.sh                  Wraps the .app into an unsigned .ipa
.github/workflows/build.yml       CI: build -> unsigned .ipa artifact
```

---

## Build it (GitHub Actions)

1. Push this repo to GitHub.
2. Go to **Actions → "Build unsigned IPA" → Run workflow** (or just push to `main`).
   - Optionally set the `llama.cpp` tag (defaults to `b9553`). It must be a release
     that publishes an `xcframework.zip` asset — see
     <https://github.com/ggml-org/llama.cpp/releases>.
3. When it finishes, download the **`LlamaServer-unsigned-ipa`** artifact. It
   contains `LlamaServer-unsigned.ipa`.

The pipeline:
`fetch prebuilt llama.xcframework` → `xcodegen generate` →
`xcodebuild (CODE_SIGNING_ALLOWED=NO)` → `package Payload/ → .ipa` → upload artifact.

> The prebuilt framework is **downloaded, not compiled** — and cached across runs.
> No part of the build compiles llama.cpp from source.

---

## Install on your iPhone (no paid account)

The `.ipa` is **unsigned**; sign + install it on-device with any of:

- **[Sideloadly](https://sideloadly.io/)** (Windows/macOS) — sign with your free
  Apple ID and install over USB. Free-account apps expire after 7 days; just
  re-install when needed.
- **[AltStore](https://altstore.io/)** — similar, with on-device refresh.
- **ESign / TrollStore** (device/iOS-version dependent).

After install — three ways to add a model, then run it:

1. **Add a `.gguf` model** (any of):
   - **Download from URL**: paste a direct `.gguf` link (e.g. a Hugging Face
     `…/resolve/main/model.gguf`) into the *Download from URL* field and tap
     **Download** — progress is shown live.
   - **Import**: tap **Import** (Models section) to copy a `.gguf` from Files /
     iCloud into the app.
   - **Files app / Finder**: drop a `.gguf` into On My iPhone → **LlamaServer**,
     then Import it.
2. **Pick a model** from the **Models** list (tap to select; swipe to delete).
   All models live in the app's `Documents/models` and persist between launches.
3. Set a port (default `8443`) and tap **Start Server**.
4. The UI shows the endpoint, e.g. `http://192.168.1.50:8443`. Point your OpenAI
   client there (plain HTTP — no certificate needed).

> Use small, quantized models (e.g. 1B–3B, `Q4_K_M`) — phones are RAM-limited.

---

## Build locally (only if you have a Mac)

```bash
brew install xcodegen
bash scripts/fetch-llama-xcframework.sh     # downloads vendor/llama.xcframework
xcodegen generate
open LlamaServer.xcodeproj
```

> `fetch-llama-xcframework.sh` only needs `curl` + `unzip`, so you can even run it
> on Windows/Git-Bash to pre-populate `vendor/` — no Mac or compiler required for
> that step.

---

## Caveats

The project **builds green** in CI and produces a valid unsigned IPA. Remaining
caveats:

1. **Runtime is unverified.** The build + IPA structure are confirmed, but actually
   loading a model and serving inference can only be proven on a real iPhone.
2. **llama.cpp API drift.** `Sources/App/LlamaInference.swift` is written against the
   pinned tag **`b9553`** C API. If you change `LLAMA_TAG` in
   `scripts/fetch-llama-xcframework.sh` and the build fails, adjust that file.
3. **Plain HTTP.** The server has no TLS — only expose it on trusted local networks.
4. **`UIBackgroundModes: audio`** is a placeholder to reduce suspension; iOS still
   limits background networking. Keep the app foregrounded for reliable serving.

## Roadmap / TODO

- [x] Model library: list / select / import / **download from URL**.
- [ ] Streaming responses (SSE) for `/v1/chat/completions`.
- [ ] Persist the last-used model and settings.
- [x] Pin a specific `llama.cpp` release tag (`b9553`, prebuilt xcframework).
- [ ] Optional bearer-token auth for the API.
