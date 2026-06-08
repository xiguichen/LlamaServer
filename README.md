# LlamaServer (iOS)

Run a local LLM on your iPhone and serve it to your LAN over an **OpenAI-compatible
HTTPS API**. Built around [llama.cpp](https://github.com/ggml-org/llama.cpp).

- **No Mac required to build.** The app is built in **GitHub Actions** on a macOS
  runner and produces an **unsigned `.ipa`**.
- **No Apple Developer account / no code signing in CI.** Re-signing happens later,
  on-device, with a sideloading tool (AltStore, Sideloadly, ESign, etc.).
- **UI** to pick a `.gguf` model, set port/context size, and **Start/Stop** the server.
- **HTTPS** server (self-signed cert) reachable from other machines on the same network.

> Status: scaffolding is complete and self-consistent, but it has **not been
> compiled** (authored on Windows). The first CI run is the source of truth — see
> [Known risks](#known-risks-first-build).

---

## How it works

iOS sandboxes apps and does **not** allow launching a separate server *binary*
(no `fork`/`exec` of arbitrary executables). So instead of running the `llama-server`
executable, this app:

1. **Links llama.cpp as a framework** (`llama.xcframework`) directly into the app.
   The framework is the **official prebuilt** artifact downloaded from llama.cpp's
   GitHub releases (no local compilation).
2. Runs an **in-process HTTPS server** ([Telegraph](https://github.com/Building42/Telegraph))
   that wraps llama inference and exposes the OpenAI API.

The UI "Start/Stop" controls loading the model + starting/stopping that embedded server.

### API endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | `/health`               | `{"status":"ok"}`                    |
| GET    | `/v1/models`            | Lists the loaded model               |
| POST   | `/v1/chat/completions`  | OpenAI chat completion (non-streaming) |

Example from another machine on the network (self-signed cert → `--insecure`):

```bash
curl --insecure https://<iphone-ip>:8443/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

---

## Repository layout

```
project.yml                       XcodeGen spec (the .xcodeproj is generated in CI)
Sources/App/
  LlamaServerApp.swift            App entry point
  ContentView.swift               SwiftUI UI (model picker, start/stop, logs)
  ServerViewModel.swift           UI <-> engine/server glue
  LlamaInference.swift            llama.cpp C API wrapper (load/tokenize/generate)
  LlamaHTTPServer.swift           HTTPS server + OpenAI routes (Telegraph)
  OpenAIModels.swift              Codable request/response types
  NetworkInfo.swift               LAN IPv4 discovery
  Info.plist                      App metadata + file sharing + local-network usage
Resources/                        server.p12 (generated) is bundled here
scripts/
  fetch-llama-xcframework.sh      Downloads official prebuilt llama.xcframework -> vendor/
  generate-cert.sh                Generates the self-signed Resources/server.p12
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
`fetch prebuilt llama.xcframework` → `generate self-signed cert` → `xcodegen generate` →
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

After install:

1. Add a `.gguf` model: open the **Files** app → On My iPhone → **LlamaServer**,
   and copy a model in (file sharing is enabled). Or pick it via **Choose .gguf**.
2. Tap **Choose .gguf**, pick the model, set a port (default `8443`), tap **Start Server**.
3. The UI shows the endpoint, e.g. `https://192.168.1.50:8443`. Point your OpenAI
   client there (disable TLS verification for the self-signed cert).

> Use small, quantized models (e.g. 1B–3B, `Q4_K_M`) — phones are RAM-limited.

---

## Build locally (only if you have a Mac)

```bash
brew install xcodegen
bash scripts/fetch-llama-xcframework.sh     # downloads vendor/llama.xcframework
bash scripts/generate-cert.sh               # produces Resources/server.p12
xcodegen generate
open LlamaServer.xcodeproj
```

> `fetch-llama-xcframework.sh` only needs `curl` + `unzip`, so you can even run it
> on Windows/Git-Bash to pre-populate `vendor/` — no Mac or compiler required for
> that step.

---

## Known risks (first build)

This was authored without a compiler. Most-likely things to adjust on the first CI run:

1. **llama.cpp API drift.** `Sources/App/LlamaInference.swift` is written against a
   recent C API (`llama_model_load_from_file`, `llama_init_from_model`,
   `llama_sampler_*`, `llama_vocab_is_eog`, `llama_kv_self_clear`, …) matching the
   pinned tag **`b9553`**. If you change `LLAMA_TAG` in
   `scripts/fetch-llama-xcframework.sh` and the build fails, adjust that file.
2. **xcframework layout / module name.** Verified for `b9553`: the zip contains
   `build-apple/llama.xcframework`, the device slice is `ios-arm64`, and the module
   is named `llama` (the modulemap auto-links Metal/Accelerate/Foundation/c++).
3. **Telegraph TLS API.** Uses `Server()`, `server.tlsConfig = TLSConfig(identity:)`,
   `CertificateIdentity(p12URL:passphrase:)`. Verify against the resolved Telegraph
   version (`from: 0.30.0` in `project.yml`).
4. **`UIBackgroundModes: audio`** is a placeholder to reduce suspension; iOS still
   limits background networking. Keep the app foregrounded for reliable serving.

## Roadmap / TODO

- [ ] Streaming responses (SSE) for `/v1/chat/completions`.
- [ ] Persist the last-used model and settings.
- [x] Pin a specific `llama.cpp` release tag (`b9553`, prebuilt xcframework).
- [ ] Optional bearer-token auth for the API.
