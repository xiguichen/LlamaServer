# LlamaServer — agent guide

iOS app that runs llama.cpp in-process and exposes an OpenAI-compatible HTTP
API (SwiftUI UI + Telegraph server). No separate server binary.

**Keep this file short.** It is prepended to the system prompt of every agent
turn, and a long prompt measurably degrades the model's tool calling
(`docs/TOOL-CALLS.md` → *Prompt dilution*). Put detail in `docs/`, not here.

## Read when you need it

| Doc | Load it when |
|---|---|
| [`docs/STACK.md`](docs/STACK.md) | you need upstream links (model card, llama.cpp, Telegraph, XcodeGen), how pi discovers these docs, or how to point a client at the server |
| [`docs/ENGINE.md`](docs/ENGINE.md) | touching `LlamaInference.swift`: MTP rounds, sampling, generation loop/timings, context recreate, decode-failure recovery |
| [`docs/MEMORY.md`](docs/MEMORY.md) | touching KV cache, context sizing/clamping, YaRN, or the memory budget |
| [`docs/TOOL-CALLS.md`](docs/TOOL-CALLS.md) | touching tool prompts, parsing, streaming filters, thinking switches — or debugging a turn that fails to call a tool |
| [`docs/API.md`](docs/API.md) | touching request/response types, token limits, `stop`, `enable_thinking`, OpenAI field presence |
| [`docs/LOGGING.md`](docs/LOGGING.md) | adding log lines, changing the in-app panel, or fetching logs |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | debugging a deployed build, uploading an IPA, or running on-device compliance tests |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | making or repeating any performance claim (measured baseline, method, disproved claims) |

## Build & test (CI only)

No local Xcode required. Everything runs in GitHub Actions:

1. Commit and push to `main`/`master` (or open a PR).
2. `.github/workflows/build.yml` on an `xcode-27` runner (macOS 27 arm64, Xcode 27 → iOS 27 SDK) runs two jobs:
   - **test**: `xcodegen generate` → `xcodebuild test` (simulator; destination `iPhone 17` — Xcode 27 images have no iPhone 16 device type)
   - **build**: `xcodegen generate` → `xcodebuild` (unsigned Release) → `bash scripts/package-ipa.sh` → upload artifact.
3. The test job must pass; the build job produces `LlamaServer-unsigned.ipa`.

## CI notes

- **No `Package.swift`.** The project is defined in `project.yml` (XcodeGen). Always edit `project.yml`, never the `.xcodeproj`; CI regenerates it and it is never committed.
- `vendor/llama.xcframework` is prebuilt, fetched by `bash scripts/fetch-llama-xcframework.sh`, pinned to tag `b11557`. There is no local llama.cpp compilation.
- The framework download is cached across runs by release tag + script hash.
- New test-target sources must be listed in `project.yml` and must **not** `import` the app module (tests compile app sources directly).

## Architecture

- **Single process.** llama.cpp linked as a framework (`import llama`), runs in-process.
- **HTTP server**: Telegraph. Streaming requests are intercepted via `interceptHandler` on `StreamableServer` and written straight to `HTTPConnection`.
- **Serial inference queue**: `DispatchQueue(label: "llama.inference.serial")` — every `LlamaInference` call goes through it (not thread-safe).
- **Context recreated** every 10 generations to flush Metal/allocator state. The recreate runs off the response path and the prefix-reuse snapshot survives it. See `docs/ENGINE.md`.
- **Streaming** flushes each token as SSE, with empty delta keep-alives every 2 s; **non-streaming** runs synchronously on the serial queue.

## Key source files

| File | Role |
|------|------|
| `Sources/App/LlamaInference.swift` | C API wrapper: load model, tokenize, decode, sample (MTP-aware) |
| `Sources/App/LlamaHTTPServer.swift` | Telegraph routes, streaming handler, tool-call parsing, recreate logic |
| `Sources/App/OpenAIModels.swift` | Codable OpenAI request/response types + `ChatRequestSpec` (pure 400-on-missing-model) |
| `Sources/App/ToolCallParser.swift` | Parse JSON/XML `<tool_call>` envelopes and `<think>` blocks; streaming filters |
| `Sources/App/ToolPrompt.swift` | Pure tool-schema injection, wire-format detection, history replay |
| `Sources/App/ThinkingSwitch.swift` | Pure: closes the reasoning block when `enable_thinking:false` |
| `Sources/App/ServerViewModel.swift` | @MainActor glue: model library, start/stop engine, UI state |
| `Sources/App/KVCacheSpec.swift` | Pure KV decisions: bytes/token, head dims, quant capability, context clamp, YaRN |
| `Sources/App/RecreateTracker.swift` | Pure: when a recreate is due, and its off-response-path placement |
| `Sources/App/ReuseSurvival.swift` | Pure: the prefix-reuse snapshot survives `recreateContext()` |
| `Sources/App/FileLogger.swift` | Crash-survivable on-disk log; error/warn fsync'd, info+ buffered |
| `Sources/App/LogRouter.swift` | Pure: which file (server/inference/payloads) a line lands in |
| `Sources/App/LogPanelSpec.swift` | Pure panel budget: basic-only filter, line cap, truncation |
| `Sources/App/LogFetchSpec.swift` | Pure `GET /logs` contract |
| `Sources/App/UpdateSpec.swift` / `UpdateStore.swift` | Self-upgrade contract + the upload's filesystem half |
| `Sources/App/StartupTimer.swift` | Pure startup-phase accumulator (`load phases:` / `Startup gap:`) |
| `Sources/App/PrefillStats.swift` | Pure `prefill=` segment: rate over **decoded** tokens + `(X of Y, Z reused)` |

## Invariants (do not break)

- **One queue.** Never call `LlamaInference` off `llama.inference.serial`.
- **Prefix reuse.** Dense caches trim with `seq_rm`; hybrid/SSM caches cannot trim a suffix and restore a whole-state snapshot instead. `docs/ENGINE.md`.
- **`unload()` before discarding `LlamaInference`**, or the multi-GB Metal allocation stays resident and the next model load returns NULL (OOM).
- **`llama_memory_clear(mem, false)`** — `true` memsets the buffers (~39 ms for a 2 GiB q8_0 KV) and buys nothing.
- **The memory ceiling is measured**, never an assumed fraction of RAM; the MTP draft context counts at its own size. `docs/MEMORY.md`.
- **Tool calls use the model's own wire format**, replay assistant `tool_calls` into history, and serialize arguments with `.fragmentsAllowed` (without it a scalar argument raises an uncatchable ObjC exception → process abort). `docs/TOOL-CALLS.md`.
- **A tool turn filters reasoning then tool envelopes** (`ReasoningStreamFilter` → `ToolStreamFilter`).
- **OpenAI field presence is load-bearing**: `logprobs`/`refusal`/`content` are explicit null when absent, and a missing/empty `model` is a 400. These gate the compliance tests. `docs/API.md`.
- **Context-cap messages name the real cause** (model window vs memory), never "to fit device memory" by default.

## Performance

**Measured, not inferred — [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) is the baseline.** Read it before making or repeating a claim; add a row when you measure something.

Headlines (A19 Pro, 4B Q4_K_M, ctx 131072, KV q8_0, build `470957a`): prompt 170–350 t/s for new tokens; generation 10–17 t/s falling with context; `verify-gpu` is 80–90% of every MTP round (CPU work is ~3 ms/round); thermals swing results ~30%, so quote context length and pair A/B runs. The MTP figures below are **depth 1** (the recorded baseline; depth 2 is the current default and is being re-measured): depth 1 is **+11–16% generation / ~4% slower prefill**.

## Tests

- Pure-logic specs in `Tests/LlamaServerTests/`: `OpenAIModelsTests`, `ToolCallParserTests`, `ToolPromptTests`, `ThinkingSwitchTests`, `ChatSpecComplianceTests`, `LogFetchSpecTests`, `LogRoutingTests`, `PrefillStatsTests`, `UpdateSpecTests`, `UpdateStoreTests`, and the engine specs (`MtpSpecTests`, `MtpSpeculatorTests`, `KVCacheSpecTests`, `MemoryBudgetTests`, `DecodeRetryTests`, …).
- All are pure Swift — no model file, no inference. They run on the simulator in CI.
- Run focused: `xcodebuild test -project LlamaServer.xcodeproj -scheme LlamaServerTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing:LlamaServerTests/ToolCallParserTests CODE_SIGNING_ALLOWED=NO`
- Local machine is CLT-only (no XCTest): `xcodegen generate` is the local syntax/project validator; CI runs the tests.

## Style

- `OSAllocatedUnfairLock` for thread-safe flags/counters (not the `os_unfair_lock` C API).
- No `#warning` — use `FileLogger.shared.warn(...)` for recoverable issues.
- Logging rules (levels, routing, fsync policy, panel budget) are in `docs/LOGGING.md`.
