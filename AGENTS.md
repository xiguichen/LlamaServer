# LlamaServer — agent guide

iOS app that runs llama.cpp in-process and exposes an OpenAI-compatible HTTP
API (SwiftUI UI + Telegraph server). No separate server binary.

**Keep this file short.** It is prepended to the system prompt of every agent
turn, and a long prompt measurably degrades the model's tool calling
(`docs/TOOL-CALLS.md` → *Prompt dilution*). Put detail in `docs/`, not here.

## Read when you need it

| Doc | Load it when |
|---|---|
| [`docs/STACK.md`](docs/STACK.md) | upstream links and versions (model card, llama.cpp, Telegraph, XcodeGen) or pointing a client at the server |
| [`docs/ENGINE.md`](docs/ENGINE.md) | touching `LlamaInference.swift`: MTP rounds, sampling, generation loop/timings, context recreate, decode-failure recovery |
| [`docs/MEMORY.md`](docs/MEMORY.md) | touching KV cache, context sizing/clamping, YaRN, or the memory budget |
| [`docs/TOOL-CALLS.md`](docs/TOOL-CALLS.md) | touching tool prompts, parsing, streaming filters, thinking switches — or debugging a turn that fails to call a tool |
| [`docs/API.md`](docs/API.md) | touching request/response types, token limits, `stop`, `enable_thinking`, OpenAI field presence |
| [`docs/LOGGING.md`](docs/LOGGING.md) | adding log lines, changing the in-app panel, or fetching logs |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | debugging a deployed build, uploading an IPA, or running on-device compliance tests |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | making or repeating any performance claim (measured baseline, method, disproved claims) |

## Build & test (CI only)

No local Xcode. Push to `main` (or open a PR); `.github/workflows/build.yml` runs
on an `xcode-27` runner (macOS 27, Xcode 27 → iOS 27 SDK):

- **test**: `xcodegen generate` → `xcodebuild test` (simulator, `iPhone 17`)
- **build**: `xcodegen generate` → unsigned Release → `scripts/package-ipa.sh` → `LlamaServer-unsigned.ipa`

The test job must pass; the build job produces the IPA.

## CI notes

- **No `Package.swift`.** The project is defined in `project.yml` (XcodeGen). Always edit `project.yml`, never the `.xcodeproj`; CI regenerates it and it is never committed.
- `vendor/llama.xcframework` is prebuilt, fetched by `bash scripts/fetch-llama-xcframework.sh`, pinned to tag `b11557`. There is no local llama.cpp compilation.
- The framework download is cached across runs by release tag + script hash.

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

**Measured, not inferred — [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) is the only source of numbers.** Read it before quoting or measuring; add a row when you measure. Never infer performance from code.

A19 Pro, 4B Q4_K_M: prefill 170–350 t/s for new tokens; generation 10–17 t/s at 24–48k context (shorter context is faster); `verify-gpu` is 80–90% of each MTP round. Thermals swing ~30%: quote the context and pair A/B runs. MTP depth 2 (current): **+1.6–4.8%** vs depth 1 at ≤14k prompt tokens, **−2.5%** at 21k; depth 1 vs MTP off is **+11–16%** generation / −4% prefill.

## Tests

- Pure-logic specs in `Tests/LlamaServerTests/` — no model file, no inference, all run in CI: OpenAI models, tool parser/prompt, thinking switch, chat-spec compliance, log routing/fetch, prefill stats, update spec/store, and the engine specs (`Mtp*`, `KVCacheSpec`, `MemoryBudget`, `DecodeRetry`). Add new sources to `project.yml`; do not `import` the app module (tests compile app sources directly).
- Run focused: `xcodebuild test -project LlamaServer.xcodeproj -scheme LlamaServerTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17' -only-testing:LlamaServerTests/ToolCallParserTests CODE_SIGNING_ALLOWED=NO`
- Local machine is CLT-only (no XCTest): `xcodegen generate` is the local syntax/project validator; CI runs the tests.

## Style

- `OSAllocatedUnfairLock` for thread-safe flags/counters (not the `os_unfair_lock` C API).
- No `#warning` — use `FileLogger.shared.warn(...)` for recoverable issues.
- Logging rules (levels, routing, fsync policy, panel budget) are in `docs/LOGGING.md`.
