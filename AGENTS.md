# LlamaServer — compact reference for agents

## Build & test (CI only)

No local Xcode required. Everything runs in GitHub Actions:

1. Commit and push to `main`/`master` (or open a PR).
2. CI pipeline `.github/workflows/build.yml` on a `macos-14` runner runs two jobs:
   - **test**: `xcodegen generate` → `xcodebuild test` (simulator, unit tests only)
   - **build**: `xcodegen generate` → `xcodebuild` (unsigned Release) → `bash scripts/package-ipa.sh` → upload artifact.
3. Check the "Actions" tab for results. The test job must pass; the build job produces `LlamaServer-unsigned.ipa`.

## CI notes

- `vendor/llama.xcframework` is cached across runs by the release tag + script hash. Only re-downloaded when the tag or `fetch-llama-xcframework.sh` changes.
- **No Package.swift.** Project is defined in `project.yml` (XcodeGen). Always edit `project.yml`, never the `.xcodeproj` directly.
- **No local llama.cpp compilation.** `vendor/llama.xcframework` is prebuilt, fetched with `bash scripts/fetch-llama-xcframework.sh`. Pinned to tag `b9553`.
- The `.xcodeproj` is **never committed** (`*.xcodeproj` in `.gitignore`). CI regenerates it on every run.

## Architecture

- **Single process.** llama.cpp linked as a framework (`import llama`), runs in-process. No separate server binary.
- **HTTP server**: Telegraph framework. Intercepts streaming requests via `interceptHandler` on `StreamableServer` (bypasses normal route handler for SSE writes direct to `HTTPConnection`).
- **Serial inference queue**: `DispatchQueue(label: "llama.inference.serial")` — all `LlamaInference` calls go through this queue (not thread-safe).
- **Context recreated** every 10 generations (`contextRecreateThreshold`) to flush Metal GPU/allocator state that can cause hangs.
- **Non-streaming** chat completions: synchronous on the serial queue, returns full response once done.
- **Streaming** chat completions: async on the serial queue, each token flushed as SSE via direct connection write. Keep-alive sends empty delta chunks every 2s.

## Key source files

| File | Role |
|------|------|
| `Sources/App/LlamaInference.swift` | C API wrapper: load model, tokenize, decode, sample (MTP-aware) |
| `Sources/App/LlamaHTTPServer.swift` | Telegraph server, routes, streaming handler, tool-call parsing, recreate logic |
| `Sources/App/OpenAIModels.swift` | Codable request/response types for OpenAI API |
| `Sources/App/ToolCallParser.swift` | Parse `<tool_call>{…}</tool_call>` envelopes and `<think>…</think>` reasoning blocks from model output |
| `Sources/App/ServerViewModel.swift` | @MainActor glue: model library, start/stop engine, UI state |
| `Sources/App/FileLogger.swift` | Crash-survivable on-disk log; sync-flushes every line |

## Important gotchas

- **MTP (Multi-Token Prediction)**: When enabled, `n_outputs_max` is derived from the model's GGUF metadata key `nextn_predict_layers` (or fallback `n_predict_layers`). Total output slots = `layers + 1`. Falls back to 3 if metadata is absent. **Must not sample from more slots than the model supports** — will `ggml_abort` in `llama_sampler_sample`. The runtime guard `llama_get_logits_ith(ctx, i) != nil` prevents this.
- **`recreateContext()`** frees old context before creating new one. Sets `contextValid = false` immediately after free so a subsequent failure can't use the dangling pointer.
- **`unload()`** must be called before discarding `LlamaInference` — otherwise ARC runs `deinit` and the (multi-GB) Metal/CPU allocation stays resident, causing next `llama_model_load_from_file` to return NULL (OOM).
- **LLAMA_TAG** in `scripts/fetch-llama-xcframework.sh` must point to a llama.cpp release that publishes an `xcframework.zip` asset. Changing it may require adapting `Sources/App/LlamaInference.swift` for API drift.
- **Stop field** in API: OpenAI sends `stop` as string or array of strings — decoded via `StopField` enum (handles both).
- **Tool calls** are implemented via system prompt injection (`injectToolDefs`), not GBNF grammar. Model wraps tool calls in `<tool_call>{json}</tool_call>` envelopes. `ToolCallParser.parse()` extracts them post-generation.
- **`<think>` reasoning** blocks are stripped from streamed content by `ReasoningStreamFilter` (character-by-character state machine, NSRegularExpression for final parse).
- **`resolvedMaxTokens`** precedence: `n_predict` > `max_completion_tokens` > `max_tokens` for chat, `n_predict` > `max_tokens` for `/v1/completions`.
- **`enable_thinking`** can be top-level or nested under `chat_template_kwargs.enable_thinking`. Default is `false` (opt-in per request).

## Tests

- Two test files: `OpenAIModelsTests` (JSON decode/encode round-trips, model parsing) and `ToolCallParserTests` (regex parsing, think-block stripping, streaming filter state machine).
- Both are pure Swift — no model file, no inference. They run on simulator in CI.
- Run focused: `xcodebuild test -project LlamaServer.xcodeproj -scheme LlamaServerTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing:LlamaServerTests/ToolCallParserTests CODE_SIGNING_ALLOWED=NO`

## Style conventions

- `OSAllocatedUnfairLock` for thread-safe flags/counters (not `os_unfair_lock` C API).
- No `#warning` — use `FileLogger.shared.warn(...)` for recoverable issues; the log is crash-survivable.
- FileLogger sync-flushes to disk. Set `minimumLevel` from UI `.verbose` to capture full request/response payloads.
- `llama_log_set` routes llama.cpp native logs into FileLogger at `.debug` level.
