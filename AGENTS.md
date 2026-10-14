# LlamaServer — compact reference for agents

## Build & test (CI only)

No local Xcode required. Everything runs in GitHub Actions:

1. Commit and push to `main`/`master` (or open a PR).
2. CI pipeline `.github/workflows/build.yml` on an `xcode-27` runner (macOS 27 arm64, Xcode 27 → iOS 27 SDK) runs two jobs:
   - **test**: `xcodegen generate` → `xcodebuild test` (simulator, unit tests only; destination `iPhone 17` — Xcode 27 images have no iPhone 16 device type)
   - **build**: `xcodegen generate` → `xcodebuild` (unsigned Release) → `bash scripts/package-ipa.sh` → upload artifact.
3. Check the "Actions" tab for results. The test job must pass; the build job produces `LlamaServer-unsigned.ipa`.

## CI notes

- `vendor/llama.xcframework` is cached across runs by the release tag + script hash. Only re-downloaded when the tag or `fetch-llama-xcframework.sh` changes.
- **No Package.swift.** Project is defined in `project.yml` (XcodeGen). Always edit `project.yml`, never the `.xcodeproj` directly.
- **No local llama.cpp compilation.** `vendor/llama.xcframework` is prebuilt, fetched with `bash scripts/fetch-llama-xcframework.sh`. Pinned to tag `b11557`.
- The `.xcodeproj` is **never committed** (`*.xcodeproj` in `.gitignore`). CI regenerates it on every run.

## Architecture

- **Single process.** llama.cpp linked as a framework (`import llama`), runs in-process. No separate server binary.
- **HTTP server**: Telegraph framework. Intercepts streaming requests via `interceptHandler` on `StreamableServer` (bypasses normal route handler for SSE writes direct to `HTTPConnection`).
- **Serial inference queue**: `DispatchQueue(label: "llama.inference.serial")` — all `LlamaInference` calls go through this queue (not thread-safe).
- **Context recreated** every 10 generations (`contextRecreateThreshold`) to flush Metal GPU/allocator state that can cause hangs. The recreate runs OFF the response path (`RecreateTracker`, called after the finish chunk in streaming), and the prompt-prefix reuse snapshot survives it (`ReuseSurvival`) — a dense model's `cachedTokenIds` and the on-disk seq blob are preserved so the next turn restores from disk instead of re-decoding the whole prompt.
- **Non-streaming** chat completions: synchronous on the serial queue, returns full response once done.
- **Streaming** chat completions: async on the serial queue, each token flushed as SSE via direct connection write. Keep-alive sends empty delta chunks every 2s.

## Key source files

| File | Role |
|------|------|
| `Sources/App/LlamaInference.swift` | C API wrapper: load model, tokenize, decode, sample (MTP-aware) |
| `Sources/App/LlamaHTTPServer.swift` | Telegraph server, routes, streaming handler, tool-call parsing, recreate logic |
| `Sources/App/OpenAIModels.swift` | Codable request/response types for OpenAI API + `ChatRequestSpec` (pure 400-on-missing-model validation) |
| `Sources/App/ToolCallParser.swift` | Parse `<tool_call>{…}</tool_call>` envelopes and `<think>…</think>` reasoning blocks from model output |
| `Sources/App/ServerViewModel.swift` | @MainActor glue: model library, start/stop engine, UI state |
| `Sources/App/KVCacheSpec.swift` | Pure KV-cache decisions: bytes/token for f16/q8_0/q4_0, head-dim resolution, quant capability, context-window cap, YaRN recipe |
| `Sources/App/RecreateTracker.swift` | Pure bookkeeping: when the periodic context recreate is due, and the off-response-path placement decision |
| `Sources/App/ReuseSurvival.swift` | Pure predicate: the prompt-prefix reuse snapshot survives a `recreateContext()` (dense and hybrid) |
| `Sources/App/FileLogger.swift` | Crash-survivable on-disk log; error/warn fsync'd sync, info+ buffered |
| `Sources/App/StartupTimer.swift` | Pure startup-phase accumulator: records named phases + durations, formats the load-gap breakdown line (`load phases:` / `Startup gap:`) |

## Important gotchas

- **MTP (Multi-Token Prediction)**: When enabled, `n_outputs_max` (output slots) is resolved by `MtpSpec.resolveOutputSlots` — the larger of the model's GGUF-derived head count (`nextn_predict_layers`, or fallback `n_predict_layers`, `+1` for the main head, fallback 3 when absent, overridable by an explicit count) and the verify batch `maxDraft + 1`. **Must not sample from more slots than the model supports** — will `ggml_abort` in `llama_sampler_sample`. The runtime guard `llama_get_logits_ith(ctx, i) != nil` prevents this (the plain path probes via `MtpProbe`). **A non-negative index is a BATCH token index, not an output slot**: a row with `logits = 0` returns NULL and `llama_sampler_sample` aborts. Read the draft context's logits with index `-1` (last output row), never a batch index whose row has no logits. The target's unmasked nextn hidden states are indexed by token offset within the last `llama_decode` batch, so after the prompt the last token's h is row `lastChunkCount − 1`.
- **MTP speculative depth** is `mtpMaxDraft = 1` (hard-coded, no UI toggle). The round machinery is depth-N capable (chained drafts, N-row verify, draft seed rows), but depth-2 measured **~9–13% slower** on device (MTP-on 18.3/17.5 vs off 19.9/19.7 tok/s). Each round drafts up to `mtpMaxDraft` tokens with the `draft` context (`LLAMA_CONTEXT_TYPE_MTP`); the draft token is a greedy `MtpSpec.argmax` (vDSP) over `llama_get_logits_ith(dft, -1)`. The confidence gate (`MtpSpec.draftConfidence >= MtpSpec.pMin`) is off: `pMin` is 0.0 (`MtpSpec.gateEnabled` false), so a computed draft is always verified. The `target` verifies `[idLast] + drafts` in one batch and **every emitted token is sampled from the target**, so a bad draft costs a round but never corrupts output. Verify rows are sampled lazily: sampling stops at the first real that misses its draft or is eog, because later rows are never emitted. `MtpSpeculator.decide` is the pure acceptance rule (`accepted`, `emitCount`, `rollbackPos`, `carryHFromRow = emitCount − 1`). On a full accept the draft context has a one-position gap (the last accepted token was never drafted into it); `MtpSpeculator.draftSeedIndices` lists the missing rows, which are decoded as leading seed rows of the NEXT round's draft decode (one weight pass), not as a separate catch-up decode. Partial accepts stay contiguous after the two-context `seq_rm` rollback. **The draft context must also see the prompt**: llama.cpp drops `ctx_other` for non-Gemma4 archs, so the draft has its own memory, and `mirrorPromptChunkIntoDraft` decodes every target prompt chunk into it (row k = token x_k with target h_{k-1}, logits off), like upstream draft-mtp `process()`. It is skipped when the draft shares the target's memory (`is_mem_shared`). The draft KV is kept across generations and trimmed to the target's reused prefix when `draftPromptTokens` shows it holds the same tokens; otherwise it starts empty. A failed mirror leaves the draft empty, never with a position gap, because a gap makes every later draft decode fail. `llama_sampler_sample` accepts the token it returns, so the driver never calls `llama_sampler_accept` itself. The `MTP stats:` log line reports `accepted`/`depth` (avg accepted drafts per round) plus per-phase timings (`draft`/`verify`/`sample` ms per round, `MtpPhaseStats`). Without `llama_synchronize`, the verify decode's GPU time is billed to `sample` (Metal decode is async; the first logits read waits for it).
- **`recreateContext()`** frees old context before creating new one. Sets `contextValid = false` immediately after free so a subsequent failure can't use the dangling pointer.
- **`unload()`** must be called before discarding `LlamaInference` — otherwise ARC runs `deinit` and the (multi-GB) Metal/CPU allocation stays resident, causing next `llama_model_load_from_file` to return NULL (OOM).
- **LLAMA_TAG** in `scripts/fetch-llama-xcframework.sh` must point to a llama.cpp release that publishes an `xcframework.zip` asset. Changing it may require adapting `Sources/App/LlamaInference.swift` for API drift.
- **Stop field** in API: OpenAI sends `stop` as string or array of strings — decoded via `StopField` enum (handles both).
- **Tool calls** are implemented via system prompt injection (`injectToolDefs`), not GBNF grammar. Model wraps tool calls in `<tool_call>{json}</tool_call>` envelopes. `ToolCallParser.parse()` extracts them post-generation.
- **`<think>` reasoning** blocks are stripped from streamed content by `ReasoningStreamFilter` (character-by-character state machine, NSRegularExpression for final parse).
- **`resolvedMaxTokens`** precedence: `n_predict` > `max_completion_tokens` > `max_tokens` for chat, `n_predict` > `max_tokens` for `/v1/completions`.
- **`enable_thinking`** can be top-level or nested under `chat_template_kwargs.enable_thinking`. Default is `false` (opt-in per request).
- **OpenAI-spec field presence**: llm-comply's pinned spec marks `logprobs` (per choice), `refusal` and `content` (per message) required, and `model` required on input. The server encodes `logprobs`/`refusal`/`content` as **explicit null** when absent (present, never fabricated — see the `encode(to:)` override in `ChatMessage`/`ChatCompletionChoice`; Swift would omit a nil optional) and rejects a missing/empty `model` with a 400 via `ChatRequestSpec`. Do not remove these — the on-device `llm-compliance-test` skill (running `llm-comply openai-chat` against a deployed build) gates on them; regression shows as `chat-error-handling` failing (200 instead of 400) or content tests failing on missing fields (e.g. `chat-tool-calling` fails on a missing `content` when a tool-call turn drops it). The streaming path (`StreamingChoice`/`Delta`) needs no such fields and passes as-is.
- **Head dims come from GGUF `attention.key_length`/`value_length`** (suffix-matched via `LlamaInference.ggufMetaValue`), falling back to `n_embd/n_head`. Qwen3-4B stores 128 while derived is 80 — deriving alone under-counts f16 KV cost by 1.6× and the clamp approved unbackable contexts.
- **Hybrid models keep KV only on full-attention layers** (`KVCacheSpec.kvLayerCount`): the GGUF `<arch>.full_attention_interval` (e.g. Qwen3.5 SSM+attention, interval 4 → 8 of 32 layers) selects which layers cache K/V; recurrent layers have fixed-size state. Counting all layers over-estimates KV cost interval× (observed 36,864 vs actual 9,216 B/token → a viable 131,072 context clamped to 83,968).
- **Quantized KV cache** (`kvCacheType` q8_0/q4_0): only when `head_dim % 32 == 0` (llama.cpp b9553 validates per-head); quantized `type_v` requires flash attention, so `flash_attn_type` is forced `ENABLED` (AUTO can resolve off → constructor throw → NULL). Ineligible models fall back to f16 with a warn log (`KVCacheSpec.resolveCacheType`).
- **YaRN long-context** (`longContextYarn`): sets `rope_scaling_type=YARN`, `rope_freq_scale=0.25` (factor 4), `yarn_orig_ctx=32768` on every context (target/MTP/recreate) and lifts the `n_ctx_train` clamp (Qwen3-4B: 40,960 → up to 131,072, still memory-clamped).

## Tests

- Two test files: `OpenAIModelsTests` (JSON decode/encode round-trips, model parsing) and `ToolCallParserTests` (regex parsing, think-block stripping, streaming filter state machine).
- Both are pure Swift — no model file, no inference. They run on simulator in CI.
- Run focused: `xcodebuild test -project LlamaServer.xcodeproj -scheme LlamaServerTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing:LlamaServerTests/ToolCallParserTests CODE_SIGNING_ALLOWED=NO`

## Style conventions

- `OSAllocatedUnfairLock` for thread-safe flags/counters (not `os_unfair_lock` C API).
- No `#warning` — use `FileLogger.shared.warn(...)` for recoverable issues; the log is crash-survivable.
- FileLogger: error/warn are written AND fsynced synchronously (crash breadcrumbs survive aborts); info/debug/verbose are buffered on the logger queue (fsync ≤ 1s / 64KB) so the inference path never blocks on I/O; NSLog fires for warn+ only; the on-screen mirror delivers ≤ 10Hz batches. Set `minimumLevel` from UI `.verbose` to capture full request/response payloads.
- `llama_log_set` routes llama.cpp native logs into FileLogger at `.debug` level.
