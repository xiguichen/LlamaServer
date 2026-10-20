# Stack: the models and frameworks this server rests on

**Why this file exists.** Model and upstream-library questions are answered from
the upstream source, not inferred from our call sites. This is the canonical
list: the link to read and the version we pin. Performance numbers live **only**
in [`PERFORMANCE.md`](PERFORMANCE.md); engine/memory/tool-call decisions live in
[`ENGINE.md`](ENGINE.md), [`MEMORY.md`](MEMORY.md) and
[`TOOL-CALLS.md`](TOOL-CALLS.md).

Every link below was verified to return HTTP 200 on **2026-10-18**.

## Models

### `Qwopus3.5-4B-Coder-MTP` — the model we serve and measure against

**Model card (read this first):**
<https://huggingface.co/Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF>

**Base model:** <https://huggingface.co/Qwen/Qwen3.5-4B>

A 4B coder/agent fine-tune of the Qwen3.5 family (Trace Inversion + curriculum
SFT on agent trajectories), published with Multi-Token-Prediction layers and a
vision projector. The card describes it as optimized for local execution, code
debugging, structured tool use and reasoning-heavy developer workflows.

**The file we actually run:** `Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf` (2.78 GB).
The repo publishes the full quant ladder (sizes from the HF repo tree API,
2026-10-18):

| Quant | Size | Note |
|---|---|---|
| BF16 | 8.67 GB | will not fit the phone's budget with a large KV |
| Q8_0 | 4.61 GB | |
| Q6_K | 3.56 GB | |
| Q5_K_M | 3.16 GB | |
| Q5_K_S | 3.07 GB | |
| **Q4_K_M** | **2.78 GB** | **what we run and measure** |
| Q4_K_S | 2.63 GB | |
| IQ4_XS | 2.59 GB | |
| Q3_K_L | 2.48 GB | |
| Q3_K_M | 2.32 GB | |
| Q3_K_S | 2.12 GB | |
| Q2_K | 1.96 GB | |
| `mmproj-F32.gguf` | 0.68 GB | vision projector — **we never load it**; there is no multimodal path in this server |

#### What the card documents, and where our code matches it

- **Long-context recipe (YaRN).** The card recommends RoPE/YaRN scaling rather
  than only raising `--ctx-size` for contexts beyond 32K, and gives the
  `llama.cpp` invocations:

  ```bash
  # 128K
  --ctx-size 131072 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768
  # 256K
  --rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 32768
  ```

  These are literally what we implement: `KVCacheSpec.yarnOrigCtx = 32768` and
  `KVCacheSpec.yarnFactor(forContext:)`, which derives the factor from the
  context being built (4 at 131,072, 8 at 262,144). We cannot pass these as CLI
  flags — the engine is in-process — so they are set as context params, and
  `llama.cpp` re-sets `n_ctx_train = yarn_orig_ctx / rope_freq_scale` to match.
- **MTP.** The card benchmarks "MTP n=2" against the Qwen3.5-4B-MTP reference.
  We run depth 2 (`mtpMaxDraft`), with the output-slot count resolved from the
  GGUF's own head count — see `docs/ENGINE.md` (*MTP*). The measured depth-1 vs
  depth-2 result is in `docs/PERFORMANCE.md`.

#### What we verified ourselves (provenance matters)

| Fact | Value | Where it came from |
|---|---|---|
| Trained window | `n_ctx_train = 262144` | llama.cpp load log on device |
| Architecture | 32 layers, hybrid: 3 × linear attention + 1 × full attention, repeating (`full_attention_interval = 4`) | base `config.json` `layer_types`; device log `kv layers 8/32` |
| KV-bearing layers | 8 of 32 (only the full-attention ones) | as above |
| Head geometry | `head_dim` 256, `num_key_value_heads` 4, `num_attention_heads` 16, `hidden_size` 2560 | base `config.json`; device log `head_dim K=256 V=256` |
| MTP layers | `nextn_predict_layers` present → 2 output slots at depth 1, 3 at depth 2 | GGUF metadata read at load |

**Cross-check that the two sources agree.** The KV arithmetic that the codebase
depends on follows from the config alone:

```
8 KV layers × (256 K + 256 V) head dims × 4 KV heads × 34/32 (q8_0) = 17,408 B/token
```

which is exactly the `17,408 B/token` measured on device and recorded in
`PERFORMANCE.md`. If a future model swap changes the config, that identity is
the fastest way to re-derive the KV budget.

## Frameworks

| Framework | Link | Version we use | Wired in |
|---|---|---|---|
| **llama.cpp** | <https://github.com/ggml-org/llama.cpp> | tag `b11557` | `scripts/fetch-llama-xcframework.sh` → `vendor/llama.xcframework` (gitignored), linked in `project.yml` |
| **Telegraph** | <https://github.com/Building42/Telegraph> | SwiftPM `from: "0.30.0"` (no lockfile is committed) | `project.yml` → `packages:` |
| **XcodeGen** | <https://github.com/yonaskolb/XcodeGen> | whatever `brew install xcodegen` resolves | `.github/workflows/build.yml` → `xcodegen generate` |

### llama.cpp — the inference engine

The whole engine runs in-process: `import llama`, no separate server binary.

- **Pinned release asset:** <https://github.com/ggml-org/llama.cpp/releases/download/b11557/llama-b11557-xcframework.zip>
- **Headers we compile against:** `vendor/llama.xcframework/ios-arm64/llama.framework/Headers/llama.h`
- **Used for:** GGUF model loading, the Metal backend, tokenize/decode/sample,
  the KV cache, MTP/nextn hidden states (`llama_set_embeddings_nextn`), and
  sequence-state snapshots (`llama_state_seq_save_file` / `_load_file`).
- **Why the pin is a decision, not a detail:** the tag must publish an
  `xcframework.zip` asset, and bumping it can require adapting our Swift for API
  drift. See `docs/ENGINE.md` (*Vendored llama.cpp*).
- **Two upstream limits that shaped this codebase** (details in
  `docs/TOOL-CALLS.md` and `docs/MEMORY.md`):
  1. `llama_chat_apply_template` takes **no `tools` parameter**, and the
     framework exports no `common_chat_*` symbols — so tool schemas are injected
     as a system block (`ToolPrompt`) and calls are parsed out of the generated
     text (`ToolCallParser`) rather than rendered by the template.
  2. `ctx_other` is dropped for non-Gemma4 architectures, so the MTP draft
     context has **its own memory**: the prompt must be mirrored into it, and
     its KV cache is a real allocation the memory budget has to count — but at
     its own size. The draft context is typed `LLAMA_CONTEXT_TYPE_MTP`, so
     llama.cpp builds only the model's nextn blocks
     (`llama_model_n_layer_nextn`): measured on device, its cache is one layer
     (2,176 B/token on the 4B model), not a second full cache
     (`KVCacheSpec.chargedKvBytesPerToken`).

### Telegraph — the HTTP server

An HTTP/1.1 server. We use its route table (`/v1/chat/completions`,
`/v1/models`, `/v1/completions`, `/logs`) and, critically, the
`StreamableServer.interceptHandler` hook, which lets the streaming path bypass
the normal route handler and write SSE frames straight to the `HTTPConnection`
so each token is flushed as it is sampled.

### XcodeGen — project generation

The build is defined by `project.yml`; `*.xcodeproj` is gitignored and **never
committed**. CI runs `xcodegen generate` before building, so a change to the
project file is always a change to `project.yml`.

**CI toolchain for reference:** GitHub Actions runner `xcode-27` (macOS 27
arm64, Xcode 27 → iOS 27 SDK).

## How an agent finds these docs

Agents auto-load `AGENTS.md` at startup (pi: `--no-context-files` opts out;
OpenCode uses the same convention). It is deliberately a short **map**: a table
of which doc to load when, so a session reads only what a question needs. One
canonical file per subsystem, no per-client copies. This is not just tidiness —
a long system prompt measurably suppresses tool calls (see
`docs/TOOL-CALLS.md` → *Prompt dilution*).

## Consuming this server as a backend

The server speaks the OpenAI API, so any OpenAI-compatible client works. A pi
provider entry, for example:

```json
{
  "providers": {
    "llamacpp": {
      "baseUrl": "http://<device>:8443/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [{ "id": "qwen3.5:4b", "contextWindow": 160000 }]
    }
  }
}
```

- **Endpoints:** `GET /v1/models`, `POST /v1/chat/completions` (SSE streaming and
  non-streaming), `POST /v1/completions` (legacy text completion), and
  `GET /logs` (or `/v1/logs`) for pulling the on-device log files over the LAN.
- **`apiKey` is ignored** by the server, but clients often insist on one.
- **The `model` id is a client-side label.** The server does not match it to the
  loaded GGUF, but it must be present and non-empty (a missing/empty `model` is
  a 400 — deliberately, for the OpenAI spec compliance gate).
- **Footgun — declare a context window the server can actually back.** The
  server allocates `min(request, hardMaxContext = 262,144)`, then reduces it
  further to what the memory budget (and, with MTP on, *two* KV caches) can hold.
  A client that declares more will assemble prompts the server answers with
  `.contextTooFull` (HTTP 400). Keep the client's `contextWindow` at or below
  what the load line reports as the effective context.
