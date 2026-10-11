---
name: llm-compliance-test
description: Use when validating a deployed LlamaServer build on a physical iPhone against the OpenAI Chat Completions spec with llm-comply — after deploying the app, to confirm the server's API surface is still spec-compliant and nothing regressed.
---

# LlamaServer OpenAI Compliance Test (llm-comply)

## Overview

`llm-comply` (PyPI: `Oaklight/llm-comply`, v0.5.0) validates an HTTP endpoint
against the official OpenAI Chat Completions spec. LlamaServer serves an
OpenAI-compatible surface over **plain HTTP, no TLS, no auth**, so compliance
testing the phone after a deploy is a real end-to-end check that the server
still speaks the spec after any change to `LlamaHTTPServer` / `OpenAIModels`.

Core principle: this runs against the **live app on the phone**, over the LAN —
it is the on-device test that CI (simulator-only, no model) cannot do.

## When to Use

- After installing a freshly built IPA on the iPhone and starting the server.
- After changing chat/completions routing, streaming SSE, tool-call parsing, or
  the Codable request/response types.
- To catch a spec regression that unit tests (pure logic) and CI (no model) miss.

Do NOT use for: pure-logic changes with no server/API surface change (the local
harness + CI already cover those).

## Verified Ground Truth (llm-comply 0.5.0)

Facts checked against the installed tool and a local stub that mirrors
LlamaServer's exact response shape — do not guess these:

| Fact | Value |
|------|-------|
| `--api-key` / `-k` | **Required** by the CLI; a placeholder string is fine because LlamaServer ignores auth headers entirely (it still sends `Authorization: Bearer <placeholder>`, harmless). |
| LlamaServer scheme | Plain `http`, no TLS, no auth — base URL is `http://<ip>:<port>` with no cert to trust. |
| Default port | `8443` (UI default; confirm on the phone's server screen — it's user-editable). |
| Model name | Discover via `GET /v1/models` → `data[0].id`. Pass the same string to `-m` (the server ignores `-m`, but a real name keeps output readable). |
| Exit code | `1` if any test failed, `0` otherwise. Ignored errors are emptied, so a fully-ignored run reads as PASSED. |
| Per-test status | `passed` / `failed` / `skipped`. |

## Setup (once per machine)

```bash
python3 -m venv ~/.llm-comply-venv
~/.llm-comply-venv/bin/pip install llm-comply          # add [rich] for color
```

## Known Spec Gaps (measured, not a regression)

LlamaServer's `ChatCompletionChoice` emits only `index`, `message`,
`finish_reason`; `ChatMessage` has no `refusal`; and nothing emits `logprobs`.
The pinned OpenAI spec in llm-comply 0.5.0 marks `logprobs` and
`message.refusal` **required**, so content tests fail schema validation against
the real server. This is a known, stable baseline — NOT a sign the build is
broken. The measured per-test outcome on the real server shape is the table
below; only a *change* from that table is a regression.

## Full openai-chat Roster & Measured Outcome on LlamaServer

| Test ID | Outcome | Why |
|---------|---------|-----|
| `chat-streaming-response` | **PASS** | SSE deltas + `[DONE]` terminator — the core streaming path. |
| `chat-error-handling` | **PASS** | Expects 400/422 on a no-`model` request — server returns a 4xx. |
| `chat-basic-response` | FAIL (gap) | Schema requires `logprobs` + `message.refusal`; server omits both. |
| `chat-system-prompt` | FAIL (gap) | Same required-field gap. |
| `chat-multi-turn` | FAIL (gap) | Same required-field gap. |
| `chat-reasoning-summary` | SKIP | Auto-skips: model name has no `o-series`/`gpt-5`/`deepseek-r` token. |
| `chat-tool-calling` | FAIL (gap) | Also needs `tool_choice:"required"` honored; server returns 200 without `tool_calls`. |
| `chat-image-input` | FAIL (gap) | Text-only model; server ignores image content (same required-field gap too). |
| `chat-streaming-usage` | FAIL (gap) | Needs `stream_options.include_usage` usage accounting in the final chunk. |

## The Test Invocation

Ask the user for the iPhone's LAN IP (the app shows it on the server screen),
then run the full suite and compare against the table above:

```bash
V=~/.llm-comply-venv/bin
IP=<iphone-ip>; PORT=8443
$V/llm-comply --format openai-chat \
  -u "http://$IP:$PORT/v1" \
  -k "llamaserver-placeholder" \
  -m "<model-name-from-/v1/models>" \
  --json
```

## Interpreting Results

- **`chat-streaming-response` and `chat-error-handling` PASS, and the gap tests
  fail exactly as tabulated** → the deployed build is behaving as designed.
  The API surface (routing, SSE streaming, error shape) is intact. Ship
  confidence confirmed.
- **A gap test that *previously passed* now fails, or a PASS test fails** →
  real regression. Re-run with `-v` to dump request/response, then check the
  phone's on-disk log (Files app → LlamaServer → llamaserver.log) for the
  matching breadcrumb.
- **`chat-streaming-response` fails** → highest-priority signal: the SSE
  streaming path regressed. Investigate before shipping.
- **Connection error / timeout** → confirm the server is running (green on the
  phone), Mac and iPhone are on the same LAN, and iOS local-network permission
  is granted (the app triggers the prompt via Bonjour).

## Optional: Close the Gap

To make the content tests pass, add the two missing fields to LlamaServer —
`logprobs: null` on each choice and `refusal: null` on each message (and usage
accounting for the streaming-usage test). That is a server change, not a test
change; this skill only measures. Leave it out unless asked.

## Common Mistakes

- **Forgetting `-k`** → CLI errors "api-key is required". Always pass a
  placeholder.
- **Using `https://`** → LlamaServer is plain HTTP. No cert.
- **Wrong port** → default 8443, but user-editable; read it off the phone.
- **Treating the gap FAILs as a broken build** → they are the measured,
  expected baseline (missing `logprobs`/`refusal`). Only the two PASS tests
  gating, and only a *change* from the table, signals a regression.
- **Assuming `-m` matters** → the server ignores the model field; use the real
  name for readable output only.
