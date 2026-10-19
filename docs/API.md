# OpenAI API contract

Read this when touching `OpenAIModels.swift`, request validation, token limits,
`stop`, or OpenAI field presence (the on-device compliance tests gate on these).

## Token limits

`resolvedMaxTokens` precedence:

- chat: `n_predict` > `max_completion_tokens` > `max_tokens`
- `/v1/completions`: `n_predict` > `max_tokens`

## `stop`

OpenAI sends `stop` as a string or an array of strings. It is decoded via the
`StopField` enum, which handles both.

## `enable_thinking`

May be top-level or nested under `chat_template_kwargs.enable_thinking`. Default
is `false` (opt-in per request).

The C API takes no template kwargs, so the flag reaches the prompt only through
`ThinkingSwitch` (`Sources/App/ThinkingSwitch.swift`): thinking off appends the
closed block the template's `enable_thinking is false` branch would have emitted
(`<think>\n\n</think>\n\n` after the bare primer, or just the closer if a
renderer already opened one), and any other prompt tail is left untouched.

Without it the model reasons anyway — `ReasoningStreamFilter` still keeps that out
of the client's `content`, but the tokens are spent and a short strict call can
truncate (`finish=length`). `enable_thinking:true` returns the prompt unchanged.

## OpenAI-spec field presence

llm-comply's pinned spec marks `logprobs` (per choice), `refusal` and `content`
(per message) required, and `model` required on input.

- The server encodes `logprobs` / `refusal` / `content` as **explicit null** when
  absent (present, never fabricated — see the `encode(to:)` override in
  `ChatMessage` / `ChatCompletionChoice`; Swift would omit a nil optional).
- A missing or empty `model` is rejected with a 400 via `ChatRequestSpec`.
- Do not remove these: the on-device `llm-compliance-test` skill (running
  `llm-comply openai-chat` against a deployed build) gates on them. Regression
  shows as `chat-error-handling` failing (200 instead of 400) or content tests
  failing on missing fields (e.g. `chat-tool-calling` fails on a missing
  `content` when a tool-call turn drops it).
- The streaming path (`StreamingChoice` / `Delta`) needs no such fields and
  passes as-is.

## Endpoints

| Route | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming and non-streaming |
| `POST /v1/completions` | legacy completions |
| `GET /v1/models` | model list |
| `GET /v1/logs`, `GET /logs` | see `docs/LOGGING.md` |
| `GET /v1/updates`, `PUT /v1/updates/ipa?build=<sha>` | self-upgrade, see `docs/OPERATIONS.md` |

The update routes exist **only while a model is loaded** — they ride the
inference listener (no second port, no second permission prompt).
