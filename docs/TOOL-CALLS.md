# Tool calls and prompt rendering

Read this when touching `ToolPrompt.swift`, `ToolCallParser.swift`,
`ThinkingSwitch.swift`, prompt rendering in `LlamaInference.formatPrompt`, or
when a turn fails to call a tool.

## Prompt rendering

**The model's own GGUF template renders; our fallback only backstops it.**
`LlamaInference.formatPrompt` asks the vendored `llama_chat_apply_template`
first, and for this model (`qwen35`) it **succeeds on device** — verified
2026-10-17 from a device log: `chat template loaded (7992 chars), tool
format=qwenXML`, and not one `chat template: … failed …` line on any request.

(The template's `multi_step_tool` scan iterates `messages[::-1]`, which the
bundled `minja` cannot do, but that branch only runs when a `tools` list is
passed to the template — something this C API cannot do — so in practice it
never declines. A jinja2 render done *with* tools shows the thinking scaffold;
the device's template render does not.)

What it produces is the plain ChatML layout `fallbackPrompt` would also produce
(matching the logged `FULL PROMPT`), so the fallback is a backstop for another
model or a future API change and **must stay byte-compatible with that render**:

```
<|im_start|>role\n{promptText}<|im_end|>\n      per message
<|im_start|>assistant\n                          bare primer
```

Two details are load-bearing:

- **`|trim` applies to the message content before the call envelope is
  appended.** `ChatMessage.promptText(format:)` trims the prose first; otherwise
  pi's replayed prose (which ends `\n\n`) put four newlines in front of
  `<tool_call>` instead of two.
- **`<tool_response>` user turns are rewritten** by
  `ToolPrompt.normalizeToolResults` (the template's `tool`-role branch is what we
  emulate).

`formatPrompt` logs which renderer ran (`chat template: llama_chat_apply_template
failed …` / `chat template: model has no GGUF template …`). If those lines
appear, the fallback is live and the prompt shape has changed — treat it as a
signal, not noise.

## Tool calls

Implemented via **system prompt injection** (`ToolPrompt.inject`, a pure tested
builder), not GBNF grammar.

`ToolWireFormat.detect(template:)` reads the model's GGUF chat template
(`LlamaInference.chatTemplateString`, set at load) and picks the **model's own**
wire format:

- **`qwenXML`** — a Qwen3.5 template (which carries `<function=…>` /
  `<parameter=…>` literally): schemas in a `# Tools … <tools>{json}</tools>`
  system block, calls as:

  ```
  <tool_call>
  <function=NAME>
  <parameter=K>
  V
  </parameter>
  </function>
  </tool_call>
  ```

- **`jsonHermes`** — anything else:
  `<tool_call>{"name":…,"arguments":{…}}</tool_call>`

This matters: injecting the JSON format to a Qwen3.5 model puts it out of
distribution and it answers with prose instead of a call (verified on device —
the native format calls deterministically).

The vendored C API (`llama_chat_apply_template`) has **no `tools` parameter** and
the framework exports no `common_chat_*` symbols, so llama.cpp's template-driven
rendering is unavailable; we emulate the template's `tools` branch instead.

`inject` **prepends** the block to the leading system message (stable every turn,
so KV-prefix reuse is preserved) and, for `qwenXML`, rewrites `role:"tool"` turns
into the `<tool_response>` user turn the template produces
(`normalizeToolResults`) — the generic template has no tool role. There is **no
trailing reminder** any more (the old one was a workaround for the wrong format).

### Parsing and replay

`ToolCallParser.parse()` extracts calls from **both** syntaxes (JSON and XML,
including an unclosed XML call) post-generation, typing XML parameter values
against the tool schema (`parse(_:tools:)`: string-typed params stay strings,
everything else is JSON-normalized, matching llama.cpp's schema-aware Qwen parser
in `common/chat-peg-parser.cpp`).

On the next request, `ChatMessage.promptText(format:)` must replay an assistant
turn's prose `content` **and** its `tool_calls` (rendered via
`ToolPrompt.toolEnvelope`) into the prompt. Dropping the envelope when prose is
present removes the only "prose + tool call" example from the history and trains
the model to announce a tool and then stop without calling it.

### Argument serialization

`ToolPrompt.orderedArguments` renders each argument value with
`JSONSerialization.data(withJSONObject:options:[.fragmentsAllowed])`.

`.fragmentsAllowed` is **load-bearing**, not tidiness: without it a scalar
argument (number/bool/null) is a JSON *fragment* and the call raises an
**uncatchable ObjC exception** (`Invalid top-level type in JSON write`) instead
of returning an error — `try?` cannot catch it, so it terminates the process
(`_objc_terminate` → `ggml_uncaught_exception` → `abort`) on the serial inference
queue. A model `read` call with `{"path":…,"limit":150,"offset":200}` aborted the
server the moment pi replayed that turn (device crash 2026-10-18 11:58:53; the
log's last line is the *request* that replayed it —
`streaming #13 … async block started` — because the crash is in `formatPrompt`
before any output).

Every other `dataWithJSONObject` call site is safe by construction:
`ToolCallParser.parse` guards with `isValidJSONObject` (which returns false for
fragments) and the other two pass container roots — keep one of those two
properties when adding a call site.

## Streaming filters

A tool turn runs each decoded piece through `ReasoningStreamFilter` **then**
`ToolStreamFilter` (both in `ToolCallParser.swift`), so a `<think>` block emitted
on a tool turn is stripped rather than streamed. It used to bypass the reasoning
filter, leak the chain-of-thought as `content`, and — via
`didStreamContentDuringToolGen` — suppress the cleaned content.

`ToolStreamFilter` streams prose outside `<tool_call>…</tool_call>` and buffers
the envelope; envelope tags are matched as whole literals (assumes the tokenizer
emits `<tool_call>` / `</tool_call>` as single tokens).

`<think>` blocks are stripped from streamed content by `ReasoningStreamFilter`
(character-by-character state machine, NSRegularExpression for final parse).

## Prompt dilution: long prompts suppress tool calls

**Measured 2026-10-19 on device** (build `b5502e5`, Qwen3.5-4B, thinking off).
A 21,234-token prompt with 17 tools produced 16 tokens of prose and
`finish=eog` — no tool call:

```
Using using-superpowers to confirm the MTP depth discrepancy before implementing anything.
```

The parser ran (`tools active, parsing tool calls`); there was nothing to parse.
The trigger is **prompt size**, not the tool machinery:

| Variation | Result |
|---|---|
| Minimal prompt + one `bash` tool | calls `{"command":"ls"}` |
| 17 tools + tiny system prompt | calls `web_search` |
| Exact request (52 KB system message, 21.2k tokens) | announcement, `stop` |
| Same, `enable_thinking:true` | longer prose plan, `stop` |
| AGENTS.md demoted to a user message | prose, `stop` |
| "Begin with a tool call" at the **end of the system message** | prose, `stop` |
| System message truncated to 8,000 chars | calls a tool |
| "Use the tools now, don't announce" **after** the skill text in the last user message | calls a tool |

What is happening: the last instruction in the prompt is the agent skill's
"announce 'Using [skill] to [purpose]'", and the model performs the announcement
and ends the turn. At 21k tokens it cannot override that; at 8k it can.

Practical rules:

- **Keep the system prompt short.** This is why `AGENTS.md` is a map and detail
  lives in `docs/` (loaded on demand).
- A behavioural nudge placed **after** the competing instruction (at the very end
  of the last user message) restores the call; the same nudge earlier does not.
- Repeating the `<tools>` block near the question does **not** help; the
  instruction's position is what matters.
