# Running a small model on this project

Read this when using `qwen3.5:4b` (pi's default) or when an agent stalls,
announces a tool and stops, or ignores `AGENTS.md`.

`docs/TOOL-CALLS.md` has the measurement: a 21k-token prompt makes the 4B emit
prose and stop; an 8k prompt still calls a tool. Every setting below exists to
keep the prompt small and remove competing instructions.

## Launch profile

```bash
scripts/pi4b.sh
```

That is `pi` with:

| Flag | Why |
|---|---|
| `--tools read,bash,edit,write` | Four tools this repo needs. Removes web/notes/meeting/history tools the model used to avoid reading the repo. |
| `--no-skills` | Removes superpowers skill text. Its "announce Using [skill]" line is what produced the `Using …` → stop turns. |
| `--no-extensions` | Removes `WROTE_ENTRY`, `notes_*`, `history_*`, `meeting_*`, and the extra tool declarations. |
| `-e .pi/extensions/project-guard.ts` | Explicit load (still works under `--no-extensions`): blocks local `xcodebuild`, `curl api.github.com`, and a back-to-back identical `bash` command. |
| `--thinking off` | `docs/TOOL-CALLS.md` measured that thinking on at 21k still ends in prose, with longer output. |

Do not invoke `/skill:*` on the 4B. If a skill is needed, read its `SKILL.md`
with the `read` tool, or run the task on a larger model.

## Model settings

`~/.pi/agent/models.json`, under provider `llamacpp`:

```json
{
  "id": "qwen3.5:4b",
  "contextWindow": 32768,
  "maxTokens": 2048,
  "samplingParams": {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.05
  }
}
```

`contextWindow` is pi's working budget, not the server's. The app can still
serve 256k; a 256k client budget just lets sessions grow into the zone where
tool calls stop. 32k makes compaction happen around 16k.

`~/.pi/agent/settings.json`:

```json
{
  "compaction": {
    "modelOverrides": {
      "llamacpp/qwen3.5:4b": {
        "reserveTokens": 16384,
        "keepRecentTokens": 6000
      }
    }
  }
}
```

If the model still announces instead of acting at ~16k, lower its
`contextWindow` to `24576`.

## Task routing

Use the 4B for:

- one-file edits with a specific failing line or error
- renames, formatting, small mechanical refactors
- applying a diff that is already worked out

Use a larger model (`deepseek-v4-pro`) for:

- MTP/performance analysis and measurement
- multi-file changes
- crash/Jetsam root-cause work
- anything where the answer is an architecture decision

## Session hygiene

- One task per session. Start a new session after ~10–15 turns.
- Keep a `todo.md` in the repo when a task has several steps; the model will not
  remember them across a long context.
- After two failed attempts at the same fix, revert the file and stop. Do not
  let it try a third blind fix.

## Verify, don't trust

```bash
bash scripts/check-swift.sh <changed-file>   # ~2 s per file
bash scripts/check-swift.sh                  # all Sources/App/*.swift, ~10 s
bash scripts/install-hooks.sh                # pre-commit gate, once per clone
bash scripts/ci-wait.sh <run-id>             # wait for CI, prints failure log
bash scripts/ci-logs.sh <run-id>             # failed log only
```

CI is the only build/test authority (`docs/CI.md`). Never `curl` GitHub and
never run `xcodebuild` locally.

## What to watch

A working 4B session has:

- a tool call in the first assistant turn
- `swiftc -parse` output after every edit
- `gh run view --log-failed` output before any "fixed" claim
- no repeated identical `bash` command

If those four are missing, stop the session and restart with the profile.
