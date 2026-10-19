# Logging

Read this when adding log lines, changing the in-app log panel, or fetching logs
off the device.

## Routing

`FileLogger` routes each line to one of three files by
`LogRouter.category(level:sourceFile:)`:

| File | Contents |
|---|---|
| `llamaserver.log` | server/HTTP lifecycle, everything not covered below |
| `llamaserver-inference.log` | anything logged from `LlamaInference.swift` (engine internals) |
| `llamaserver-payloads.log` | `.verbose` payload dumps (prompts, responses) |

Each file rotates at 1 MB to `<name>.1.log`. The `file: String = #file` default on
every log method supplies the source, so call sites need no category argument.

## Levels and durability

- **error/warn** are written **and fsynced synchronously** — crash breadcrumbs
  survive aborts (including an `abort`/`ggml_abort`).
- **info/debug/verbose** are buffered on the logger queue (fsync ≤ 1 s / 64 KB), so
  the inference path never blocks on I/O.
- NSLog fires for warn+ only.
- The on-screen mirror delivers ≤ 10 Hz batches and carries **basic
  (error/warn/info) lines only**.
- Set `minimumLevel` from UI `.verbose` to capture full request/response payloads.
- `llama_log_set` routes llama.cpp native logs into FileLogger at `.debug` level.

## In-app panel budget

The panel renders **basic (error/warn/info) lines only** (`LogPanelSpec`),
truncated to 500 chars, in a `LazyVStack`.

Do not route payload/debug lines back to the panel or raise the cap without
checking the cost: a `cpu_resource` report (bug_type 202, 57% CPU) traced the
app's heaviest stack to the panel laying out 72k-char prompt lines with
`.textSelection(.enabled)` at 10 Hz.

## Fetching logs over HTTP

`GET /logs` (also `/v1/logs`) returns the three files so a LAN client can pull
them without the Files app:

```bash
curl 'http://<device>:8443/logs?name=inference&tail=500'
curl 'http://<device>:8443/logs?name=server&tail=400'
curl 'http://<device>:8443/logs?name=payloads&tail=200'
curl 'http://<device>:8443/logs?name=server&tail=400&backups=0&format=json'
```

Query: `name` (server / inference / payloads), `tail`, `backups` (include the
rotated `.1` file), `format` (text or json). The contract is the pure
`LogFetchSpec`; `FileLogger.snapshot(_:)` flushes buffered lines first, then reads
backup+live per category.

It returns whatever the files hold, so with "log prompt & response content" on it
exposes prompts and completions (the payload file) to anyone on the LAN; that
toggle is off by default. Each fetch logs one `GET /logs …` line to the server
file.
