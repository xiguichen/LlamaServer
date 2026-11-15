# On-device benchmark

Read this when measuring prefill/generation on the phone, comparing builds or
models, or adding a row to [`PERFORMANCE.md`](PERFORMANCE.md).

`scripts/bench.py` drives the running server, reads the engine's own log lines,
and reports prefill and generation at several prompt sizes. It is stdlib-only
and bypasses HTTP proxies because the device is on the LAN.

## Before running

- **LlamaServer is in the foreground.** iOS suspends the server when the app is
  backgrounded; that looks like `Connection refused` or a dropped request
  mid-run. The app's keep-alive is not a guarantee.
- The phone is on the same Wi-Fi. The default URL is
  `http://192.168.3.3:8443` (see [`OPERATIONS.md`](OPERATIONS.md)); override it
  with `--url`.
- **No other client is generating.** Inference is serialized on one queue, so a
  second request contaminates the measurement.
- The model is loaded with the intended settings. `contextSize`, `useMtp`,
  `kvCacheType` and `longContextYarn` reset on every install/relaunch
  (`OPERATIONS.md`); confirm the engine block the script prints before trusting
  a comparison.

## Quick start

```bash
python3 scripts/bench.py
```

Defaults: prompt targets of **200, 8,000, 16,000, 32,000** tokens, **128**
generated tokens, `temperature: 0`, one run per target. The script prints the
loaded engine first, then one line per case, then a Markdown table, and writes
JSON to `.superpowers/bench/results/<time>-<model>.json`.

Common forms:

```bash
# Longer sweep; the last case must still fit the effective context.
python3 scripts/bench.py --contexts 200,8000,16000,32000,48000

# Three repeats per target; the summary adds the median (thermal noise).
python3 scripts/bench.py --repeats 3 --label mtp-on

# Show the engine config and the planned cases without sending requests.
python3 scripts/bench.py --dry-run

# Verify the log parsers without a device.
python3 scripts/bench.py --selftest
```

## MTP detection

The script reads the newest model-load block from the inference log and
classifies the decode path before running a case:

| `creating context … mtp=` | following log line | decode path | stats parsed |
|---|---|---|---|
| `true` | `MTP enabled (two-context speculative, …)` | speculative | `MTP stats:` |
| `true` | `MTP unavailable (two-context speculative, …)` | plain fallback | `plain stats:` |
| `false` | (none) | plain | `plain stats:` |
| unknown | (none) | inferred from the case | whichever line appears |

`MTP unavailable` means the draft context could not be set up — the GGUF has no
usable NextN head, or setup failed. It is not a configuration error, and the
run is still a valid measurement of the plain path. For example, the 9B
`Qwopus3.5-9B-v3.Q4_K_M.gguf` loads with `mtp=true` but logs `MTP unavailable`,
so its numbers come from `plain stats` (per-step), not `MTP stats` (per-round).

The script does **not** toggle MTP: it is an app setting that requires a model
reload. To A/B it, run once with MTP on and once off, keep `--contexts` and
`--repeats` identical, alternate the order, and label the two JSON files.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `http://192.168.3.3:8443` | device base URL |
| `--contexts` | `200,8000,16000,32000` | target prompt tokens per case |
| `--max-tokens` | `128` | tokens generated per case |
| `--repeats` | `1` | runs per target, each with a unique prompt |
| `--label` | empty | free-form tag stored in the JSON |
| `--out` | `.superpowers/bench/results/<time>-<model>.json` | JSON path |
| `--timeout` | `1800` | per-request timeout in seconds |
| `--log-tail` | `12000` | inference log lines fetched per case |
| `--dry-run` | off | print config and plan, send nothing |
| `--selftest` | off | run offline parser checks and exit |

## Fields

- `target` is what was requested; **`prompt` is the actual token count** from
  the engine. The prompt builder calibrates characters/token from earlier cases,
  but the actual count is the number to quote.
- `prefill_s` / `prefill_tps` cover the prompt decode (including the snapshot
  save, excluding the restore), and the rate is over **decoded, not reused**
  tokens (see `PERFORMANCE.md` → *Log-line semantics*).
- `gen_s` / `gen_tps` start at the first sampled token, so they exclude the
  prompt tail and the HTTP round trip.
- `path` is `mtp` or `plain`; the stats columns follow it:
  - MTP: `draft-gpu`, `verify-gpu`, `wall` are milliseconds per speculative
    **round**; `depth` is accepted tokens/round.
  - plain: `gpu-wait`, `wall` are milliseconds per **step**.
- `finish=length` is expected when the answer reaches `--max-tokens`; it does
  not invalidate the generation rate.

## Measurement rules

These are the rules from [`PERFORMANCE.md`](PERFORMANCE.md) → *How to measure*;
the script cannot enforce them:

1. One client at a time.
2. Quote the context with every t/s figure.
3. Thermals swing results by ~30%: back-to-back runs at the same context are not
   comparable to a run after an idle period. Use `--repeats` and report medians.
4. A/B comparisons must be paired: same prompts, same order policy, one engine
   restart between configs, and both runs on the same day.
5. A run containing a GPU reset (`llama_decode failed`, `backend is in error
   state`, `kIOGPUCommandBufferCallbackErrorInnocentVictim`) is not a
   measurement. The script stops when one appears in the recent log window.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `cannot reach … Connection refused` | app is backgrounded/closed, wrong IP/port, or phone off Wi-Fi |
| `RemoteDisconnected` mid-case | iOS suspended the app, or it crashed; partial JSON is kept |
| `no 'generated …' line in the log tail` | raise `--log-tail`; the log rotated or the request did not finish |
| `no 'loading model' line` | log rotated past the load; reload the model or raise `--log-tail` |
| `MTP unavailable` | model has no usable NextN head; plain-path result is valid |
| config changed mid-run | app restarted/reloaded; the script aborts rather than mixing configs |

The script bypasses `http_proxy`/`all_proxy` deliberately. If a manual `curl`
fails while the script works, the proxy is the difference — add
`curl --noproxy '*'`.

## Adding a result to PERFORMANCE.md

Record the model and build, effective context, KV type and bytes/token, MTP
state, prompt sizes, and run date. Use the actual `prompt` tokens, not the
targets. Add the table to the matching section and state anything that could
not be controlled (thermal state, order, single client).
