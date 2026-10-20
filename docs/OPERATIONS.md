# Operating the device build

Read this when debugging a deployed build, uploading an IPA, or running on-device
compliance tests. The app runs on device, so behaviour is diagnosed from its
logs — there is no debugger attached and no local build to fall back on.

## Identify the build before concluding anything

Every build reports itself:

- `system_fingerprint` on any chat/completions response is `llamaserver-<sha>`
- the model-load line says
  `Model loaded (context …, build llamaserver-<sha>)`
  (`LlamaServerBuildSHA`, injected by CI)

A stale artifact and the current one are otherwise indistinguishable — this cost
two redeploys on 2026-10-17. The build id lives in `latest.json` / the log and
**not** in the file name (Files sorts by name, so a hash-suffixed name hides the
newest build).

## Logs

- **Pull them over HTTP** (no cable, no Files app): `curl 'http://<device>:8443/logs?name=inference&tail=500'`.
  Full query and caveats: `docs/LOGGING.md`.
- **The in-app panel is not the log.** It renders basic (error/warn/info) lines
  only, capped and truncated (`LogPanelSpec`); the three on-disk files are
  authoritative.
- **Prompt/response payloads are opt-in**: they land in the payloads file only
  while "log prompt & response content" is enabled (which needs log level
  *Verbose*). Missing prompts are a toggle, not a bug.

Store fetched logs on the filesystem before analysing them (e.g.
`.superpowers/logs/<date>-<topic>/`), so a conclusion can be re-checked.

## Deploying a new build (LAN push, not a cable)

`scripts/upload-ipa.sh --url http://<device>:8443` downloads the latest green
`build.yml` artifact with `gh` and `PUT`s it to
`GET|PUT /v1/updates[/ipa?build=<sha>]`. The app stores it in
`Documents/updates/` (`LlamaServer-unsigned.ipa`, rotated
`LlamaServer-unsigned.prev.ipa`, `latest.json`); the user taps the `.ipa` in Files
and AltStore installs it in place (models survive). Full runbook: the
`upload-ipa` skill. A healthy run is ~15 s (artifact download ~11 s, 4 MB over
Wi-Fi <1 s). If the phone is unreachable the script fails in ~5 s instead of
hanging — foreground the app, since iOS suspends it in the background.

Two things to hold onto:

- **The routes exist only while a model is loaded** — they ride the inference
  listener (no second port, no second permission prompt). A build that cannot
  start a model cannot be replaced this way; pick a smaller model in the UI as
  the escape hatch.
- **Validation runs before any filesystem call** (`UpdateSpec.rejection`), so a
  refused upload — no `?build=`, over the 128 MB cap, no zip header — leaves both
  slots untouched. The cap is best-effort (Telegraph buffers the whole body
  before the handler runs; the real IPA is ~4 MB), and uploads are serialized by
  their own lock, never the inference queue.

AltStore never uninstalls the app; it installs over it. It deletes only *the file
it was handed*, and which file that is depends on the entry point (a deep-link
passes our URL → deleted; `UIDocumentPickerViewController(asCopy: true)` gives it
an Inbox copy → ours survives).

## Engine settings are not persisted

`contextSize`, `useMtp`, `kvCacheType`, `longContextYarn` are launch defaults in
`ServerViewModel` (no `UserDefaults`/`@AppStorage`). **Every install and relaunch
reverts them.** After an upload, re-select the intended configuration before
drawing conclusions from a log — e.g. a build that shows
`Context reduced to 160512 tokens by the device memory budget (requested 262144)`
with `KV cache q8_0` was simply started on defaults.

Measured: **KV q4_0 reaches 262,144** on this device; q8_0 clamps to ~160,512.

## On-device compliance

The `llm-compliance-test` skill runs `llm-comply openai-chat` against a deployed
build. The field-presence rules it gates on are in `docs/API.md`.
