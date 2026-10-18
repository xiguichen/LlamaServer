---
name: upload-ipa
description: Use when a CI-built IPA needs to get onto the iPhone — deploying a fresh build, replacing a stale or broken one, or verifying that the self-upgrade upload route works. Pushes the latest GitHub Actions artifact to the phone over the LAN, where it is tapped in the Files app and installed in place by AltStore.
---

# Push a CI build onto the phone (self-upgrade)

## Overview

The app serves two routes on **the same HTTP listener as inference**:

| Method | Path | Meaning |
|--------|------|---------|
| `GET`  | `/v1/updates` | What IPA builds are stored on the phone |
| `PUT`  | `/v1/updates/ipa?build=<sha>` | Store the request body as the current build |

An uploaded IPA lands in `Documents/updates/` inside the app container, which
`UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace` (project.yml) expose in
the **Files** app. The flow is: script pushes → user opens Files → taps the `.ipa` →
AltStore installs it **in place** (the container, and therefore your models, survive).
No cable, no AirDrop, no AltServer, no Files-app shuffle from a Mac.

Running it is one command:

```bash
scripts/upload-ipa.sh --url http://<iphone-ip>:8443
```

## When to Use

- A CI run just went green and you want that build on the phone.
- The deployed build is stale or behaving badly and you want to replace it without
  a cable.
- After changing `LlamaHTTPServer` / `UpdateSpec` / `UpdateStore`, to confirm the
  upload surface still works on device.

Do NOT use for: pulling logs off the device (that is `GET /logs`, see AGENTS.md),
or for a first-ever install (nothing is on the phone to receive the upload).

## Prerequisites

- `gh` authenticated (`gh auth status`) for the download path; `curl` and `unzip`.
- The phone on the same LAN, and **the server running in the app**.
- The deployed build must already have these routes. They are new — a build from
  before them answers `404`.

## The gap this route cannot close

The update route rides the inference listener, which the app only starts **after a
model loads successfully**. So:

| State | What an upload gets |
|-------|---------------------|
| Server running (green in the app) | works |
| Engine stopped / no model loaded | connection refused |
| A model that will not fit or load | connection refused — **the routes never came up** |

The last row is the important one: the case where you most want to replace the
build is the case where the app cannot receive one. The escape hatch is on the
phone itself — pick a smaller model (or a different one) in the UI so the engine
starts, then push the build. There is no route around this: an iOS app has no
always-on listener without the app running.

## The Command

```bash
# Latest successful CI build on main -> phone
scripts/upload-ipa.sh --url http://192.168.3.3:8443

# A specific CI run
scripts/upload-ipa.sh --url http://192.168.3.3:8443 --run 39124706228

# A local build (skips gh entirely); records git HEAD short SHA unless --build
scripts/upload-ipa.sh --url http://192.168.3.3:8443 --file build/LlamaServer-unsigned.ipa

# Explicit build id
scripts/upload-ipa.sh --url http://192.168.3.3:8443 --build c13a8bd
```

`--url` (or `$LLAMASERVER_URL`) is required — no device address is baked into the
repo, because a hard-coded IP goes stale on the next DHCP lease.

## What Success Looks Like

```
==> Resolving the latest successful CI build
==> Downloading LlamaServer-unsigned-ipa from run 39124706228 (build 237d5ab)
    /var/.../LlamaServer-unsigned.ipa (4133108 bytes)
==> Uploading to http://192.168.3.3:8443/v1/updates/ipa?build=237d5ab
{"current":{"build":"237d5ab","bytes":4133108,"fileName":"LlamaServer-unsigned.ipa",
            "modifiedAt":"2026-10-18T…Z","present":true},
 "previous":{…},"uploadedAt":"2026-10-18T…Z"}
==> On the phone now:
{…}
```

Check the byte count against the artifact (`gh run view <id>` → the artifact size,
or the local file size). The phone also logs it — one line each, in the **server**
file:

```
PUT /v1/updates/ipa build=237d5ab 4133108 bytes -> LlamaServer-unsigned.ipa (replaced nothing)
GET /v1/updates current=LlamaServer-unsigned.ipa 4133108B build=237d5ab previous=none uploaded=2026-10-18T…
```

```bash
curl -s 'http://<iphone-ip>:8443/logs?name=server&tail=40' | grep updates
```

## The Tap (the one manual step)

1. Open **Files** on the iPhone.
2. **On My iPhone → LlamaServer → updates**.
3. Tap **`LlamaServer-unsigned.ipa`** → AltStore offers to install it.
4. The install is **in place**: the container is preserved, so models, logs and the
   stored IPAs stay.

> An uninstall (not an upgrade) deletes the whole container — models, logs and the
> stored IPA go with it. iOS gives the app no way around that.

## Rollback

Every upload rotates the build it replaces to **`LlamaServer-unsigned.prev.ipa`**
(one slot, overwritten by the next upload). If the new build is broken on device,
install the `.prev` file the same way — tap it in Files.

Note the container caveat above: if a broken build somehow requires an uninstall,
the `.prev` copy goes with it, and recovery means a cable/AltStore from a Mac.

## Why the file name carries no build hash

`LlamaServer-unsigned.ipa` keeps the name CI gives it. Files sorts by name, so a
hash-suffixed name (`LlamaServer-unsigned-237d5ab.ipa`) would leave the newest build
impossible to spot among older ones — you would have to work out which hash is newer
before deciding what to tap. The build identity lives in `latest.json` on the phone,
in `GET /v1/updates`, and in the log line instead.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `could not reach http://…` | engine stopped, or phone off the LAN, or wrong IP | start the server in the app; check the address shown in the app |
| `404` on the PUT | deployed build predates these routes | upload this build via AltStore from a Mac once; check `system_fingerprint` / the `Model loaded … build llamaserver-<sha>` line |
| `400` `missing ?build=` | `--build` empty and no run resolved | pass `--build <sha>` |
| `400` `body is not an IPA` | the download saved an error page, not a zip | `gh run view <id>`; the device refuses the file rather than replacing a working build |
| `413` | over the 128 MB cap | nothing today is near it (~4 MB); a genuinely huge upload needs the cap raised in `UpdateSpec` |
| `500 could not store the IPA` | the phone refused the write (disk) | free space; the previous build is still in `.prev` |
| Script says "has no Payload/" | the artifact is not an IPA | wrong artifact name, or a truncated download — re-run |
| Upload succeeds, install fails | signing/expiry on the device side | that is AltStore, not this route |

## Verified Ground Truth

Checked on device / locally, do not re-derive:

| Fact | Value |
|------|-------|
| CI artifact name | `LlamaServer-unsigned-ipa`, containing `LlamaServer-unsigned.ipa` |
| Workflow | `.github/workflows/build.yml`, branch `main` |
| IPA size | ~4.13 MB (`4133108` bytes at build `237d5ab`) — well under the 8 MB chat-body cap, so no timeout tuning needed |
| Storage on the phone | `Documents/updates/`: current IPA, `LlamaServer-unsigned.prev.ipa`, `latest.json` |
| Upload cap | 128 MB (`UpdateSpec.maxUploadBytes`), refused with 413 |
| Body buffering | Telegraph hands the handler a fully buffered body, so the cap *refuses* an oversized upload but cannot prevent it being read into memory — the cap is a guard, not a defence |
| Concurrency | uploads are serialized by their own lock, never the inference queue (an upload does not wait for a generation) |
| Refusal is side-effect-free | validation happens before any filesystem call, so a refused upload leaves both slots untouched |
