# CI configuration

Read this when debugging CI, setting up runs, or understanding test & build jobs.

## CI runner & env

All jobs run on `xcode-27` (macOS 27 → iOS 27 SDK).

## Build & test

- CI triggers on `push` to `main` or `master`, and on `pull_request`.
- Manual runs: `workflow_dispatch` input `llama_tag` (default: `b11557`).

### Build job

1. Checkout repo.
2. Select latest-stable Xcode.
3. Install `xcodegen` via `brew`.
4. Cache `vendor/llama.xcframework` keyed by tag + `fetch-llama-xcframework.sh` hash.
5. Fetch if miss.
6. Run `xcodegen generate` (project defined in `project.yml`; never edit `.xcodeproj` directly).
7. `xcodebuild -resolvePackageDependencies`.
8. Build Release for `iphoneos`, unsigned (`CODE_SIGNING_ALLOWED=NO`).
9. Package with `scripts/package-ipa.sh` (ad-hoc signature + entitlements).
10. Upload artifact `LlamaServer-unsigned-ipa`.

### Test job

1. Checkout repo.
2. Select latest-stable Xcode.
3. Install `xcodegen` via `brew`.
4. Run `xcodegen generate`.
5. `xcodebuild test`:
   - `-project LlamaServer.xcodeproj`
   - `-scheme LlamaServerTests`
   - `-sdk iphonesimulator`
   - `-destination 'platform=iOS Simulator,name=iPhone 17'`
   - `-derivedDataPath build`
   - `CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO ONLY_ACTIVE_ARCH=NO`

The test job must pass before the build job runs.

## CI notes

- **No `Package.swift`.** Project is defined in `project.yml` (XcodeGen). Always edit `project.yml`; CI regenerates it and it is never committed.
- `vendor/llama.xcframework` is prebuilt, fetched by `bash scripts/fetch-llama-xcframework.sh`, pinned to tag `b11557`. There is no local llama.cpp compilation.
- The framework download is cached across runs by release tag + script hash.

## Local development

Local machine is CLT-only (no XCTest). Use `xcodegen generate` for local syntax / project validation; CI runs the tests.