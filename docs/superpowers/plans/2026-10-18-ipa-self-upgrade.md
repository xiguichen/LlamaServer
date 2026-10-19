# IPA Self-Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a LAN client push the CI-built unsigned IPA onto the phone, so a new build can be installed from the Files app without a cable, AirDrop, or AirDrop-shaped round trip through a Mac.

**Architecture:** The **existing** inference server grows two routes: `PUT /v1/updates/ipa?build=<sha>` stores the body in `Documents/updates/`, and `GET /v1/updates` reports what is stored. No second listener, no auth, no new UI — the file is written where `UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace` already expose it in the Files app, and the user taps it to hand it to AltStore. A Mac-side script (`scripts/upload-ipa.sh`) pulls the latest CI artifact with `gh` and PUTs it; a project skill documents the runbook for an agent.

**Tech Stack:** Swift (Foundation only for the pure spec), Telegraph (`HTTPMethod.PUT`, `HTTPStatus.created`/`.payloadTooLarge`), `FileLogger`, bash + `gh` + `curl` + `unzip` for the script, XCTest for the pure spec.

**Spec:** the approved design in this session — contract, storage, guards, `UploadSpec` → renamed `UpdateSpec`, `scripts/upload-ipa.sh`, `skills/upload-ipa/SKILL.md`, docs, verification. (Design decisions, not re-derived here: which routes, why the inference server, why the file name carries no hash.)

## Global Constraints

- **The pure spec is Foundation-only.** `Tests/LlamaServerTests` compiles only the files listed in `project.yml` (no Telegraph, no `llama`) — `UpdateSpec.swift` must not reference `HTTPStatus` or any Telegraph type, and must not use `FileManager` (paths are strings; the caller does I/O).
- `project.yml` is the only project definition: add `Sources/App/UpdateSpec.swift` to the `LlamaServerTests` sources list. The app target globs `Sources/App`, so it needs no entry there.
- **Fixed path, no auth:** `Documents/updates/`. File names: `LlamaServer-unsigned.ipa`, `LlamaServer-unsigned.prev.ipa`, `latest.json`. Build identity lives in `latest.json` and the log, **never in the file name** (Files sorts by name; a hash-suffixed name makes the newest build impossible to spot).
- Cap: `maxUploadBytes = 128 * 1024 * 1024` (the IPA is ~3.8 MB; this is room to grow, not a target).
- Shared mutable state uses `OSAllocatedUnfairLock`; no `#warning` — `FileLogger.shared.warn(...)` for recoverable issues.
- One log line per request, `FileLogger.shared.info(...)`, matching `GET /logs`'s style.
- Tests: pure XCTest in `Tests/LlamaServerTests/`; the local machine has no XCTest runner, so a task's immediate check is a `swiftc` harness over the pure file, and CI (`xcodebuild test`) is the final check. Commit per task, straight to `main`; CI must be green.
- Do **not** commit `.pi-meetings/`, `checkpoint.md`, `docs/superpowers/`.

## Review Focus

The inputs and failure modes the design implies but no task's happy path exercises, most likely to bite a user first:

1. **The server only exists while the model is loaded** (`ServerViewModel.start()` → load → `server.start()`). With no model loaded or the engine stopped, a PUT gets connection-refused, not a 404 — the script and the skill must say that instead of reporting a broken route. (Task 3 + Task 4.)
2. **A body that is not an IPA** — e.g. an HTML/JSON error page saved by a failed download — must be rejected **before** anything is rotated, or it silently replaces a good build. (Task 1 `notAnIPA`, Task 2 ordering: validate, then touch the filesystem.)
3. **A second upload** when `LlamaServer-unsigned.ipa` already exists: the old file must move to `.prev`, and the new one must be the full body (not truncated, not the old bytes). (Task 1 boundary at exactly `maxUploadBytes`; Task 2 rotation.)
4. **A failed/partial write during rotation**: the current slot must never be left empty while the only copy was destroyed. Write atomically; on failure report 500 and leave `.prev` holding the last good build. (Task 2.)
5. **Missing or empty `?build=`** → 400 and the stored files untouched (the build id is the whole reason the record exists). (Task 1 `missingBuild`, Task 2 ordering.)
6. **Two concurrent uploads** (script run twice) must not interleave moves into a lost file. (Task 2 lock.)

## Task 1: `UpdateSpec` — the pure contract

**Files:**
- Create: `Sources/App/UpdateSpec.swift`
- Test: `Tests/LlamaServerTests/UpdateSpecTests.swift`
- Modify: `project.yml` (add `Sources/App/UpdateSpec.swift` to the `LlamaServerTests` sources, after the `MemoryBudget.swift` entry at line 116)

**Interfaces:**
- Consumes: nothing (Foundation only).
- Produces (Task 2 and Task 3 depend on these exact names):
  - `UpdateSpec.locations(documentsPath: String) -> UpdateSpec.Locations` with `directory`, `current`, `previous`, `metadata: String`
  - `UpdateSpec.build(fromQueryItems: [URLQueryItem]) -> String?`
  - `UpdateSpec.rejection(build: String?, body: Data) -> UpdateSpec.Rejection?` (`nil` = accept)
  - `UpdateSpec.Rejection` = `.missingBuild` | `.tooLarge(bytes: Int)` | `.notAnIPA(bytes: Int)`, each with `message: String`
  - `UpdateSpec.Record` (Codable, Equatable): `build: String`, `bytes: Int`, `uploadedAt: Date`
  - `UpdateSpec.Status` (Codable, Equatable): `current: Slot`, `previous: Slot`, `uploadedAt: Date?`; `Slot` = `fileName: String`, `present: Bool`, `bytes: Int?`, `modifiedAt: Date?`, `build: String?`
  - `UpdateSpec.status(currentBytes:currentModified:previousBytes:previousModified:record:) -> Status`
  - `UpdateSpec.encode<T: Encodable>(_ value: T) throws -> Data`, `UpdateSpec.decodeRecord(_ data: Data) -> Record?` (ISO-8601 dates in both directions)
  - `UpdateSpec.maxUploadBytes: Int`, `UpdateSpec.zipMagic: [UInt8]`

- [ ] **Step 1: Write the failing tests**

Create `Tests/LlamaServerTests/UpdateSpecTests.swift`:

```swift
import XCTest

/// The `GET /v1/updates` + `PUT /v1/updates/ipa` contract.
///
/// The guards exist because the thing being overwritten is the *only* copy of a
/// working build: a rejected upload must leave both slots exactly as they were,
/// which is why rejection is decided purely, before any filesystem call.
final class UpdateSpecTests: XCTestCase {
    private func zip(_ extra: Int = 16) -> Data {
        Data(UpdateSpec.zipMagic) + Data(repeating: 0, count: extra)
    }

    // MARK: - Storage layout

    func testLocationsLiveInDocumentsUpdates() {
        let l = UpdateSpec.locations(documentsPath: "/var/mobile/Documents")
        XCTAssertEqual(l.directory, "/var/mobile/Documents/updates")
        XCTAssertEqual(l.current, "/var/mobile/Documents/updates/LlamaServer-unsigned.ipa")
        XCTAssertEqual(l.previous, "/var/mobile/Documents/updates/LlamaServer-unsigned.prev.ipa")
        XCTAssertEqual(l.metadata, "/var/mobile/Documents/updates/latest.json")
    }

    func testCurrentFileNameCarriesNoBuildHash() {
        // Files sorts by name; the newest build must be the one at the top, and
        // the build identity must come from latest.json instead.
        XCTAssertEqual(UpdateSpec.currentFileName, "LlamaServer-unsigned.ipa")
        XCTAssertFalse(UpdateSpec.currentFileName.contains("%"))
        XCTAssertEqual(UpdateSpec.previousFileName, "LlamaServer-unsigned.prev.ipa")
    }

    // MARK: - Query

    func testBuildReadsTheQueryCaseInsensitivelyAndTrims() {
        XCTAssertEqual(
            UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "BUILD", value: " 441aadc ")]),
            "441aadc")
        XCTAssertEqual(
            UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "other", value: "x"),
                                              URLQueryItem(name: "build", value: "abc1234")]),
            "abc1234")
    }

    func testBuildIsNilWhenAbsentOrEmpty() {
        XCTAssertNil(UpdateSpec.build(fromQueryItems: []))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: "")]))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: "   ")]))
        XCTAssertNil(UpdateSpec.build(fromQueryItems: [URLQueryItem(name: "build", value: nil)]))
    }

    // MARK: - Rejection

    func testMissingBuildIsRejectedFirst() {
        // Even a valid IPA is refused: the build id is the point of the record.
        XCTAssertEqual(UpdateSpec.rejection(build: nil, body: zip()), .missingBuild)
        XCTAssertEqual(UpdateSpec.rejection(build: "", body: zip()), .missingBuild)
        XCTAssertEqual(UpdateSpec.rejection(build: "   ", body: zip()), .missingBuild)
    }

    func testOversizedBodyIsRejected() {
        let body = Data(UpdateSpec.zipMagic) + Data(repeating: 0, count: UpdateSpec.maxUploadBytes)
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: body),
                       .tooLarge(bytes: body.count))
    }

    func testBodyExactlyAtTheCapIsAccepted() {
        // The boundary: > max is dead, == max is fine.
        let body = Data(UpdateSpec.zipMagic)
            + Data(repeating: 0, count: UpdateSpec.maxUploadBytes - UpdateSpec.zipMagic.count)
        XCTAssertEqual(body.count, UpdateSpec.maxUploadBytes)
        XCTAssertNil(UpdateSpec.rejection(build: "441aadc", body: body))
    }

    func testNonIPAIsRejected() {
        // A failed download saves an HTML error page; writing it would replace a
        // working build with something AltStore cannot install.
        let html = Data("<!DOCTYPE html><html>404</html>".utf8)
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: html),
                       .notAnIPA(bytes: html.count))
    }

    func testEmptyBodyIsRejectedAsNotAnIPA() {
        XCTAssertEqual(UpdateSpec.rejection(build: "441aadc", body: Data()), .notAnIPA(bytes: 0))
    }

    func testRealIPAIsAccepted() {
        XCTAssertNil(UpdateSpec.rejection(build: "441aadc", body: zip(4096)))
    }

    func testRejectionMessagesCarryTheNumbers() {
        XCTAssertTrue(UpdateSpec.Rejection.tooLarge(bytes: 999).message.contains("999"))
        XCTAssertTrue(UpdateSpec.Rejection.tooLarge(bytes: 999).message.contains("\(UpdateSpec.maxUploadBytes)"))
        XCTAssertTrue(UpdateSpec.Rejection.notAnIPA(bytes: 12).message.lowercased().contains("ipa"))
        XCTAssertTrue(UpdateSpec.Rejection.missingBuild.message.contains("build"))
    }

    // MARK: - Records and status

    func testRecordRoundTripsWithISO8601Dates() throws {
        let record = UpdateSpec.Record(build: "441aadc", bytes: 4083117,
                                       uploadedAt: Date(timeIntervalSince1970: 1_761_000_000))
        let data = try UpdateSpec.encode(record)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"build\":\"441aadc\""), text)
        XCTAssertTrue(text.contains("2026-"), text)   // ISO-8601, not 1761000000
        XCTAssertEqual(UpdateSpec.decodeRecord(data), record)
    }

    func testCorruptMetadataDecodesToNilInsteadOfThrowing() {
        XCTAssertNil(UpdateSpec.decodeRecord(Data("not json".utf8)))
    }

    func testStatusReportsBothSlotsAndFillsBuildFromMetadata() throws {
        let record = UpdateSpec.Record(build: "441aadc", bytes: 100, uploadedAt: Date())
        let status = UpdateSpec.status(currentBytes: 4083117,
                                       currentModified: Date(timeIntervalSince1970: 1_761_000_000),
                                       previousBytes: 4083116,
                                       previousModified: Date(timeIntervalSince1970: 1_760_000_000),
                                       record: record)
        XCTAssertTrue(status.current.present)
        XCTAssertEqual(status.current.bytes, 4083117)
        XCTAssertEqual(status.current.build, "441aadc")
        XCTAssertEqual(status.current.fileName, "LlamaServer-unsigned.ipa")
        XCTAssertTrue(status.previous.present)
        XCTAssertNil(status.previous.build, "only the current slot is described by latest.json")
        XCTAssertEqual(status.uploadedAt, record.uploadedAt)

        // And the whole body survives the encoder the handler will use.
        let data = try UpdateSpec.encode(status)
        let decoded = try JSONDecoder.iso8601.decode(UpdateSpec.Status.self, from: data)
        XCTAssertEqual(decoded.current.bytes, 4083117)
    }

    func testStatusReportsAnEmptySlotAsAbsent() throws {
        let status = UpdateSpec.status(currentBytes: nil, currentModified: nil,
                                       previousBytes: nil, previousModified: nil, record: nil)
        XCTAssertFalse(status.current.present)
        XCTAssertNil(status.current.bytes)
        XCTAssertNil(status.current.build)
        XCTAssertFalse(status.previous.present)
        XCTAssertNil(status.uploadedAt)
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

The local machine has no XCTest runner, so verify the *file compiles against nothing yet* by parsing it, and rely on CI for the real run:

```bash
cd /Users/chenxigui/develop/LlamaServer
swiftc -parse Tests/LlamaServerTests/UpdateSpecTests.swift   # parses fine — the symbols are what's missing
```

Expected: parse succeeds; the symbols it references (`UpdateSpec`, `JSONDecoder.iso8601`) don't exist yet, so the CI run in Step 5 would fail. (Do not add tests to `project.yml` — `Tests/LlamaServerTests` is already globbed.)

- [ ] **Step 3: Implement `UpdateSpec`**

Create `Sources/App/UpdateSpec.swift`. Doc comment must carry the decisions (this is the file a future reader opens to learn why the routes are here and why the file name has no hash), including: the server is the *existing* inference listener (no second port → no second permission prompt, no second failure mode, and no server that is down exactly when the build is broken); `Documents/updates/` is Files-visible because of the two Info.plist keys; the current IPA keeps CI's file name because Files sorts by name, and the build identity lives in `latest.json`/the log; the previous IPA is rotated to a fixed `.prev` name. Constants and signatures exactly as in the Interfaces block; `rejection(build:body:)` checks in this order — missing build, then `body.count > maxUploadBytes`, then `body.starts(with: zipMagic)` — and `message` values state the numbers (e.g. `"upload is \(bytes) bytes; the cap is \(maxUploadBytes)"`, `"body is not an IPA (no PK\\x03\\x04 zip header); \(bytes) bytes received"`, `"missing ?build=<sha> query parameter"`). `status(...)` fills `Slot.build` from `record?.build` for the current slot only, and `present` from `bytes != nil`. `encode`/`decodeRecord` use a `JSONEncoder`/`JSONDecoder` with `.iso8601` date strategy; `decodeRecord` returns `nil` on any throw.

- [ ] **Step 4: Run the local harness to verify the pure logic**

Write `/private/var/folders/qm/c_1hqwxd7mjcx8dss2bx166w0000gn/T/opencode/ab/upd/main.swift` mirroring the tests' assertions (`expect(...)`/`expectEqual(...)` like `ab/mb/main.swift`), then:

```bash
cd /private/var/folders/qm/c_1hqwxd7mjcx8dss2bx166w0000gn/T/opencode/ab/upd
cp /Users/chenxigui/develop/LlamaServer/Sources/App/UpdateSpec.swift .
swiftc -o run UpdateSpec.swift main.swift && ./run
```

Expected: `N/N checks passed`, exit 0.

- [ ] **Step 5: Run the tests in CI**

```bash
cd /Users/chenxigui/develop/LlamaServer
git add Sources/App/UpdateSpec.swift Tests/LlamaServerTests/UpdateSpecTests.swift project.yml
git commit -m "update spec: pure contract for GET /v1/updates + PUT /v1/updates/ipa"
git push origin main
gh run watch "$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

Expected: `test` ✅ (`UpdateSpecTests` green) and `build` ✅.

## Task 2: The routes and the file store

**Files:**
- Modify: `Sources/App/LlamaHTTPServer.swift` — register routes in `configureRoutes` (lines 332-339, next to `logs`); add `updatesResponse()`, `uploadIPAResponse(request:)`, `httpStatus(for:)`, `documentsPath` and the store lock after `logsResponse` (line 377)

**Interfaces:**
- Consumes: `UpdateSpec.*` exactly as produced by Task 1; `json(encodable:status:)` (line 1013) and `errorResponse(_:status:)` (line 1153).
- Produces: the HTTP surface `GET|PUT /v1/updates[/ipa]` for Task 3's script and Task 4's skill.

- [ ] **Step 1: Register the routes**

In `configureRoutes`, after the two `logs` registrations, mirroring the file's existing "bare + /v1" convention:

```swift
        // Self-upgrade: the CI-built IPA can be pushed onto the phone over the
        // LAN, then tapped in the Files app — no cable, no AirDrop, no AltServer.
        // See `UpdateSpec` for the storage layout and the guards.
        let updates: (HTTPRequest) -> HTTPResponse = { [weak self] _ in
            self?.updatesResponse() ?? HTTPResponse(.internalServerError)
        }
        server.route(.GET, "updates", updates)
        server.route(.GET, "v1/updates", updates)

        let upload: (HTTPRequest) -> HTTPResponse = { [weak self] request in
            self?.uploadIPAResponse(request: request) ?? HTTPResponse(.internalServerError)
        }
        server.route(.PUT, "updates/ipa", upload)
        server.route(.PUT, "v1/updates/ipa", upload)
```

- [ ] **Step 2: Implement the two handlers**

Add after `logsResponse`, with these exact shapes:

```swift
    /// Documents/ — the container the Files app shows (UIFileSharingEnabled +
    /// LSSupportsOpeningDocumentsInPlace), so `updates/` lands somewhere the user
    /// can tap the `.ipa` from.
    private var documentsPath: String {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
    }

    /// Serializes uploads against each other. Deliberately not the inference
    /// queue: uploading must not wait for a generation, and a generation must not
    /// wait for a 4 MB write.
    private static let uploadLock = OSAllocatedUnfairLock()

    /// `GET /v1/updates` — what is stored on the phone right now.
    private func updatesResponse() -> HTTPResponse

    /// `PUT /v1/updates/ipa?build=<sha>` — store the body as the current IPA,
    /// rotating the old one to `.prev`. Validates *before* touching the
    /// filesystem, writes the new body atomically, and never leaves the current
    /// slot missing: a write failure leaves `.prev` holding the last good build,
    /// and the response says so.
    private func uploadIPAResponse(request: HTTPRequest) -> HTTPResponse

    private func httpStatus(for rejection: UpdateSpec.Rejection) -> HTTPStatus  // .tooLarge → .payloadTooLarge, else .badRequest
```

`updatesResponse()`: read both slots' size/mtime with `FileManager.attributesOfItem`, decode `latest.json` via `UpdateSpec.decodeRecord`, build `UpdateSpec.status(...)`, log one line (`GET /v1/updates current=… bytes=… previous=… bytes=…`), return `json(encodable:status:status: .ok)`.

`uploadIPAResponse(request:)` order — this ordering is the requirement, not a detail:
1. `let build = UpdateSpec.build(fromQueryItems: request.uri.queryItems ?? [])`
2. `guard let rejection = UpdateSpec.rejection(build: build, body: request.body) else { … }` — on rejection: `FileLogger.shared.warn("PUT /v1/updates/ipa refused: \(rejection.message) (body \(request.body.count) bytes)")`, return `errorResponse(rejection.message, status: httpStatus(for: rejection))`. **No filesystem call before this point.**
3. `Self.uploadLock.withLock { … }` around: create the directory (`withIntermediateDirectories: true`), rotate (`removeItem` `.prev` if present → `moveItem` current → `.prev`), write `request.body` to the current path with `.atomic`, `try?` `Data.write(options: .atomic)`; on any thrown error log at error level and return `.internalServerError` with the message (the previous build is still in `.prev`).
4. Write `latest.json` (`UpdateSpec.Record(build: build!, bytes: request.body.count, uploadedAt: Date())` via `UpdateSpec.encode`).
5. Log one info line: `PUT /v1/updates/ipa build=\(build) \(request.body.count) bytes -> \(UpdateSpec.currentFileName) (previous \(previousBytes) bytes)`; respond `json(encodable: status, status: .created)` (rebuilt with the new `record`, so the client sees the stored result).

- [ ] **Step 3: Verify it parses (no local iOS build exists)**

```bash
cd /Users/chenxigui/develop/LlamaServer
swiftc -parse Sources/App/LlamaHTTPServer.swift && echo "parse OK"
```

Expected: `parse OK`. CI's `build` job is the real compile check (Step 4).

- [ ] **Step 4: Push and watch CI**

```bash
git add Sources/App/LlamaHTTPServer.swift
git commit -m "http: GET /v1/updates and PUT /v1/updates/ipa (self-upgrade upload)"
git push origin main
gh run watch "$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

Expected: `test` ✅ and `build` ✅. On-device verification happens in Task 3 Step 5 (needs the user to install this build, since only they can re-sign).

## Task 3: `scripts/upload-ipa.sh`

**Files:**
- Create: `scripts/upload-ipa.sh` (executable)

**Interfaces:**
- Consumes: `PUT /v1/updates/ipa?build=<sha>` and `GET /v1/updates` from Task 2; the CI artifact `LlamaServer-unsigned-ipa` → `LlamaServer-unsigned.ipa` (`.github/workflows/build.yml:110`).
- Produces: the command `scripts/upload-ipa.sh --url http://<iphone>:8443` that Task 4's skill documents.

- [ ] **Step 1: Write the script**

`set -euo pipefail`; `ROOT_DIR` like `package-ipa.sh:17`; usage/flags:
- `--url <base>` (default `$LLAMASERVER_URL`; error with a message if neither) — no device IP baked into the repo.
- `--run <id>` (default: latest successful `build.yml` run on `main`).
- `--file <path>` (upload a local IPA; skips `gh` entirely).
- `--build <sha>` (default: `headSha` of the resolved run, truncated to 7 — the same short form CI injects as `LlamaServerBuildSHA` and `git rev-parse --short HEAD` prints).
- `--help`.

Flow: resolve run → `gh run download "$RUN" --name LlamaServer-unsigned-ipa --dir "$TMP"` → `IPA="$TMP/LlamaServer-unsigned.ipa"`; sanity-check it (`-f`, size > 0, `unzip -l` mentions `Payload/`, else "that artifact is not an IPA — check `gh run view $RUN`"); `curl -fsS -X PUT --data-binary @"$IPA" -H 'Content-Type: application/octet-stream' "$URL/v1/updates/ipa?build=$BUILD"`; print the JSON; `curl -fsS "$URL/v1/updates"`; print the tap instructions (Files → On My iPhone → LlamaServer → updates → `LlamaServer-unsigned.ipa` → AltStore). Errors must be explicit: `curl` failure on a refused connection prints "is the server running? the routes only exist while a model is loaded" (Review Focus 1) and exits non-zero.

- [ ] **Step 2: Verify the shell parses and the script's own flow works against a stub**

```bash
cd /Users/chenxigui/develop/LlamaServer
bash -n scripts/upload-ipa.sh && echo "syntax OK"
```

Then start a throwaway stub that accepts the two routes and run the script against it with `--file` (no `gh`, no device):

```bash
python3 - <<'PY' &
import http.server, json
class H(http.server.BaseHTTPRequestHandler):
    def do_PUT(self):
        n = int(self.headers.get('Content-Length') or 0)
        self.rfile.read(n)
        self.send_response(201); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps({"current":{"fileName":"LlamaServer-unsigned.ipa","present":True,"bytes":n}}).encode())
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(b'{"current":{"present":true}}')
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', 8099), H).serve_forever()
PY
sleep 1
printf 'PK\003\004stub' > /tmp/fake.ipa
scripts/upload-ipa.sh --url http://127.0.0.1:8099 --file /tmp/fake.ipa --build 441aadc
```

Expected: the script prints the stub's 201 JSON, then the `GET /v1/updates` JSON, then the tap instructions, and exits 0. Kill the stub afterwards.

- [ ] **Step 3: Commit**

```bash
git add scripts/upload-ipa.sh
git commit -m "scripts: upload-ipa.sh — push the latest CI IPA to the phone over the LAN"
git push origin main
```

- [ ] **Step 4: CI green**

```bash
gh run watch "$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

- [ ] **Step 5: End-to-end against the device (needs the user)**

Ask the user to install the Task 2 build. Then:

```bash
scripts/upload-ipa.sh --url http://192.168.3.3:8443
curl -s 'http://192.168.3.3:8443/logs?name=server&tail=40' | grep 'updates'
```

Expected: 201 + the `Status` JSON with `current.bytes` == the artifact size (4,083,117 bytes for `40b596d`-era builds), a `PUT /v1/updates/ipa build=… bytes=…` line in the server log, and `Documents/updates/` visible in the Files app on the phone. Then ask the user to tap it (the skill explains the tap) — this is the only step that cannot be automated from here.

## Task 4: The skill and the docs

**Files:**
- Create: `skills/upload-ipa/SKILL.md`
- Modify: `README.md` (API endpoint table at lines 36-43; add a section after "Getting the logs off the device" at line 52)
- Modify: `AGENTS.md` (the "Debugging a deployed build" section: a bullet on the update routes, the always-on-when-loaded caveat, and the "no build id in the file name" rule)

**Interfaces:**
- Consumes: `scripts/upload-ipa.sh` (Task 3) and the routes (Task 2).
- Produces: the agent-facing runbook.

- [ ] **Step 1: Write `skills/upload-ipa/SKILL.md`**

Frontmatter in the `llm-compliance-test` shape (`name`, `description` starting "Use when …"). Body: Overview (why the inference server carries the route); When to Use (after a CI build you want on the phone; when the deployed build is stale/broken and you need a replacement without a cable); Prerequisites (`gh` authenticated, phone on the same LAN, **server running — the route exists only while a model is loaded**, which is the one case you cannot fix from the phone: pick a smaller model in the UI to get the server up); The Command; What Success Looks Like (201 + the `Status` JSON, matching byte count); The Tap (Files → On My iPhone → LlamaServer → `updates/` → `LlamaServer-unsigned.ipa` → tap → AltStore installs in place, preserving the container; note iOS deletes the container on uninstall, so an accidental uninstall takes the models and stored IPAs with it); Troubleshooting table (connection refused → engine stopped; 404 → deployed build predates the route, check `system_fingerprint`/`Model loaded … build llamaserver-<sha>`; 400 missing build → pass `--build`; 400 not an IPA → the download failed, check `gh run view`; 413 → over the cap); Why the file name has no hash (Files sorts by name; `latest.json`/`GET /v1/updates` carry the build; `.prev` is the rollback slot).

- [ ] **Step 2: Update `README.md`**

Add to the endpoint table:

```markdown
| GET    | `/v1/updates`           | What IPA builds are stored on the phone (self-upgrade, see below) |
| PUT    | `/v1/updates/ipa`       | Store the CI IPA as the current build (`?build=<sha>`) |
```

and a short section ("Pushing a new build onto the phone") after "Getting the logs off the device": the one-liner `scripts/upload-ipa.sh --url http://<iphone>:8443`, what it stores, the tap in Files, and the caveat that the routes exist only while the server runs.

- [ ] **Step 3: Update `AGENTS.md`**

In "Debugging a deployed build", add a bullet: `GET /v1/updates` / `PUT /v1/updates/ipa?build=<sha>` store the uploaded IPA in `Documents/updates/` (`LlamaServer-unsigned.ipa` + rotated `LlamaServer-unsigned.prev.ipa` + `latest.json`), the build id lives in `latest.json`/the log and **not** in the file name (Files sorts by name), the routes only exist while the engine runs (they ride the inference listener — no second port), the 128 MB cap cannot prevent Telegraph buffering the body (the check is best-effort, the IPA is ~4 MB), and the upload is serialized by its own lock (never the inference queue).

- [ ] **Step 4: Verify links and shape**

```bash
cd /Users/chenxigui/develop/LlamaServer
grep -n "v1/updates" README.md AGENTS.md skills/upload-ipa/SKILL.md
sed -n '1,5p' skills/upload-ipa/SKILL.md
```

Expected: the routes appear in all three; the frontmatter is `name:` + `description:` (same shape as `skills/llm-compliance-test/SKILL.md:1-4`).

- [ ] **Step 5: Commit and CI**

```bash
git add skills/upload-ipa/SKILL.md README.md AGENTS.md
git commit -m "docs: the self-upgrade runbook — upload-ipa skill, README, agent notes"
git push origin main
gh run watch "$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

Expected: `test` ✅ and `build` ✅.

## Self-Review

- **Spec coverage:** routes + contract (Tasks 1-2), script (Task 3), skill + docs (Task 4). Storage (`Documents/updates/`, `.prev` rotation, `latest.json`) and guards (build required, 128 MB cap, zip magic, no filesystem call before validation) are each pinned by a Task 1 test or a Task 2 step. No new UI (the log line is the only visible effect) — stated in Task 2 Step 2. Upload-only (no AltStore source JSON, no `altstore://` link) — Task 4 documents the tap instead.
- **Step scan:** every step names one action with one checkable result; code appears only where the exact values are decisions (test bodies, route registrations, the store order).
- **Type consistency:** `UpdateSpec.` names used in Task 2 are exactly the Interfaces block of Task 1; `JSONDecoder.iso8601` in the test is satisfied by the same ISO-8601 strategy Task 1 Step 3 specifies for `encode`.
- **Review Focus:** each of the six lines has an owning test/step — 1 → Task 3 Step 1 + Task 4 Step 1; 2 → Task 1 `testNonIPAIsRejected`/`testEmptyBodyIsRejectedAsNotAnIPA` + Task 2 Step 2 ordering; 3 → Task 1 `testBodyExactlyAtTheCapIsAccepted` + Task 2 Step 2 rotation; 4 → Task 2 Step 2 (atomic write, error leaves `.prev`); 5 → Task 1 `testMissingBuildIsRejectedFirst`; 6 → Task 2 Step 2 lock.
- **Proportion:** ~4 tasks for ~5 source/doc files; the test bodies are the bulk, which is where the decisions are.
