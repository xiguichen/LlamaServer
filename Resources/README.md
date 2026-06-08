# Resources

`server.p12` (the self-signed TLS identity bundled into the app) is **generated
at build time** by `scripts/generate-cert.sh` and is intentionally git-ignored.

If you build locally, run `bash scripts/generate-cert.sh` before
`xcodegen generate` so the file is present and gets bundled into the app.
