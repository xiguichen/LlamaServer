#!/usr/bin/env bash
#
# Packages an unsigned .app into an unsigned .ipa.
#
# An .ipa is just a zip with the app under Payload/. No signing happens here;
# tools like AltStore / Sideloadly / ESign re-sign on the device at install time.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
OUT_IPA="${OUT_IPA:-$ROOT_DIR/LlamaServer-unsigned.ipa}"

echo "==> Locating built .app"
APP_PATH="$(find "$BUILD_DIR" -type d -name '*.app' -path '*Release-iphoneos*' | head -n 1)"
if [ -z "${APP_PATH:-}" ]; then
  echo "ERROR: no Release-iphoneos .app found under $BUILD_DIR" >&2
  exit 1
fi
echo "    Found: $APP_PATH"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/Payload"
cp -R "$APP_PATH" "$STAGE/Payload/"

echo "==> Zipping to $OUT_IPA"
rm -f "$OUT_IPA"
( cd "$STAGE" && zip -qry "$OUT_IPA" Payload )

echo "==> Done: $OUT_IPA"
