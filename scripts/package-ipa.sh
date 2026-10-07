#!/usr/bin/env bash
#
# Packages the built .app into an .ipa.
#
# An .ipa is just a zip with the app under Payload/. Before zipping, the app is
# ad-hoc signed (-) with LlamaServer.entitlements so the entitlements blob lands
# in the main binary's LC_CODE_SIGNATURE. That is the only place AltSign reads
# entitlements from (ldid::Entitlements on CFBundleExecutable — it never looks at
# archived-expanded-entitlements.xcent), so re-signers like AltStore 2.3+ can
# carry increased-memory-limit over. No Apple account or profile needed.
#
# xcodebuild itself cannot do this: ad-hoc signing for the iphoneos SDK is
# rejected at build time ("Ad Hoc code signing is not allowed with SDK ..."),
# hence signing happens here, after the unsigned build.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
OUT_IPA="${OUT_IPA:-$ROOT_DIR/LlamaServer-unsigned.ipa}"
ENTITLEMENTS="${ENTITLEMENTS:-$ROOT_DIR/LlamaServer.entitlements}"

if [ ! -f "$ENTITLEMENTS" ]; then
  echo "ERROR: entitlements file not found: $ENTITLEMENTS" >&2
  exit 1
fi

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
STAGED_APP="$STAGE/Payload/$(basename "$APP_PATH")"

# Xcode-signed IPAs carry the expanded entitlements at the bundle root; write it
# before signing so the resource seal covers it. Belt-and-braces for re-sign
# tools that read the file instead of the signature.
echo "==> Writing archived-expanded-entitlements.xcent"
cp "$ENTITLEMENTS" "$STAGED_APP/archived-expanded-entitlements.xcent"

echo "==> Ad-hoc signing with entitlements"
codesign --force --sign - --entitlements "$ENTITLEMENTS" "$STAGED_APP"

echo "==> Verifying entitlements in code signature"
if ! codesign -d --entitlements - "$STAGED_APP" 2>&1 | grep -q 'com.apple.developer.kernel.increased-memory-limit'; then
  echo "ERROR: increased-memory-limit entitlement missing from code signature" >&2
  exit 1
fi

echo "==> Zipping to $OUT_IPA"
rm -f "$OUT_IPA"
( cd "$STAGE" && zip -qry "$OUT_IPA" Payload )
echo "==> Done: $OUT_IPA"
