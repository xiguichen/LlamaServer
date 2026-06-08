#!/usr/bin/env bash
#
# Downloads the OFFICIAL prebuilt llama.xcframework from llama.cpp's GitHub
# releases and places it at vendor/llama.xcframework.
#
# No compilation — just download + unzip. This works on any OS with bash + curl
# + unzip (including Windows/Git-Bash), since nothing is built locally.
#
# The framework module is named `llama` (matches `import llama` in Sources/App)
# and its modulemap auto-links Metal / Accelerate / Foundation / c++.
#
# Pin LLAMA_TAG to a llama.cpp release that publishes an `xcframework.zip` asset.
# Browse tags at: https://github.com/ggml-org/llama.cpp/releases
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b9553}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/vendor/llama.xcframework"

if [ -d "$DEST" ]; then
  echo "==> vendor/llama.xcframework already present, skipping download."
  exit 0
fi

ZIP_NAME="llama-${LLAMA_TAG}-xcframework.zip"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ZIP_NAME}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Downloading $URL"
if ! curl -fL --retry 3 -o "$TMP/$ZIP_NAME" "$URL"; then
  echo "ERROR: failed to download $URL" >&2
  echo "       Check that tag '$LLAMA_TAG' publishes an xcframework.zip asset." >&2
  exit 1
fi

echo "==> Unzipping"
unzip -q "$TMP/$ZIP_NAME" -d "$TMP/extract"

SRC="$TMP/extract/build-apple/llama.xcframework"
if [ ! -d "$SRC" ]; then
  # Fallback: locate the .xcframework wherever it landed in the archive.
  SRC="$(find "$TMP/extract" -maxdepth 3 -type d -name 'llama.xcframework' | head -n 1)"
fi
if [ -z "${SRC:-}" ] || [ ! -d "$SRC" ]; then
  echo "ERROR: llama.xcframework not found inside $ZIP_NAME" >&2
  exit 1
fi

echo "==> Installing to $DEST"
mkdir -p "$ROOT_DIR/vendor"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"

echo "==> Done: $DEST (llama.cpp $LLAMA_TAG)"
