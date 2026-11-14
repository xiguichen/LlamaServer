#!/usr/bin/env bash
#
# Local syntax gate for Swift sources.
#
# `swiftc -parse` is syntax-only: no linking, no SDK, no type checking. It is the
# only build check available on this machine (docs/CI.md), so run it after every
# edit and before every commit. CI still owns the real build and the unit tests.
#
# Usage:
#   scripts/check-swift.sh                  # all Sources/App/*.swift (~10 s)
#   scripts/check-swift.sh <file> [...]     # only the files you touched (~2 s each)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v swiftc >/dev/null 2>&1; then
  echo "ERROR: swiftc not found — install the Xcode command line tools." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  FILES=(Sources/App/*.swift)
else
  FILES=("$@")
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "ERROR: no Swift files to check." >&2
  exit 1
fi

echo "==> swiftc -parse ${#FILES[@]} file(s)"
swiftc -parse "${FILES[@]}"
echo "OK: no syntax errors"
