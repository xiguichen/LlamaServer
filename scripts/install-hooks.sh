#!/usr/bin/env bash
#
# Enable the repo's git hooks (.githooks/) for this clone.
#
#   bash scripts/install-hooks.sh
#
# The pre-commit hook runs `swiftc -parse` on staged Swift files. Hooks are not
# active until this runs in a fresh clone.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

chmod +x .githooks/pre-commit scripts/check-swift.sh
git config core.hooksPath .githooks

echo "OK: core.hooksPath=.githooks"
echo "Pre-commit now parses staged .swift files with scripts/check-swift.sh."
