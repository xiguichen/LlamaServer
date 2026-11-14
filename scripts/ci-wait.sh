#!/usr/bin/env bash
#
# Wait for a GitHub Actions run and print the failed log when it fails.
#
# Usage:
#   scripts/ci-wait.sh <run-id>
#   scripts/ci-wait.sh                 # latest build.yml run on main
#
# Exits 0 only when the run succeeds, so an agent can gate "done" on it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh not found — install the GitHub CLI. Never substitute curl." >&2
  exit 1
fi

RUN="${1:-}"
if [ -z "$RUN" ]; then
  RUN="$(gh run list --workflow build.yml --branch main --limit 1 \
          --json databaseId --jq '.[0].databaseId')"
fi
if [ -z "$RUN" ] || [ "$RUN" = "null" ]; then
  echo "ERROR: no run found. Pass a run id: scripts/ci-wait.sh <run-id>" >&2
  exit 1
fi

if gh run watch "$RUN" --exit-status; then
  echo "OK: run $RUN succeeded"
  exit 0
fi

echo "FAILED: run $RUN" >&2
bash "$ROOT_DIR/scripts/ci-logs.sh" "$RUN" >&2
exit 1
