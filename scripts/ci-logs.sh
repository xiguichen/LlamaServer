#!/usr/bin/env bash
#
# Print the failed-job log of a GitHub Actions run using the authenticated gh CLI.
#
# The GitHub REST API needs auth: an unauthenticated `curl api.github.com/...`
# returns 404 and sends an agent down a dead end. Always use gh for CI.
#
# Usage:
#   scripts/ci-logs.sh <run-id>
#   scripts/ci-logs.sh                 # latest build.yml run on main
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
  echo "ERROR: no run found. Pass a run id: scripts/ci-logs.sh <run-id>" >&2
  exit 1
fi

echo "==> gh run view $RUN --log-failed"
gh run view "$RUN" --log-failed 2>&1 | tail -n 400
