#!/usr/bin/env bash
#
# Launch pi with the small-model profile for this repo (qwen3.5:4b).
#
# A 4B model calls tools reliably only with a short prompt. Skills, extension
# tools (web/notes/meeting/history), and a huge context window dilute the prompt
# until the model announces a tool and stops. This profile keeps four tools and
# no skill text. See docs/SMALL-MODEL.md.
#
# Usage:
#   scripts/pi4b.sh                 # interactive
#   scripts/pi4b.sh "fix <task>"    # one-shot
#   scripts/pi4b.sh --resume        # any pi flag is passed through
#
# Override the model with PI_MODEL=provider/model or the binary with PI_BIN.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PI_BIN="${PI_BIN:-}"
if [ -z "$PI_BIN" ]; then
  if command -v pi >/dev/null 2>&1; then
    PI_BIN="$(command -v pi)"
  elif [ -x "$HOME/.pi/agent/bin/pi" ]; then
    PI_BIN="$HOME/.pi/agent/bin/pi"
  else
    echo "ERROR: pi not found. Set PI_BIN or put pi on PATH." >&2
    exit 1
  fi
fi

exec "$PI_BIN" \
  --model "${PI_MODEL:-llamacpp/qwen3.5:4b}" \
  --thinking off \
  --tools read,bash,edit,write \
  --no-skills \
  --no-extensions \
  -e .pi/extensions/project-guard.ts \
  "$@"
