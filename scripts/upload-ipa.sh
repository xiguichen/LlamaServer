#!/usr/bin/env bash
#
# Push the latest CI-built IPA onto the phone.
#
# The app stores it in Documents/updates/ (see Sources/App/UpdateSpec.swift), which
# the Files app shows — open Files on the iPhone, tap the .ipa, and AltStore
# installs it in place (your models stay). No cable, no AirDrop, no AltServer.
#
# Usage:
#   scripts/upload-ipa.sh --url http://192.168.3.3:8443
#   scripts/upload-ipa.sh --url http://192.168.3.3:8443 --file build/LlamaServer-unsigned.ipa
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_NAME="LlamaServer-unsigned-ipa"
IPA_NAME="LlamaServer-unsigned.ipa"

usage() {
  sed -n '3,11p' "$0" | sed 's/^#\{1,2\} \{0,1\}//'
  cat <<'EOS'

Flags:
  --url <base>    Server base URL, e.g. http://192.168.3.3:8443 (or set
                  $LLAMASERVER_URL). Required: the phone's address is not
                  knowable from the repo, and a baked-in default goes stale.
  --run <id>      GitHub Actions run id to pull the artifact from.
                  Default: the latest successful build.yml run on main.
  --file <path>   Upload a local IPA instead of downloading one (skips gh).
  --build <sha>   Build id recorded with the upload. Default: the run's short
                  head SHA (the same form CI injects as LlamaServerBuildSHA), or
                  git HEAD for --file.
  -h, --help      This text.
EOS
}

URL="${LLAMASERVER_URL:-}"
RUN=""
LOCAL_IPA=""
BUILD=""

while [ $# -gt 0 ]; do
  case "$1" in
    --url)   URL="${2:-}";   shift 2 ;;
    --run)   RUN="${2:-}";   shift 2 ;;
    --file)  LOCAL_IPA="${2:-}"; shift 2 ;;
    --build) BUILD="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$URL" ]; then
  echo "ERROR: no server URL. Pass --url http://<iphone>:8443 or set LLAMASERVER_URL." >&2
  exit 2
fi
URL="${URL%/}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- Source the IPA -----------------------------------------------------------

if [ -n "$LOCAL_IPA" ]; then
  IPA="$LOCAL_IPA"
  if [ ! -f "$IPA" ]; then
    echo "ERROR: no such file: $IPA" >&2
    exit 1
  fi
  if [ -z "$BUILD" ]; then
    BUILD="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo local)"
    echo "==> Recording build '$BUILD' (git HEAD); override with --build <sha>"
  fi
else
  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh not found — install the GitHub CLI, or pass --file <path>." >&2
    exit 1
  fi

  if [ -z "$RUN" ]; then
    echo "==> Resolving the latest successful CI build"
    RUN="$(gh run list --workflow build.yml --branch main --status success \
             --limit 1 --json databaseId --jq '.[0].databaseId')"
    if [ -z "$RUN" ] || [ "$RUN" = "null" ]; then
      echo "ERROR: no successful build.yml run on main yet. Check: gh run list --workflow build.yml" >&2
      exit 1
    fi
  fi

  if [ -z "$BUILD" ]; then
    SHA="$(gh run view "$RUN" --json headSha --jq '.headSha')"
    BUILD="${SHA:0:7}"
  fi

  echo "==> Downloading $ARTIFACT_NAME from run $RUN (build $BUILD)"
  gh run download "$RUN" --name "$ARTIFACT_NAME" --dir "$WORK" >/dev/null
  IPA="$WORK/$IPA_NAME"
  if [ ! -f "$IPA" ]; then
    echo "ERROR: run $RUN has no $IPA_NAME in artifact '$ARTIFACT_NAME'." >&2
    echo "       Check the run: gh run view $RUN" >&2
    exit 1
  fi
fi

# --- Sanity-check it before replacing a working build ------------------------

BYTES="$(wc -c < "$IPA" | tr -d ' ')"
if [ "$BYTES" -le 0 ]; then
  echo "ERROR: $IPA is empty" >&2
  exit 1
fi
echo "    $IPA ($BYTES bytes)"

if command -v unzip >/dev/null 2>&1; then
  # Match with a case pattern, not `unzip -l | grep -q`: grep exits at the first
  # match, unzip then dies on SIGPIPE (141), and `pipefail` reports that as a
  # failure — a valid IPA would be refused.
  LISTING="$(unzip -l "$IPA" 2>/dev/null || true)"
  case "$LISTING" in
    *Payload/*) : ;;
    *) echo "ERROR: $IPA has no Payload/ — it is not an IPA." >&2; exit 1 ;;
  esac
else
  echo "    (unzip not found — skipping the Payload/ check)"
fi

# --- Upload -------------------------------------------------------------------

echo "==> Uploading to $URL/v1/updates/ipa?build=$BUILD"
RESPONSE="$(curl -sS -X PUT --data-binary @"$IPA" \
  -H 'Content-Type: application/octet-stream' \
  --write-out '\n%{http_code}' \
  "$URL/v1/updates/ipa?build=$BUILD" 2>&1)" || {
    echo "ERROR: could not reach $URL/." >&2
    echo "       Is the server running (green in the app) and the phone on this network?" >&2
    echo "       The update routes only exist while a model is loaded — if the engine will" >&2
    echo "       not start, pick a smaller model in the app and start it again." >&2
    exit 1
  }

CODE="$(printf '%s' "$RESPONSE" | tail -n 1)"
BODY="$(printf '%s' "$RESPONSE" | sed '$d')"

if [ "$CODE" != "201" ]; then
  echo "ERROR: the server refused the upload (HTTP $CODE):" >&2
  printf '%s\n' "$BODY" >&2
  echo "       (existing files on the phone were left untouched)" >&2
  exit 1
fi
echo "$BODY"

echo "==> On the phone now:"
curl -fsS "$URL/v1/updates"
echo

cat <<'EOS'

==> Install it:
  1. Open Files on the iPhone.
  2. On My iPhone -> LlamaServer -> updates -> tap LlamaServer-unsigned.ipa
  3. AltStore offers to install it, in place (your models stay).

  The build this replaces is kept as LlamaServer-unsigned.prev.ipa, so a build
  that turns out to be broken on device is one tap away from being restored.
EOS
