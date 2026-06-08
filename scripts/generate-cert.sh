#!/usr/bin/env bash
#
# Generates a self-signed TLS certificate and bundles it as Resources/server.p12,
# which the app loads at runtime to serve HTTPS.
#
# The passphrase here MUST match `p12Password` in Sources/App/LlamaHTTPServer.swift.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/Resources"
PASS="${P12_PASSWORD:-llamaserver}"
DAYS="${CERT_DAYS:-3650}"
CN="${CERT_CN:-llama-server.local}"

mkdir -p "$OUT_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Generating self-signed certificate (CN=$CN)"
openssl req -x509 -newkey rsa:2048 \
  -keyout "$TMP/key.pem" \
  -out "$TMP/cert.pem" \
  -days "$DAYS" -nodes \
  -subj "/CN=$CN" \
  -addext "subjectAltName=DNS:$CN,IP:0.0.0.0"

echo "==> Exporting PKCS#12 to Resources/server.p12"
openssl pkcs12 -export \
  -inkey "$TMP/key.pem" \
  -in "$TMP/cert.pem" \
  -out "$OUT_DIR/server.p12" \
  -name "llama" \
  -passout "pass:$PASS"

echo "==> Done: $OUT_DIR/server.p12 (passphrase: $PASS)"
