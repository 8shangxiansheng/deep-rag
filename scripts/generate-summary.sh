#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RETRY_ROUNDS="${RETRY_ROUNDS:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_DELAY_SECONDS="${REQUEST_DELAY_SECONDS:-0.5}"

python "$ROOT_DIR/Knowledge-Base-File-Summary/generate.py" \
  --run-id "$RUN_ID" \
  --retry-rounds "$RETRY_ROUNDS" \
  --max-retries "$MAX_RETRIES" \
  --request-delay-seconds "$REQUEST_DELAY_SECONDS" \
  --strict
