#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/libgbm_linear_shim.so"

cc="${CC:-cc}"

"$cc" \
  -shared \
  -fPIC \
  -O2 \
  -Wall \
  -Wextra \
  "${SCRIPT_DIR}/gbm_linear_shim.c" \
  -ldl \
  -o "$OUT"

echo "$OUT"
