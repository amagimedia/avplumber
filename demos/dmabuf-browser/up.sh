#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; copy ${SCRIPT_DIR}/.env.example and edit it first." >&2
  exit 1
fi

exec docker compose \
  --env-file "${ENV_FILE}" \
  -f "${SCRIPT_DIR}/compose.yaml" \
  up -d --build
