#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${demo_dir}"

export AVPLUMBER_REVISION="$(git -C "${demo_dir}/../.." rev-parse HEAD 2>/dev/null || printf 'workspace')"
docker compose build test
