#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${demo_dir}"

mkdir -p artifacts
"${demo_dir}/build.sh"
docker compose run --rm test "$@"
