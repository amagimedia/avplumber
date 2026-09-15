#!/usr/bin/env bash
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec bash "$script_dir/../verify.sh" 7.1.5 "$@"
