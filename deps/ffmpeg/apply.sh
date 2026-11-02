#!/usr/bin/env bash
# Apply the patch series to the pinned upstream FFmpeg checkout.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 /path/to/FFmpeg" >&2; exit 2; }
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
src=$1
source "$dir/9/bases.env"
base=$(git -C "$src" rev-parse HEAD)
[[ $base == "$upstream_commit" ]] || { echo "unsupported FFmpeg base: $base (use $upstream_ref)" >&2; exit 2; }
git -C "$src" -c user.name="avplumber patches" -c user.email="patches@local" \
    am --whitespace=nowarn "$dir"/9/*.patch
