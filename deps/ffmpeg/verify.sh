#!/usr/bin/env bash
# Verify the series reproduces the pinned tree for an upstream base (n8.0 or n8.1).
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 <n8.0|n8.1> /path/to/FFmpeg" >&2; exit 2; }
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source "$dir/8/bases.env"
case "$1" in n8.0) k=n80 ;; n8.1) k=n81 ;; *) echo "unsupported base: $1" >&2; exit 2 ;; esac
v="${k}_commit"; base_commit=${!v}; v="${k}_tree"; expected_tree=${!v}
repo=$(git -C "$2" rev-parse --show-toplevel)
git -C "$repo" cat-file -e "${base_commit}^{commit}" || { echo "checkout lacks $1 ($base_commit)" >&2; exit 2; }
shopt -s nullglob; patches=("$dir"/8/*.patch)
[[ ${#patches[@]} -eq $patch_count ]] || { echo "expected $patch_count patches, found ${#patches[@]}" >&2; exit 1; }
root=$(mktemp -d "${TMPDIR:-/tmp}/avplumber-ffmpeg-verify.XXXXXX"); wt="$root/ffmpeg"
trap 'git -C "$repo" worktree remove --force "$wt" >/dev/null 2>&1 || true; rmdir "$root" 2>/dev/null || true' EXIT
git -C "$repo" worktree add --detach "$wt" "$base_commit" >/dev/null
"$dir/apply.sh" "$wt" >/dev/null
actual=$(git -C "$wt" rev-parse 'HEAD^{tree}')
[[ "$actual" == "$expected_tree" ]] || { echo "unexpected patched tree for $1: $actual (expected $expected_tree)" >&2; exit 1; }
echo "FFmpeg $1 patch series verified: $actual"
