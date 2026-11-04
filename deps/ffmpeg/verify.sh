#!/usr/bin/env bash
# Verify the patch series reproduces its pinned tree.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 master-1bfb9160 /path/to/FFmpeg" >&2; exit 2; }
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$1" in
    master-1bfb9160)
        source "$dir/9/bases.env"
        base_commit=$upstream_commit
        expected_tree=$upstream_tree
        ;;
    *) echo "unsupported base: $1" >&2; exit 2 ;;
esac
repo=$(git -C "$2" rev-parse --show-toplevel)
git -C "$repo" cat-file -e "${base_commit}^{commit}" || { echo "checkout lacks $1 ($base_commit)" >&2; exit 2; }
shopt -s nullglob; patches=("$dir"/9/*.patch)
[[ ${#patches[@]} -eq $patch_count ]] || { echo "expected $patch_count patches, found ${#patches[@]}" >&2; exit 1; }
root=$(mktemp -d "${TMPDIR:-/tmp}/avplumber-ffmpeg-verify.XXXXXX"); wt="$root/ffmpeg"
trap 'git -C "$repo" worktree remove --force "$wt" >/dev/null 2>&1 || true; rmdir "$root" 2>/dev/null || true' EXIT
git -C "$repo" worktree add --detach "$wt" "$base_commit" >/dev/null
"$dir/apply.sh" "$wt" >/dev/null
actual=$(git -C "$wt" rev-parse 'HEAD^{tree}')
[[ "$actual" == "$expected_tree" ]] || { echo "unexpected patched tree for $1: $actual (expected $expected_tree)" >&2; exit 1; }
echo "FFmpeg $1 patch series verified: $actual"
