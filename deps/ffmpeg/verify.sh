#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <7.1.5|8.1> /path/to/FFmpeg" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$1" in
    7.1.5|8.1) series_dir="$script_dir/$1" ;;
    *) echo "unsupported FFmpeg series: $1" >&2; exit 2 ;;
esac
source "$series_dir/base.env"
source_repo=$(git -C "$2" rev-parse --show-toplevel)
if ! git -C "$source_repo" cat-file -e "${base_commit}^{commit}"; then
    echo "FFmpeg checkout does not contain base commit ${base_commit}" >&2
    exit 2
fi

shopt -s nullglob
patches=("$series_dir"/*.patch)
if [[ ${#patches[@]} -ne $expected_patch_count ]]; then
    echo "expected ${expected_patch_count} patches, found ${#patches[@]}" >&2
    exit 1
fi

audit_root=$(mktemp -d "${TMPDIR:-/tmp}/avplumber-ffmpeg-verify.XXXXXX")
audit_worktree="$audit_root/ffmpeg"

cleanup() {
    if [[ -d "$audit_worktree" ]]; then
        git -C "$source_repo" worktree remove --force "$audit_worktree" \
            >/dev/null 2>&1 || true
    fi
    rmdir "$audit_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git -C "$source_repo" worktree add --detach "$audit_worktree" "$base_commit" \
    >/dev/null
git -C "$audit_worktree" -c user.name="avplumber patch verifier" \
    -c user.email="patch-verifier@local" am --whitespace=nowarn "${patches[@]}" >/dev/null

actual_tree=$(git -C "$audit_worktree" rev-parse 'HEAD^{tree}')
if [[ "$actual_tree" != "$expected_tree" ]]; then
    echo "unexpected patched tree: ${actual_tree}" >&2
    echo "expected patched tree:   ${expected_tree}" >&2
    exit 1
fi

echo "FFmpeg patch stack verified: ${actual_tree}"
