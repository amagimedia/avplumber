#!/usr/bin/env bash
set -euo pipefail

base_commit=3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587
expected_tree=ac461ac62fae9f027871e18a53e4859336abfad4
expected_patch_count=8

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/FFmpeg" >&2
    exit 2
fi

source_repo=$(git -C "$1" rev-parse --show-toplevel)
if ! git -C "$source_repo" cat-file -e "${base_commit}^{commit}"; then
    echo "FFmpeg checkout does not contain base commit ${base_commit}" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
shopt -s nullglob
patches=("$script_dir"/*.patch)
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
git -C "$audit_worktree" config user.name "avplumber patch verifier"
git -C "$audit_worktree" config user.email "patch-verifier@local"
git -C "$audit_worktree" am --whitespace=nowarn "${patches[@]}" >/dev/null

actual_tree=$(git -C "$audit_worktree" rev-parse 'HEAD^{tree}')
if [[ "$actual_tree" != "$expected_tree" ]]; then
    echo "unexpected patched tree: ${actual_tree}" >&2
    echo "expected patched tree:   ${expected_tree}" >&2
    exit 1
fi

echo "FFmpeg patch stack verified: ${actual_tree}"
