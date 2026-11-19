#!/usr/bin/env bash
# Rebuild the generated patch from the CBC sources and our two local changes.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 /path/to/FFmpeg" >&2; exit 2; }
dir=$(cd "$(dirname "$0")" && pwd)
source "$dir/9/bases.env"
fork=https://github.com/cbcrc/FFmpeg.git
pin=16abbf041307fd08db2566e488c1f4e0714a158a
output="$dir/9/0026-avformat-libmxl-demuxer-muxer.patch"
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
git clone -q --shared --no-checkout "$1" "$work/ffmpeg"
cd "$work/ffmpeg"
git checkout -q --detach "$upstream_commit"
git config user.name 'avplumber mxl builder'
git config user.email avplumber-mxl@local
for patch in "$dir"/9/*.patch; do
    [[ "$patch" != "$output" ]] || break
    git am --quiet --whitespace=nowarn "$patch"
done
git fetch -q --no-tags --depth=1 "$fork" "$pin"
git restore --source="$pin" --worktree --staged -- \
    'libavformat/mxl*' 'libavformat/jsmn.*' 'libavformat/tests/mxl*' \
    tests/fate/mxl.mak 'tests/ref/fate/mxl-*'
git apply --index "$dir/mxl/register.diff" "$dir/mxl/resume-pts.patch"
# Use the source commit date so identical inputs produce the same patch bytes.
export GIT_AUTHOR_DATE=$(git show -s --format=%aI "$pin")
export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"
git commit -q -m 'avformat/mxl: import libmxl muxer and demuxer' \
    -m "Source: $fork at $pin (dmf-mxl/9.0). AVPlumber supplies registration and PTS reset handling." \
    -m 'Co-authored-by: jpt <jim.trainor@jptrainor.com>'
git format-patch -1 --zero-commit --no-signature --stdout > "$work/mxl.patch"
cp "$work/mxl.patch" "$output"
