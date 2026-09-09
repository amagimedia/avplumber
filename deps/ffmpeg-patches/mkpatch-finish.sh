#!/usr/bin/env bash
# Squash the MXL cherry-picks into a single provenance-preserving patch
# and write 0008-avformat-libmxl-demuxer-muxer.patch to /out.
#
# $1 = commit hash of "base after 0001-0007 applied". Passed by mkpatch-0008-mxl.sh
#      after a successful cherry-pick, or by the user manually after resolving
#      conflicts and running `git cherry-pick --continue`.

set -eu
set -o pipefail
FFMPEG_TAG="${FFMPEG_TAG:-n7.1.5}"

BASE_AFTER_STACK="${1:-}"
if [ -z "$BASE_AFTER_STACK" ]; then
    # If invoked manually, recover the base by walking back over MXL commits.
    BASE_AFTER_STACK="$(git log --format=%H --grep='cherry picked from' --invert-grep -1)"
fi

OUT_DIR="${OUT_DIR:-/out}"
OUT_FILE="$OUT_DIR/0008-avformat-libmxl-demuxer-muxer.patch"

log() { printf '[mkpatch] %s\n' "$*"; }

log "collecting authorship from cherry-picked commits"
AUTHOR_TRAILERS="$(
    git log --format='Co-authored-by: %an <%ae>' "$BASE_AFTER_STACK"..HEAD \
        | awk '!seen[$0]++'
)"
ORIGINAL_COMMITS="$(
    git log --format='  * %h %s' "$BASE_AFTER_STACK"..HEAD
)"

log "squashing cherry-picks to a single commit"
git reset --soft "$BASE_AFTER_STACK"

MSG_FILE="$(mktemp)"
{
    printf 'avformat/mxl: add libmxl demuxer and muxer\n\n'
    printf 'Adds the MXL demuxer, muxer, and --enable-libmxl configure glue.\n'
    printf 'Cherry-picked from https://github.com/cbcrc/FFmpeg (branch\n'
    printf 'dmf-mxl/master) and rebased onto %s on top of the existing\n' "$FFMPEG_TAG"
    printf 'avplumber patch stack (0001-0007).\n\n'
    printf 'Original commits:\n%s\n\n' "$ORIGINAL_COMMITS"
    [ -n "$AUTHOR_TRAILERS" ] && printf '%s\n' "$AUTHOR_TRAILERS"
} >"$MSG_FILE"

GIT_COMMITTER_NAME="avplumber mxl builder" \
GIT_COMMITTER_EMAIL="avplumber-mxl@local" \
git commit --quiet -F "$MSG_FILE" \
    --author="avplumber mxl builder <avplumber-mxl@local>"
rm -f "$MSG_FILE"

log "writing patch to $OUT_FILE"
git format-patch -1 --stdout HEAD >"$OUT_FILE"

NEW_TREE="$(git rev-parse HEAD^{tree})"
log "new expected patched tree: $NEW_TREE"
printf '%s\n' "$NEW_TREE" >"$OUT_DIR/expected-tree.txt"

log "done."
log "next: update 'Expected patched tree' in deps/ffmpeg-patches/README.md"
log "      to: $NEW_TREE"
log "and rerun deps/ffmpeg-patches/verify.sh against a fresh $FFMPEG_TAG checkout."
