#!/usr/bin/env bash
# Squash the MXL cherry-picks into a single provenance-preserving patch and
# write 0010-avformat-libmxl-demuxer-muxer.patch to /out.
#
# $1 = commit hash of "base after 0001-0009 applied". Passed by
#      mkpatch-0010-mxl.sh after a successful cherry-pick, or by the user
#      manually after resolving conflicts and running
#      `git cherry-pick --continue`.

set -eu
set -o pipefail
FFMPEG_TAG="${FFMPEG_TAG:-n8.1}"

BASE_AFTER_STACK="${1:-}"
if [ -z "$BASE_AFTER_STACK" ]; then
    # If invoked manually, recover the base by walking back over MXL commits.
    BASE_AFTER_STACK="$(git log --format=%H --grep='cherry picked from' --invert-grep -1)"
fi

OUT_DIR="${OUT_DIR:-/out}"
OUT_FILE="$OUT_DIR/0010-avformat-libmxl-demuxer-muxer.patch"

log() { printf '[mkpatch] %s\n' "$*"; }

# The series must apply to both n8.0 and n8.1 from one copy, so the configure
# hunks may only touch lines the two bases share. As imported, the MXL work
# anchors its `require_pkg_config libmxl` next to libmpeghdec, which exists
# only in 8.1, and carries a whitespace-only reindent of the mmal check. Move
# the require next to the mxl_* deps lines (whose neighbours are identical in
# both bases) and drop the reindent.
log "re-anchoring the configure hunks for n8.0/n8.1 portability"
awk '
    /^enabled libmxl  *&& require_pkg_config libmxl/ { require = $0; next }
    /^ +check_func_headers interface\/mmal\/mmal\.h "MMAL_PARAMETER_VIDEO_MAX_NUM_CALLBACKS"; }$/ {
        sub(/^ +/, "                               "); print; next
    }
    /^enabled mxl_demuxer  *&& prepend avformat_deps/ {
        if (require == "") { print "mkpatch: libmxl require line not found" > "/dev/stderr"; exit 1 }
        print require; moved = 1
    }
    { print }
    END { if (!moved) { print "mkpatch: mxl_demuxer prepend line not found" > "/dev/stderr"; exit 1 } }
' configure >configure.mkpatch
mv configure.mkpatch configure
chmod +x configure

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
git add -A

MSG_FILE="$(mktemp)"
{
    printf 'avformat/mxl: add libmxl demuxer and muxer\n\n'
    printf 'Adds the MXL demuxer, muxer, URI parser, JSON/diagnostic helpers\n'
    printf 'and --enable-libmxl configure glue.\n\n'
    printf 'Cherry-picked from https://github.com/cbcrc/FFmpeg (branch\n'
    printf '%s) and squashed onto %s on top of the avplumber\n' "${MXL_REMOTE_REF:-dmf-mxl/8.1}" "$FFMPEG_TAG"
    printf 'series 0001-0009, with the configure hunks re-anchored so the\n'
    printf 'patch applies to both n8.0 and n8.1.\n\n'
    printf 'Original commits:\n%s\n\n' "$ORIGINAL_COMMITS"
    [ -n "$AUTHOR_TRAILERS" ] && printf '%s\n' "$AUTHOR_TRAILERS"
} >"$MSG_FILE"

GIT_COMMITTER_NAME="avplumber mxl builder" \
GIT_COMMITTER_EMAIL="avplumber-mxl@local" \
git commit --quiet -F "$MSG_FILE" \
    --author="avplumber mxl builder <avplumber-mxl@local>"
rm -f "$MSG_FILE"

log "writing patch to $OUT_FILE"
# Minimal context, like the rest of the series: it lets one copy apply to
# bases that differ only cosmetically around the touched lines.
git format-patch -1 --stdout --unified=1 HEAD >"$OUT_FILE"

log "done."
log "next: refresh deps/ffmpeg/8/bases.env (patch_count and both trees) from"
log "      deps/ffmpeg/verify.sh n8.1 <ffmpeg-repo>"
log "      deps/ffmpeg/verify.sh n8.0 <ffmpeg-repo>"
