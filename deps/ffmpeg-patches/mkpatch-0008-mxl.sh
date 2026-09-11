#!/usr/bin/env bash
# Generate deps/ffmpeg-patches/0008-avformat-libmxl-demuxer-muxer.patch
# by rebasing the MXL commits from dmf-mxl/FFmpeg onto n7.1.5 on top of
# the existing patch stack.
#
# Runs inside a plain Linux container (see mkpatch-0008-mxl.docker.sh).
# All state lives in /tmp; the resulting patch is written to /out.
#
# The MXL commit set is identified by two heuristics:
#   1. commits reachable from dmf-mxl/master but not from FFmpeg n7.1.5
#      that touch libavformat/mxl*,
#   2. plus any of those commits whose subject line contains "mxl"
#      (case-insensitive) — catches the configure/allformats/Makefile
#      glue commits that don't add mxl* files themselves.
#
# The cherry-picks are then squashed into a single provenance-preserving
# patch so verify.sh keeps its "one feature per numbered patch" invariant.

set -euo pipefail

FFMPEG_TAG="${FFMPEG_TAG:-n7.1.5}"
# cbcrc's guidance-for-building-ffmpeg-with-mxl (scripts/get-src.sh)
# hosts the MXL FFmpeg patches at cbcrc/FFmpeg, branch dmf-mxl/master.
MXL_REMOTE_URL="${MXL_REMOTE_URL:-https://github.com/cbcrc/FFmpeg.git}"
MXL_REMOTE_REF="${MXL_REMOTE_REF:-dmf-mxl/master}"
MXL_PIN="${MXL_PIN:-5c5d593}"
OUT_DIR="${OUT_DIR:-/out}"
PATCH_DIR="${PATCH_DIR:-/patches}"
WORK_DIR="${WORK_DIR:-/tmp/ffmpeg-mxl-build}"

log() { printf '[mkpatch] %s\n' "$*"; }

test -d "$PATCH_DIR" || { echo "expected patch input dir at $PATCH_DIR" >&2; exit 2; }
test -d "$OUT_DIR"   || { echo "expected patch output dir at $OUT_DIR"  >&2; exit 2; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

log "cloning FFmpeg $FFMPEG_TAG"
git clone --quiet --branch "$FFMPEG_TAG" --depth 1 \
    https://github.com/FFmpeg/FFmpeg.git "$WORK_DIR/ffmpeg"

cd "$WORK_DIR/ffmpeg"
git config user.name  "avplumber mxl builder"
git config user.email "avplumber-mxl@local"

log "applying existing 0001-0007 stack"
# shellcheck disable=SC2046
git am $(ls "$PATCH_DIR"/0[0-9][0-9][0-9]-*.patch | sort)

BASE_AFTER_STACK="$(git rev-parse HEAD)"

log "fetching MXL FFmpeg fork ($MXL_REMOTE_URL @ $MXL_REMOTE_REF, pin=$MXL_PIN)"
git remote add mxl-fork "$MXL_REMOTE_URL"
# The fork's dmf-mxl/master branch carries the MXL commits; it can be
# many commits above the base so no --depth here.
git fetch --no-tags --quiet mxl-fork "$MXL_REMOTE_REF:refs/remotes/mxl-fork/master"
git rev-parse --verify "$MXL_PIN^{commit}" >/dev/null 2>&1 || \
    git fetch --no-tags --quiet mxl-fork "$MXL_PIN"

MXL_TIP="$MXL_PIN"
MERGE_BASE="$(git merge-base "$MXL_TIP" "$FFMPEG_TAG" || true)"
if [ -z "$MERGE_BASE" ]; then
    log "no shared history with $MXL_TIP — falling back to $FFMPEG_TAG as range base"
    MERGE_BASE="$FFMPEG_TAG"
fi
log "range base: $MERGE_BASE"

log "collecting MXL commits"
mapfile -t MXL_COMMITS < <(
    {
        git log --reverse --format=%H "$MERGE_BASE".."$MXL_TIP" -- 'libavformat/mxl*'
        git log --reverse --format='%H %s' "$MERGE_BASE".."$MXL_TIP" \
            | awk 'tolower($0) ~ /mxl/ { print $1 }'
    } | awk '!seen[$0]++'
)

if [ "${#MXL_COMMITS[@]}" -eq 0 ]; then
    echo "no MXL commits found in $MXL_REMOTE_REF; nothing to do" >&2
    exit 3
fi

log "picking ${#MXL_COMMITS[@]} commit(s):"
for c in "${MXL_COMMITS[@]}"; do
    printf '  %s %s\n' "$c" "$(git log -1 --format=%s "$c")"
done

log "cherry-picking onto n7.1.5 + existing stack (-X theirs, auto-resolve)"
# --keep-redundant-commits lets pure-metadata cherry-picks succeed.
# -X theirs auto-resolves content-conflicts in additive files.
# For rename/delete cases -X can't handle, we fall through to a manual
# loop: prefer the "theirs" version (the commit being picked), and if
# the path was removed there, remove it. Since we squash to one patch
# anyway, the end state is what matters.

auto_resolve_conflicts() {
    local pending
    pending="$(git diff --name-only --diff-filter=U)"
    if [ -z "$pending" ]; then
        return 1
    fi
    while IFS= read -r path; do
        [ -z "$path" ] && continue
        if git checkout --theirs -- "$path" 2>/dev/null; then
            git add -- "$path"
        else
            git rm -f -- "$path" >/dev/null
        fi
    done <<<"$pending"
    # Also handle unmerged entries that only appear in --name-only
    # via `git status --porcelain=v1` ("DU"/"UD"/"AA"/"DD").
    while IFS= read -r line; do
        local code path2
        code="${line:0:2}"
        path2="${line:3}"
        case "$code" in
            DU|UD|DD|AU|UA|AA|UU)
                if git checkout --theirs -- "$path2" 2>/dev/null; then
                    git add -- "$path2"
                else
                    git rm -f -- "$path2" >/dev/null 2>&1 || true
                fi
                ;;
        esac
    done < <(git status --porcelain=v1)
    return 0
}

set +e
git cherry-pick --keep-redundant-commits \
    --strategy=recursive -X theirs -x "${MXL_COMMITS[@]}"
rc=$?
while [ "$rc" -ne 0 ]; do
    if ! auto_resolve_conflicts; then
        log "cherry-pick failed with no unresolved paths — aborting"
        exit 4
    fi
    if git diff --cached --quiet; then
        log "empty commit after resolve — skipping"
        git cherry-pick --skip
    else
        GIT_EDITOR=true git cherry-pick --continue
    fi
    rc=$?
done
set -e

exec bash /usr/local/bin/mkpatch-finish "$BASE_AFTER_STACK"
