#!/usr/bin/env bash
# Generate deps/ffmpeg/8/0010-avformat-libmxl-demuxer-muxer.patch by
# replaying the MXL commits from cbcrc/FFmpeg on top of the avplumber
# series (0001-0009).
#
# Runs inside a plain Linux container (see Dockerfile.mkpatch). All state
# lives in /tmp; the resulting patch is written to /out.
#
# The fork keeps a branch per FFmpeg base (dmf-mxl/8.1, dmf-mxl/9.0, ...),
# each carrying the whole integration in a couple of commits on top of
# upstream release commits. We take the ones that touch libavformat/mxl*,
# plus any whose subject mentions mxl (the configure/allformats/Makefile
# glue), and squash them into one provenance-preserving patch so
# verify.sh keeps its "one feature per numbered patch" invariant.

set -euo pipefail

FFMPEG_TAG="${FFMPEG_TAG:-n8.1}"
# cbcrc's guidance-for-building-ffmpeg-with-mxl (scripts/get-src.sh)
# hosts the MXL FFmpeg work at cbcrc/FFmpeg. The dmf-mxl/8.1 branch
# forks from the same n8.1 commit that 8/bases.env pins.
MXL_REMOTE_URL="${MXL_REMOTE_URL:-https://github.com/cbcrc/FFmpeg.git}"
MXL_REMOTE_REF="${MXL_REMOTE_REF:-dmf-mxl/8.1}"
MXL_PIN="${MXL_PIN:-9eddb90ac0cf6063aaacc4fc2775f19d873500eb}"
OUT_DIR="${OUT_DIR:-/out}"
PATCH_DIR="${PATCH_DIR:-/patches}"
WORK_DIR="${WORK_DIR:-/tmp/ffmpeg-mxl-build}"

log() { printf '[mkpatch] %s\n' "$*"; }

test -d "$PATCH_DIR" || { echo "expected patch input dir at $PATCH_DIR" >&2; exit 2; }
test -d "$OUT_DIR"   || { echo "expected patch output dir at $OUT_DIR"  >&2; exit 2; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/ffmpeg"

cd "$WORK_DIR/ffmpeg"
git init --quiet .
git config user.name  "avplumber mxl builder"
git config user.email "avplumber-mxl@local"

# Blobless fetches keep this quick while preserving the history needed to
# compute a merge base with the fork.
log "fetching upstream $FFMPEG_TAG"
git remote add upstream https://github.com/FFmpeg/FFmpeg.git
git fetch --quiet --filter=blob:none --no-tags upstream "tag" "$FFMPEG_TAG"
git checkout --quiet -b work "$FFMPEG_TAG"

log "applying the existing series (everything below 0010)"
# A previous 0010 in $PATCH_DIR is what we are about to replace, so skip it.
mapfile -t EXISTING < <(ls "$PATCH_DIR"/0[0-9][0-9][0-9]-*.patch | sort | grep -v '/0010-')
git am --whitespace=nowarn "${EXISTING[@]}"

BASE_AFTER_STACK="$(git rev-parse HEAD)"

log "fetching MXL FFmpeg fork ($MXL_REMOTE_URL @ $MXL_REMOTE_REF, pin=$MXL_PIN)"
git remote add mxl-fork "$MXL_REMOTE_URL"
git fetch --quiet --filter=blob:none --no-tags \
    mxl-fork "refs/heads/$MXL_REMOTE_REF:refs/remotes/mxl-fork/pinned"
git rev-parse --verify "$MXL_PIN^{commit}" >/dev/null

MERGE_BASE="$(git merge-base "$MXL_PIN" "$FFMPEG_TAG" || true)"
if [ -z "$MERGE_BASE" ]; then
    log "no shared history with $MXL_PIN — falling back to $FFMPEG_TAG as range base"
    MERGE_BASE="$FFMPEG_TAG"
fi
log "range base: $MERGE_BASE"

log "collecting MXL commits"
mapfile -t MXL_COMMITS < <(
    {
        git log --reverse --format=%H "$MERGE_BASE".."$MXL_PIN" -- 'libavformat/mxl*'
        git log --reverse --format='%H %s' "$MERGE_BASE".."$MXL_PIN" \
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

log "cherry-picking onto $FFMPEG_TAG + the existing series"
# The 8.1 branch needs no conflict resolution. The fallback below stays for
# future bases (9.0 and up): prefer the version from the commit being
# picked, and if the path was removed there, remove it. We squash to one
# patch anyway, so only the end state matters.

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
    # Also handle unmerged entries that only appear in `git status`
    # ("DU"/"UD"/"AA"/"DD").
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
