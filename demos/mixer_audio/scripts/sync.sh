#!/usr/bin/env bash
# Sync the local avplumber checkout to the demo host over rsync+ssh. No git
# push/pull required (the host has no git access).
#
# Uses --checksum (content comparison) rather than rsync's default
# size+mtime quick-check: the host's avplumber checkout tends to carry
# uncommitted edits made directly on it, so size/mtime alone can't be
# trusted to decide what's actually changed.
#
# --delete is opt-in (only once you've confirmed nothing important gets
# removed on the host -- /media/ and other host-local state are excluded
# via .gitignore regardless).
#
# Also pushes demo.example.json over the live /media/demo.json: the mixer
# reads /media/demo.json as-is on every (re)start and never re-derives it
# from the template on its own (pyplumber/mixer/gui/setup_runtime.py
# SetupRuntime.resume() -> apply(): recipe_path.exists() short-circuits
# straight to the stored file). A restart alone never picks up a template
# edit; the file demo.json itself has to change. This always overwrites
# whatever was last applied through the web UI's setup.html.
#
# Usage:
#   scripts/sync.sh            # sync, never delete
#   scripts/sync.sh --delete   # also remove files on the host not present locally
#   scripts/sync.sh --dry-run  # preview only (code sync and demo.json push both skipped)

. "$(dirname "$0")/_lib.sh"

DELETE_ARGS=""
DRYRUN_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --delete)  DELETE_ARGS="--delete" ;;
    --dry-run) DRYRUN_ARGS="--dry-run" ;;
    *) die "unknown argument: $arg" ;;
  esac
done

export COPYFILE_DISABLE=1

log "syncing ${REPO_ROOT} -> ${EC2_USER}@${EC2_HOST}:${EC2_AVPLUMBER}"
rsync -avz --checksum $DELETE_ARGS $DRYRUN_ARGS \
  -e "ssh -i $EC2_PEM" \
  --exclude='.git' --exclude-from="${REPO_ROOT}/.gitignore" \
  "${REPO_ROOT}/" \
  "${EC2_USER}@${EC2_HOST}:${EC2_AVPLUMBER}/"

if [[ -z "$DRYRUN_ARGS" ]]; then
  log "refreshing live /media/demo.json from demo.example.json (overwrites any web-UI-applied settings)"
  ssh_run "mkdir -p '${EC2_AVPLUMBER}/media' && cp -f \
    '${EC2_AVPLUMBER}/demos/mixer_audio/demo.example.json' '${EC2_AVPLUMBER}/media/demo.json'"
fi

log "done"
