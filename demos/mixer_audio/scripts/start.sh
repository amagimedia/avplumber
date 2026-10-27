#!/usr/bin/env bash
# (Re)start the mixer_audio demo stack (janus, janus-preview, wayland,
# dma-browser, mixer) on the host. Run scripts/build.sh first after any code
# change -- this does not rebuild the image.
#
# This demo binds the same ports (7681/7777/8088/5002-5004) as the
# audio_video_mixer demo on the same host; run
# scripts/stop-audio-video-mixer.sh first if that one is up.
#
# Usage:
#   scripts/start.sh                  # up -d (full stack, creates what's missing)
#   scripts/start.sh --recreate-mixer # also force-recreate just the mixer service

. "$(dirname "$0")/_lib.sh"

RECREATE_MIXER=0
for arg in "$@"; do
  case "$arg" in
    --recreate-mixer) RECREATE_MIXER=1 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

log "starting mixer_audio stack on ${EC2_HOST}"
ssh_run "cd '${EC2_AVPLUMBER}' && docker compose -f '${COMPOSE_REL}' up -d"

if (( RECREATE_MIXER )); then
  log "force-recreating mixer (picks up a freshly built image / /media/demo.json)"
  ssh_run "cd '${EC2_AVPLUMBER}' && docker compose -f '${COMPOSE_REL}' up -d --force-recreate mixer"
fi

log "done. URLs:"
log "  operator:       http://${EC2_HOST}:7681/"
log "  janus preview:  http://${EC2_HOST}:8080/"
