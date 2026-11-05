#!/usr/bin/env bash
# Stop the newer audio_video_mixer demo (avplumber-nodes-2/demos/audio_video_mixer)
# if it's running on the host -- it binds the same ports (7681/7777/8088/
# 5002-5004) as this demo's own services. Safe to run even if it's already
# stopped.

. "$(dirname "$0")/_lib.sh"

AV_MIXER_COMPOSE_REL="avplumber-nodes-2/demos/audio_video_mixer/docker/compose.yaml"

log "stopping audio_video_mixer demo on ${EC2_HOST} (port collision guard)"
ssh_run "cd '${EC2_REPO}' && docker compose -f '${AV_MIXER_COMPOSE_REL}' down"
