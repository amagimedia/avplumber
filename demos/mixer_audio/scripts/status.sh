#!/usr/bin/env bash
# Status/health check for the mixer_audio demo stack on the host.

. "$(dirname "$0")/_lib.sh"

ssh_run "cd '${EC2_AVPLUMBER}' && docker compose -f '${COMPOSE_REL}' ps"
ssh_run "cd '${EC2_AVPLUMBER}' && docker compose -f '${COMPOSE_REL}' logs --tail 50 mixer \
  | grep -E 'READY|running|mixer.cut|underrun|Error|preheat' || true"
