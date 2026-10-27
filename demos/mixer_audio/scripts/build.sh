#!/usr/bin/env bash
# Build the mixer_audio demo's avplumber-mixer:local image on the host.
#
# Launched detached (nohup+setsid+disown) and then polled over fresh ssh
# connections, rather than held open as one blocking `docker compose build`:
# a dropped SSH/VPN session kills a foreground build with it (SIGHUP to the
# buildkit client), silently -- see demos/audio_video_mixer/OPS.md's
# detached-build section for how this was found. The CUDA avplumber binary
# compiles the full C++ sources; a cold build can run well past 10 minutes.
#
# Completion is detected via an explicit `EXITCODE:N` sentinel line appended
# to the log after the build exits, not by grepping the build host's process
# list for our own command string: `pgrep -f` only excludes its own PID, not
# the wrapping ssh/shell process whose argv contains that same string --
# which makes it match its own invocation and falsely report "still running"
# forever.

. "$(dirname "$0")/_lib.sh"

BUILD_LOG="${EC2_REPO}/mixer-audio-build.log"

log "starting build on ${EC2_HOST} (detached, log: ${BUILD_LOG})"
ssh_run "cd '${EC2_AVPLUMBER}' && rm -f '${BUILD_LOG}' && \
  nohup setsid bash -c \"docker compose -f '${COMPOSE_REL}' build mixer > '${BUILD_LOG}' 2>&1; echo EXITCODE:\\\$? >> '${BUILD_LOG}'\" < /dev/null & disown"

log "polling (Ctrl-C stops watching, not the remote build)..."
while ! ssh_run "tail -1 '${BUILD_LOG}' 2>/dev/null | grep -q '^EXITCODE:'"; do
  sleep 10
done

EXIT_LINE="$(ssh_run "tail -1 '${BUILD_LOG}'")"
if [[ "$EXIT_LINE" == "EXITCODE:0" ]]; then
  log "build finished OK"
else
  die "build failed (${EXIT_LINE}) -- check ${BUILD_LOG} on ${EC2_HOST}"
fi
