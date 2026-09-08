#!/usr/bin/env bash
# The replay demo on the Rust core, with Docker Compose: Janus and its browser
# preview in their own containers, the player (Rust avplumber + Python TUI) in
# another, run per session. No GPU.
#
#   demos/replay-rust/run.sh                         start Janus, make a sample recording if
#                                                    there is none, open the TUI
#   demos/replay-rust/run.sh up                      start Janus and the preview (detached)
#   demos/replay-rust/run.sh down                    stop them
#   demos/replay-rust/run.sh build                   build the player image (and Janus)
#   demos/replay-rust/run.sh sample [seconds] [fps]  a testsrc2 clip, transcoded into media/
#   demos/replay-rust/run.sh transcode <vod> [fps]   convert your own file into media/replay.ts
#   demos/replay-rust/run.sh play [player options]   the TUI (needs media/replay.ts)
#   demos/replay-rust/run.sh exercise                the RUN V2 checks, no TUI
#   demos/replay-rust/run.sh shell
#
# Environment: JANUS_HOST_IP (127.0.0.1; the address your browser reaches),
# JANUS_HTTP_PORT (8088), JANUS_PREVIEW_PORT (8080), JANUS_VIDEO_PORT (5004),
# JANUS_RTP_PORT_RANGE (20000-20100), REPLAY_MEDIA (demos/replay-rust/media).
set -euo pipefail

cd "$(dirname "$0")/../.."
readonly compose_file=demos/replay-rust/compose.yaml
readonly media="${REPLAY_MEDIA:-${PWD}/demos/replay-rust/media}"
export REPLAY_MEDIA="${media}"

compose() {
    docker compose -f "${compose_file}" "$@"
}

services_up() {
    compose up -d --build janus janus-preview
}

# `run` for the player: per session, removed afterwards, media mounted.
player() {
    mkdir -p "${media}"
    compose run --rm "$@"
}

command="${1:-}"
shift || true

case "${command}" in
    up)
        services_up
        ;;
    down)
        compose down
        ;;
    build)
        compose build replay janus janus-preview
        ;;
    sample)
        player replay sample "$@"
        ;;
    transcode)
        [[ $# -ge 1 ]] || { echo "usage: $0 transcode <vod> [fps]" >&2; exit 2; }
        input="$(realpath "$1")"
        fps="${2:-30}"
        player -v "$(dirname "${input}"):/input:ro,z" replay \
            transcode --input "/input/$(basename "${input}")" --output /media/replay.ts --fps "${fps}" --force
        ;;
    play)
        services_up
        player replay play --recording /media/replay.ts "$@"
        ;;
    exercise)
        services_up
        player replay exercise --recording /media/replay.ts "$@"
        ;;
    shell)
        player replay shell
        ;;
    "")
        services_up
        if [[ ! -f "${media}/replay.ts" ]]; then
            player replay sample
        fi
        echo "Open http://${JANUS_HOST_IP:-127.0.0.1}:${JANUS_PREVIEW_PORT:-8080} in a browser to watch."
        player replay play --recording /media/replay.ts
        ;;
    *)
        sed -n '2,19p' "$0" >&2
        exit 2
        ;;
esac
