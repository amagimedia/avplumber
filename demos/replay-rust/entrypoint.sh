#!/usr/bin/env bash
# Inside the player image. Janus is another container; the player only sends
# RTP to JANUS_HOST:JANUS_VIDEO_PORT and, before starting, checks that Janus
# answers on its REST port so a missing gateway is reported at once.
#
#   replay-entrypoint play [player.py options]       the TUI
#   replay-entrypoint exercise [player.py options]   the RUN V2 checks, no TUI
#   replay-entrypoint transcode [transcode.py opts]  convert a VOD into a recording
#   replay-entrypoint sample [seconds] [fps] [size]  a testsrc2 clip into /media, transcoded
#   replay-entrypoint avplumber [args]               the Rust executable itself
#   replay-entrypoint shell                          bash
#
# REPLAY_BACKEND picks the codecs: `auto` (default), `nvidia` for NVDEC and
# NVENC with every frame kept on the GPU, or `cpu`.
set -euo pipefail

readonly demo_dir=/opt/avplumber/demos/replay
readonly media_dir=/media
readonly log_dir=/tmp/replay-demo
readonly backend="${REPLAY_BACKEND:-auto}"
mkdir -p "${log_dir}"

# With the NVIDIA devices passed in directly, the driver libraries are
# bind-mounted at their own paths but nothing has indexed them yet; ldconfig
# builds the soname links libcuda and friends are loaded by. The container
# toolkit does this itself, and then this is a no-op.
if [[ -e /dev/nvidiactl ]]; then
    ldconfig 2>/dev/null || true
fi

wait_for_janus() {
    local url="http://${JANUS_HOST}:${JANUS_HTTP_PORT}/janus/info"
    for _ in $(seq 1 50); do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "[replay] Janus answers on ${url}; sending RTP to ${JANUS_HOST}:${JANUS_VIDEO_PORT}"
            return 0
        fi
        sleep 0.2
    done
    echo "[replay] no Janus on ${url}: start it first (demos/replay-rust/run.sh up); streaming anyway" >&2
}

# The container runs as root; give the files it wrote to whoever owns the
# mounted media directory, so the host user can delete them.
own_media() {
    chown --reference="${media_dir}" "${media_dir}"/* 2>/dev/null || true
}

player_args() {
    echo --janus-host "${JANUS_HOST}" --janus-video-port "${JANUS_VIDEO_PORT}" \
         --backend "${backend}" --janus-bitrate "${JANUS_BITRATE:-4000k}" \
         --avplumber-log "${log_dir}/avplumber.log"
}

mode="${1:-play}"
shift || true
cd "${demo_dir}"

case "${mode}" in
    play)
        wait_for_janus
        # shellcheck disable=SC2046
        exec python3 player.py $(player_args) "$@"
        ;;
    exercise)
        wait_for_janus
        # shellcheck disable=SC2046
        exec python3 player.py --no-tui --exercise-v2 $(player_args) "$@"
        ;;
    transcode)
        python3 transcode.py --backend "${backend}" "$@"
        own_media
        ;;
    sample)
        seconds="${1:-20}"
        fps="${2:-30}"
        size="${3:-640x360}"
        source="${media_dir}/source.mp4"
        echo "[replay] generating ${seconds} s of ${size} testsrc2 at ${fps} fps into ${source}"
        # The recording is all-intra, so a big picture makes a big file: about
        # 30 MB per second at 1080p60. `-threads 0` lets x264 use the machine.
        ffmpeg -nostdin -y -v error -f lavfi -i "testsrc2=size=${size}:rate=${fps}:duration=${seconds}" \
            -c:v libx264 -preset veryfast -g 15 -pix_fmt yuv420p "${source}"
        python3 transcode.py --backend "${backend}" \
            --input "${source}" --output "${media_dir}/replay.ts" --fps "${fps}" --force
        own_media
        ;;
    avplumber)
        exec /usr/local/bin/avplumber "$@"
        ;;
    shell)
        exec bash "$@"
        ;;
    *)
        sed -n '2,11p' "$0" >&2
        exit 2
        ;;
esac
