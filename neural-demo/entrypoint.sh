#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_ROOT="/home/tensorrt"
readonly TEMPLATE_ROOT="/opt/avp-neural-demo/templates"
readonly RENDER_DIR="/tmp/avp-rendered"

die() {
    echo "error: $*" >&2
    exit 1
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "missing required environment variable: ${name}"
}

is_false() {
    case "${1:-}" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF) return 0 ;;
        *) return 1 ;;
    esac
}

sed_escape() {
    printf '%s' "$1" | sed -e 's/[|&]/\\&/g'
}

require_env AVP_EXAMPLE
require_env AVP_MODE

case "${AVP_EXAMPLE}" in
    tracker|metadata|tracker-cropped|tracker_compositor) ;;
    *) die "unsupported AVP_EXAMPLE=${AVP_EXAMPLE}; expected tracker, metadata, tracker-cropped, or tracker_compositor" ;;
esac

case "${AVP_MODE}" in
    vod|live) ;;
    *) die "unsupported AVP_MODE=${AVP_MODE}; expected vod or live" ;;
esac

require_env AVP_INPUT
require_env AVP_OUTPUT

if [[ "${AVP_EXAMPLE}" == "tracker_compositor" ]]; then
    require_env AVP_PIP_INPUT
fi

if [[ ! -e /dev/nvidiactl ]]; then
    die "GPU device /dev/nvidiactl is missing; run with NVIDIA container runtime and --gpus all"
fi

if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so'; then
    if ! find /usr/lib64 /usr/lib /usr/local/nvidia/lib64 -maxdepth 2 -name 'libcuda.so*' -print -quit 2>/dev/null | grep -q .; then
        die "libcuda.so is not visible inside the container"
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed; host GPU runtime is not healthy"
fi

[[ -f /opt/tensorrt/lib/libnvinfer.so.10 ]] || [[ -f /opt/tensorrt/lib/libnvinfer.so ]] || die "TensorRT libnvinfer is missing"
[[ -f /opt/tensorrt/lib/libnvinfer_plugin.so.10 ]] || [[ -f /opt/tensorrt/lib/libnvinfer_plugin.so ]] || die "TensorRT libnvinfer_plugin is missing"

ffmpeg_filters="$("/usr/local/bin/ffmpeg" -hide_banner -filters 2>/dev/null)"
grep -q ' overlay_many_cuda ' <<<"${ffmpeg_filters}" || die "patched FFmpeg filter overlay_many_cuda is missing"

required_models=(
    "${MODEL_ROOT}/ball_960x544.plan"
    "${MODEL_ROOT}/court-segmentation_960x544.plan"
    "${MODEL_ROOT}/basketball-players-full_960x544.plan"
    "${MODEL_ROOT}/player-seg/player-seg_960x544.plan"
    "${MODEL_ROOT}/pose-small/pose-small.plan"
    "${MODEL_ROOT}/pose-small/pose-small_960x544.plan"
    "${MODEL_ROOT}/court-pose-4/court-pose.plan"
)

for path in "${required_models[@]}"; do
    if [[ ! -f "${path}" ]]; then
        die "required baked model is missing: ${path}; rebuild the image with neural-demo/build-neural-demo-image.sh --models-onnx /path/to/models_onnx.tgz"
    fi
done

mkdir -p "${RENDER_DIR}"

output_url="${AVP_OUTPUT}"
output_options="{}"
keyframe_interval_sec="3/1"
video_encoder_options='{ "b": "12000k", "maxrate": "12000k", "bufsize": "12000k", "rc": "cbr", "g": 75, "bf": 0, "preset": "p7", "profile": "high", "multipass": "disabled", "zerolatency": 1, "spatial_aq": 1, "temporal_aq": 1 }'

if [[ "${AVP_MODE}" == "vod" ]]; then
    artifact_dir="$(dirname "${AVP_OUTPUT}")"
    case "${AVP_OUTPUT}" in
        *.mp4) output_format="mp4" ;;
        *.ts|*.mpegts) output_format="mpegts" ;;
        *) die "unsupported vod output extension for AVP_OUTPUT=${AVP_OUTPUT}; expected .mp4 or .ts" ;;
    esac
else
    artifact_dir="${AVP_ARTIFACT_DIR:-/tmp/avp-sidecars}"
    mkdir -p "${artifact_dir}"
    case "${AVP_OUTPUT}" in
        rtmp://*|rtmps://*) output_format="flv" ;;
        srt://*) output_format="mpegts" ;;
        janus|janus://*)
            output_format="rtp"
            output_name="janus"
            janus_host="${AVP_JANUS_HOST:-127.0.0.1}"
            janus_video_port="${AVP_JANUS_VIDEO_PORT:-5004}"
            janus_video_rtcp_port="${AVP_JANUS_VIDEO_RTCP_PORT:-$((janus_video_port + 1))}"
            janus_video_pt="$(( ${AVP_JANUS_VIDEO_PT:-96} ))"
            janus_video_ssrc="$(( ${AVP_JANUS_VIDEO_SSRC:-0x41565001} ))"
            rtp_pkt_size="${AVP_RTP_PKT_SIZE:-1200}"
            output_url="rtp://${janus_host}:${janus_video_port}?pkt_size=${rtp_pkt_size}&rtcp_port=${janus_video_rtcp_port}"
            output_options="{\"payload_type\":${janus_video_pt},\"rtpflags\":\"skip_rtcp\",\"ssrc\":${janus_video_ssrc}}"
            keyframe_interval_sec="1/1"
            video_bitrate="${AVP_JANUS_VIDEO_BITRATE:-8000k}"
            video_encoder_options="{ \"b\": \"${video_bitrate}\", \"maxrate\": \"${video_bitrate}\", \"bufsize\": \"${video_bitrate}\", \"rc\": \"cbr\", \"g\": 30, \"bf\": 0, \"preset\": \"p6\", \"profile\": \"baseline\", \"level\": \"4.0\", \"tune\": \"ull\", \"rc-lookahead\": 0, \"zerolatency\": 1, \"delay\": 0, \"forced-idr\": 1, \"no-scenecut\": 1, \"strict_gop\": 1, \"aud\": 1, \"spatial-aq\": 1, \"temporal-aq\": 0 }"
            ;;
        *) die "unsupported live output URL scheme for AVP_OUTPUT=${AVP_OUTPUT}; expected rtmp://, rtmps://, srt://, or janus" ;;
    esac
fi

mkdir -p "${artifact_dir}"
output_name="${output_name:-$(basename "${AVP_OUTPUT}")}"
output_stem="${output_name%.*}"
if [[ "${output_stem}" == "${output_name}" ]]; then
    output_stem="${output_name}"
fi
video_label="${AVP_VIDEO_LABEL:-${AVP_INPUT##*/}}"

if is_false "${AVP_METADATA_DUMPS:-1}"; then
    ball_track_dump=""
    metadata_dump=""
    metadata_dump_court=""
    metadata_dump_outlines=""
    metadata_dump_trail=""
    metadata_dump_events=""
    metadata_dump_possessions=""
    metadata_dump_pbp=""
    metadata_dump_summary=""
else
    ball_track_dump="${artifact_dir}/${output_stem}_ball.csv"
    metadata_dump="${artifact_dir}/${output_stem}_metadata.json"
    metadata_dump_court="${artifact_dir}/${output_stem}_metadata_court.json"
    metadata_dump_outlines="${artifact_dir}/${output_stem}_metadata_outlines.json"
    metadata_dump_trail="${artifact_dir}/${output_stem}_metadata_trail.json"
    metadata_dump_events="${artifact_dir}/${output_stem}_metadata_events.ndjson"
    metadata_dump_possessions="${artifact_dir}/${output_stem}_metadata_possessions.ndjson"
    metadata_dump_pbp="${artifact_dir}/${output_stem}_metadata_pbp.ndjson"
    metadata_dump_summary="${artifact_dir}/${output_stem}_metadata_summary.json"
fi

template_path="${TEMPLATE_ROOT}/${AVP_EXAMPLE}-${AVP_MODE}.avplumber"
[[ -f "${template_path}" ]] || die "template not found: ${template_path}"

rendered="${RENDER_DIR}/${AVP_EXAMPLE}-${AVP_MODE}.avplumber"
sed \
    -e "s|__INPUT__|$(sed_escape "${AVP_INPUT}")|g" \
    -e "s|__PIP_INPUT__|$(sed_escape "${AVP_PIP_INPUT:-}")|g" \
    -e "s|__OUTPUT__|$(sed_escape "${output_url}")|g" \
    -e "s|__OUTPUT_FORMAT__|${output_format}|g" \
    -e "s|__OUTPUT_OPTIONS__|${output_options}|g" \
    -e "s|__KEYFRAME_INTERVAL_SEC__|${keyframe_interval_sec}|g" \
    -e "s|__VIDEO_ENCODER_OPTIONS__|${video_encoder_options}|g" \
    -e "s|__MODELS_DIR__|$(sed_escape "${MODEL_ROOT}")|g" \
    -e "s|__BALL_TRACK_DUMP__|$(sed_escape "${ball_track_dump}")|g" \
    -e "s|__JOIN_PLAYERS_BALL_DUMP__|$(sed_escape "${artifact_dir}/${output_stem}_join_players_ball_pts.csv")|g" \
    -e "s|__JOIN_INFERRED_DUMP__|$(sed_escape "${artifact_dir}/${output_stem}_join_inferred_pts.csv")|g" \
    -e "s|__JOIN_1080P_TRACKED_DUMP__|$(sed_escape "${artifact_dir}/${output_stem}_join_1080p_tracked_pts.csv")|g" \
    -e "s|__METADATA_DUMP__|$(sed_escape "${metadata_dump}")|g" \
    -e "s|__METADATA_DUMP_COURT__|$(sed_escape "${metadata_dump_court}")|g" \
    -e "s|__METADATA_DUMP_OUTLINES__|$(sed_escape "${metadata_dump_outlines}")|g" \
    -e "s|__METADATA_DUMP_TRAIL__|$(sed_escape "${metadata_dump_trail}")|g" \
    -e "s|__METADATA_DUMP_EVENTS__|$(sed_escape "${metadata_dump_events}")|g" \
    -e "s|__METADATA_DUMP_POSSESSIONS__|$(sed_escape "${metadata_dump_possessions}")|g" \
    -e "s|__METADATA_DUMP_PBP__|$(sed_escape "${metadata_dump_pbp}")|g" \
    -e "s|__METADATA_DUMP_SUMMARY__|$(sed_escape "${metadata_dump_summary}")|g" \
    -e "s|__VIDEO_LABEL__|$(sed_escape "${video_label}")|g" \
    "${template_path}" > "${rendered}"

echo "example=${AVP_EXAMPLE}"
echo "mode=${AVP_MODE}"
echo "input=${AVP_INPUT}"
echo "output=${output_url}"
echo "artifact_dir=${artifact_dir}"
echo "models_dir=${MODEL_ROOT}"
echo "models_status=ready"

avplumber_args=(/usr/local/bin/avplumber)
if [[ -n "${AVP_LOGFILE:-}" ]]; then
    mkdir -p "$(dirname "${AVP_LOGFILE}")"
    avplumber_args+=("--logfile" "${AVP_LOGFILE}")
fi
if [[ -n "${AVP_REMOTE_CONTROL_PORT:-}" && "${AVP_REMOTE_CONTROL_PORT}" != "0" ]]; then
    avplumber_args+=("--port" "${AVP_REMOTE_CONTROL_PORT}")
fi
if [[ -n "${AVP_WEBUI_API:-}" ]]; then
    avplumber_args+=("--webui-api" "${AVP_WEBUI_API}")
fi
if [[ -n "${AVP_INSTANCE_NAME:-}" ]]; then
    avplumber_args+=("--instance-name" "${AVP_INSTANCE_NAME}")
fi

exec "${avplumber_args[@]}" -s "${rendered}"
