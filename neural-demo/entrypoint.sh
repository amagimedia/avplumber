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
        *) die "unsupported live output URL scheme for AVP_OUTPUT=${AVP_OUTPUT}" ;;
    esac
fi

mkdir -p "${artifact_dir}"
output_name="$(basename "${AVP_OUTPUT}")"
output_stem="${output_name%.*}"
if [[ "${output_stem}" == "${output_name}" ]]; then
    output_stem="${output_name}"
fi
video_label="${AVP_VIDEO_LABEL:-${AVP_INPUT##*/}}"

template_path="${TEMPLATE_ROOT}/${AVP_EXAMPLE}-${AVP_MODE}.avplumber"
[[ -f "${template_path}" ]] || die "template not found: ${template_path}"

rendered="${RENDER_DIR}/${AVP_EXAMPLE}-${AVP_MODE}.avplumber"
sed \
    -e "s|__INPUT__|${AVP_INPUT}|g" \
    -e "s|__PIP_INPUT__|${AVP_PIP_INPUT:-}|g" \
    -e "s|__OUTPUT__|${AVP_OUTPUT}|g" \
    -e "s|__OUTPUT_FORMAT__|${output_format}|g" \
    -e "s|__MODELS_DIR__|${MODEL_ROOT}|g" \
    -e "s|__BALL_TRACK_DUMP__|${artifact_dir}/${output_stem}_ball.csv|g" \
    -e "s|__JOIN_PLAYERS_BALL_DUMP__|${artifact_dir}/${output_stem}_join_players_ball_pts.csv|g" \
    -e "s|__JOIN_INFERRED_DUMP__|${artifact_dir}/${output_stem}_join_inferred_pts.csv|g" \
    -e "s|__JOIN_1080P_TRACKED_DUMP__|${artifact_dir}/${output_stem}_join_1080p_tracked_pts.csv|g" \
    -e "s|__METADATA_DUMP__|${artifact_dir}/${output_stem}_metadata.json|g" \
    -e "s|__METADATA_DUMP_COURT__|${artifact_dir}/${output_stem}_metadata_court.json|g" \
    -e "s|__METADATA_DUMP_OUTLINES__|${artifact_dir}/${output_stem}_metadata_outlines.json|g" \
    -e "s|__METADATA_DUMP_TRAIL__|${artifact_dir}/${output_stem}_metadata_trail.json|g" \
    -e "s|__METADATA_DUMP_EVENTS__|${artifact_dir}/${output_stem}_metadata_events.ndjson|g" \
    -e "s|__METADATA_DUMP_POSSESSIONS__|${artifact_dir}/${output_stem}_metadata_possessions.ndjson|g" \
    -e "s|__METADATA_DUMP_PBP__|${artifact_dir}/${output_stem}_metadata_pbp.ndjson|g" \
    -e "s|__METADATA_DUMP_SUMMARY__|${artifact_dir}/${output_stem}_metadata_summary.json|g" \
    -e "s|__VIDEO_LABEL__|${video_label}|g" \
    "${template_path}" > "${rendered}"

echo "example=${AVP_EXAMPLE}"
echo "mode=${AVP_MODE}"
echo "input=${AVP_INPUT}"
echo "output=${AVP_OUTPUT}"
echo "artifact_dir=${artifact_dir}"
echo "models_dir=${MODEL_ROOT}"
echo "models_status=ready"

exec /usr/local/bin/avplumber -s "${rendered}"
