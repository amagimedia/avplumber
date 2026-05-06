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
require_env AVP_MODELS_TAR_URL

case "${AVP_EXAMPLE}" in
    tracker|tracker-cropped|tracker_compositor) ;;
    *) die "unsupported AVP_EXAMPLE=${AVP_EXAMPLE}; expected tracker, tracker-cropped, or tracker_compositor" ;;
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
)

have_all_models=1
for path in "${required_models[@]}"; do
    if [[ ! -f "${path}" ]]; then
        have_all_models=0
        break
    fi
done

if [[ "${have_all_models}" -ne 1 ]]; then
    tmpdir="$(mktemp -d)"
    archive="${tmpdir}/models.tar.gz"
    extract_dir="${tmpdir}/extract"
    trap 'rm -rf "${tmpdir}"' EXIT
    mkdir -p "${MODEL_ROOT}"
    mkdir -p "${extract_dir}"
    curl -fsSL "${AVP_MODELS_TAR_URL}" -o "${archive}" || die "failed to download model archive from ${AVP_MODELS_TAR_URL}"
    tar -xzf "${archive}" -C "${extract_dir}" || die "failed to extract model archive"

    find_model() {
        local target="$1"
        find "${extract_dir}" -type f -name "${target}" -print -quit
    }

    for filename in ball_960x544.plan court-segmentation_960x544.plan basketball-players-full_960x544.plan; do
        if [[ ! -f "${MODEL_ROOT}/${filename}" ]]; then
            source_path="$(find_model "${filename}" || true)"
            [[ -n "${source_path}" ]] || die "required model ${filename} not found in extracted archive"
            cp -f "${source_path}" "${MODEL_ROOT}/${filename}"
        fi
    done

    rm -rf "${tmpdir}"
    trap - EXIT
fi

for path in "${required_models[@]}"; do
    [[ -f "${path}" ]] || die "required model missing after setup: ${path}"
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
    artifact_dir="/tmp/avp-sidecars"
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
    "${template_path}" > "${rendered}"

echo "example=${AVP_EXAMPLE}"
echo "mode=${AVP_MODE}"
echo "input=${AVP_INPUT}"
echo "output=${AVP_OUTPUT}"
echo "models_dir=${MODEL_ROOT}"
echo "models_status=ready"

exec /usr/local/bin/avplumber -s "${rendered}"
