#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_ROOT="${MODEL_ROOT:-/home/tensorrt}"
readonly TENSORRT_TIMING_CACHE="${TENSORRT_TIMING_CACHE:-}"
readonly TRT_AVG_TIMING="${TRT_AVG_TIMING:-1}"
readonly AVP_REBUILD_MODELS="${AVP_REBUILD_MODELS:-0}"
TRTEXEC="${TRTEXEC:-/opt/tensorrt/bin/trtexec}"

die() {
    echo "error: $*" >&2
    exit 1
}

archive="${1:-}"
[[ -n "${archive}" ]] || die "usage: $0 /path/to/models_onnx.tgz"
[[ -f "${archive}" ]] || die "model archive does not exist: ${archive}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

[[ -x "${TRTEXEC}" ]] || die "trtexec is missing or not executable: ${TRTEXEC}"
compgen -G "/opt/tensorrt/lib/libnvonnxparser.so*" >/dev/null || die "TensorRT ONNX parser is missing from /opt/tensorrt/lib"
export LD_LIBRARY_PATH="/opt/tensorrt/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${TENSORRT_TIMING_CACHE}" ]]; then
    mkdir -p "$(dirname "${TENSORRT_TIMING_CACHE}")"
fi

if [[ ! -e /dev/nvidiactl ]]; then
    die "GPU device /dev/nvidiactl is missing; run the plan builder with NVIDIA runtime and --gpus all"
fi

tar -xzf "${archive}" -C "${tmpdir}" || die "failed to extract ${archive}"

find_model() {
    local rel="$1"
    local exact="${tmpdir}/models_onnx/${rel}"
    if [[ -f "${exact}" ]]; then
        printf '%s\n' "${exact}"
        return 0
    fi

    local basename_rel
    basename_rel="$(basename "${rel}")"
    find "${tmpdir}" -type f -name "${basename_rel}" -print -quit
}

require_model() {
    local rel="$1"
    local path
    path="$(find_model "${rel}" || true)"
    [[ -n "${path}" ]] || die "required ONNX/support file not found in archive: ${rel}"
    printf '%s\n' "${path}"
}

build_engine() {
    local rel_onnx="$1"
    local rel_plan="$2"
    shift 2
    local onnx
    local plan
    local tmp_plan
    local -a trtexec_args

    onnx="$(require_model "${rel_onnx}")"
    plan="${MODEL_ROOT}/${rel_plan}"
    tmp_plan="${plan}.tmp.$$"
    mkdir -p "$(dirname "${plan}")"
    rm -f "${plan}".tmp.*

    if [[ "${AVP_REBUILD_MODELS}" != "1" && -s "${plan}" ]]; then
        echo "using cached TensorRT engine: ${rel_plan}"
        return 0
    fi

    trtexec_args=(
        --onnx="${onnx}"
        --saveEngine="${tmp_plan}"
        --fp16
        --skipInference
        --avgTiming="${TRT_AVG_TIMING}"
    )
    if [[ -n "${TENSORRT_TIMING_CACHE}" ]]; then
        trtexec_args+=(--timingCacheFile="${TENSORRT_TIMING_CACHE}")
    fi
    trtexec_args+=("$@")

    echo "building TensorRT engine: ${rel_onnx} -> ${rel_plan}"
    if ! "${TRTEXEC}" "${trtexec_args[@]}"; then
        rm -f "${tmp_plan}"
        return 1
    fi

    [[ -s "${tmp_plan}" ]] || die "trtexec did not create expected plan: ${tmp_plan}"
    mv -f "${tmp_plan}" "${plan}"
}

mkdir -p "${MODEL_ROOT}"

build_engine "ball_960x544.onnx" "ball_960x544.plan"
build_engine "basketball-players-full_960x544.onnx" "basketball-players-full_960x544.plan"
build_engine "court-segmentation_960x544.onnx" "court-segmentation_960x544.plan"
build_engine "player-seg/player-seg_960x544.onnx" "player-seg/player-seg_960x544.plan"
build_engine "pose-small/pose-small_960x544.onnx" "pose-small/pose-small_960x544.plan"
build_engine "court-pose-4/court-pose.onnx" "court-pose-4/court-pose.plan"

ln -sf "pose-small_960x544.plan" "${MODEL_ROOT}/pose-small/pose-small.plan"

ocr_onnx="$(find_model "en-ppocr-v4-rec/en_PP-OCRv3_rec.onnx")"
ocr_keys="$(find_model "en-ppocr-v4-rec/en_dict.txt")"
if [[ -n "${ocr_onnx}" || -n "${ocr_keys}" ]]; then
    [[ -n "${ocr_onnx}" && -n "${ocr_keys}" ]] || die "optional OCR files must be supplied together: en_PP-OCRv3_rec.onnx and en_dict.txt"
    build_engine "en-ppocr-v4-rec/en_PP-OCRv3_rec.onnx" "en-ppocr-v4-rec/en_PP-OCRv3_rec_48x320.plan" \
        --minShapes=x:1x3x48x320 \
        --optShapes=x:1x3x48x320 \
        --maxShapes=x:1x3x48x320
    mkdir -p "${MODEL_ROOT}/en-ppocr-v4-rec"
    cp -f "${ocr_keys}" "${MODEL_ROOT}/en-ppocr-v4-rec/en_dict.txt"
else
    echo "optional scoreboard OCR model not present; skipping OCR TensorRT engine"
fi

echo "models_dir=${MODEL_ROOT}"
find "${MODEL_ROOT}" -type f -o -type l | sort
