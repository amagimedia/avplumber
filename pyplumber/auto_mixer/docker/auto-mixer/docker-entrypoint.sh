#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_ROOT="${AVP_MODEL_DIR:-/models}"
readonly TRTEXEC="${TRTEXEC:-/opt/tensorrt/bin/trtexec}"
readonly REBUILD_FACE_ENGINE="${AVP_REBUILD_FACE_ENGINE:-0}"
readonly TRT_AVG_TIMING="${TRT_AVG_TIMING:-1}"
readonly TENSORRT_TIMING_CACHE="${AVP_TENSORRT_TIMING_CACHE:-${MODEL_ROOT}/tensorrt.timing.cache}"

die() {
    echo "error: $*" >&2
    exit 1
}

check_gpu_runtime() {
    [[ -e /dev/nvidiactl ]] || die "GPU device /dev/nvidiactl is missing; run with NVIDIA runtime and --gpus all"

    if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so'; then
        if ! find /usr/lib64 /usr/lib /usr/local/nvidia/lib64 -maxdepth 2 -name 'libcuda.so*' -print -quit 2>/dev/null | grep -q .; then
            die "libcuda.so is not visible inside the container"
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed; host GPU runtime is not healthy"
    fi
}

find_onnx() {
    if [[ -n "${AVP_FACE_ONNX:-}" ]]; then
        [[ -f "${AVP_FACE_ONNX}" ]] || die "AVP_FACE_ONNX does not exist: ${AVP_FACE_ONNX}"
        printf '%s\n' "${AVP_FACE_ONNX}"
        return
    fi

    mapfile -t onnx_files < <(find "${MODEL_ROOT}" -maxdepth 1 -type f -name '*.onnx' | sort)
    case "${#onnx_files[@]}" in
        0)
            die "no ONNX file found in ${MODEL_ROOT}; mount exactly one .onnx file or set AVP_FACE_ONNX"
            ;;
        1)
            printf '%s\n' "${onnx_files[0]}"
            ;;
        *)
            die "multiple ONNX files found in ${MODEL_ROOT}; mount exactly one .onnx file or set AVP_FACE_ONNX"
            ;;
    esac
}

prepare_face_engine() {
    export LD_LIBRARY_PATH="/opt/tensorrt/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

    if [[ -n "${AVP_FACE_ENGINE:-}" && -s "${AVP_FACE_ENGINE}" && "${REBUILD_FACE_ENGINE}" != "1" ]]; then
        echo "using cached face TensorRT engine: ${AVP_FACE_ENGINE}"
        return
    fi

    [[ -x "${TRTEXEC}" ]] || die "trtexec is missing or not executable: ${TRTEXEC}"
    compgen -G "/opt/tensorrt/lib/libnvonnxparser.so*" >/dev/null || die "TensorRT ONNX parser is missing from /opt/tensorrt/lib"

    check_gpu_runtime

    mkdir -p "${MODEL_ROOT}"
    onnx="$(find_onnx)"
    stem="$(basename "${onnx}")"
    stem="${stem%.*}"

    engine="${AVP_FACE_ENGINE:-${MODEL_ROOT}/${stem}.fp16.plan}"
    tmp_engine="${engine}.tmp.$$"
    mkdir -p "$(dirname "${engine}")" "$(dirname "${TENSORRT_TIMING_CACHE}")"
    rm -f "${engine}".tmp.*

    if [[ "${REBUILD_FACE_ENGINE}" != "1" && -s "${engine}" ]]; then
        echo "using cached face TensorRT engine: ${engine}"
        export AVP_FACE_ENGINE="${engine}"
        return
    fi

    echo "building face TensorRT engine: ${onnx} -> ${engine}"
    if ! "${TRTEXEC}" \
        --onnx="${onnx}" \
        --saveEngine="${tmp_engine}" \
        --fp16 \
        --skipInference \
        --avgTiming="${TRT_AVG_TIMING}" \
        --timingCacheFile="${TENSORRT_TIMING_CACHE}"; then
        rm -f "${tmp_engine}"
        return 1
    fi

    [[ -s "${tmp_engine}" ]] || die "trtexec did not create expected plan: ${tmp_engine}"
    mv -f "${tmp_engine}" "${engine}"
    export AVP_FACE_ENGINE="${engine}"
    echo "face_engine=${AVP_FACE_ENGINE}"
}

if [[ "$#" -eq 0 ]]; then
    exec python3 -m pyplumber.auto_mixer.cli --help
fi

if [[ "${1:0:1}" == "-" ]]; then
    case "$1" in
        -h|--help)
            exec python3 -m pyplumber.auto_mixer.cli "$@"
            ;;
    esac
    prepare_face_engine
    exec python3 -m pyplumber.auto_mixer.cli "$@"
fi

exec "$@"
