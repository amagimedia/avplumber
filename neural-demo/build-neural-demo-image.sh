#!/usr/bin/env bash
set -euo pipefail

image="avplumber-neural-demo:latest"
planner_image="avplumber-neural-demo-plan-builder:latest"
models_onnx=""
tensorrt_archive=""
cache_dir="neural-demo/.model-build-cache"
gpu_arch=""
rebuild_models=0
dry_run=0
docker_build_extra=()

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  neural-demo/build-neural-demo-image.sh \
    --models-onnx /path/to/models_onnx.tgz \
    --tensorrt /path/to/tensorrt-minimal-with-tools.tgz \
    [--gpu-arch sm75] \
    [--cache-dir neural-demo/.model-build-cache] \
    [--rebuild-models] \
    [--image IMAGE] \
    [--planner-image IMAGE] \
    [--docker-build-extra ARG] \
    [--dry-run]

The models archive must contain portable ONNX files. This helper builds a
temporary plan-builder image, runs it with --gpus all to generate TensorRT
plans for the current GPU, then bakes those plans into the final runtime image.
The TensorRT archive must contain runtime libs, headers,
bin/trtexec, and lib/libnvonnxparser.so*.
Generated plans and the TensorRT timing cache are reused from --cache-dir.
EOF
}

abspath() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "$(pwd)" "${path#./}"
    fi
}

detect_gpu_arch() {
    local compute_cap
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' .')"
    [[ -n "${compute_cap}" ]] || return 1
    printf 'sm%s\n' "${compute_cap}"
}

file_sha() {
    sha256sum "$1" | awk '{print $1}'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-onnx)
            models_onnx="${2:-}"
            shift 2
            ;;
        --tensorrt)
            tensorrt_archive="${2:-}"
            shift 2
            ;;
        --gpu-arch)
            gpu_arch="${2:-}"
            shift 2
            ;;
        --cache-dir)
            cache_dir="${2:-}"
            shift 2
            ;;
        --rebuild-models)
            rebuild_models=1
            shift
            ;;
        --image)
            image="${2:-}"
            shift 2
            ;;
        --planner-image)
            planner_image="${2:-}"
            shift 2
            ;;
        --docker-build-extra)
            docker_build_extra+=("${2:-}")
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "${models_onnx}" ]] || die "--models-onnx is required"
[[ -n "${tensorrt_archive}" ]] || die "--tensorrt is required"
models_onnx="$(abspath "${models_onnx}")"
tensorrt_archive="$(abspath "${tensorrt_archive}")"
[[ -f "${models_onnx}" ]] || die "models archive does not exist: ${models_onnx}"
[[ -f "${tensorrt_archive}" ]] || die "TensorRT archive does not exist: ${tensorrt_archive}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ -z "${gpu_arch}" ]]; then
    if ! gpu_arch="$(detect_gpu_arch)"; then
        if [[ "${dry_run}" -eq 1 ]]; then
            gpu_arch="unknown"
        else
            die "could not detect GPU architecture; pass --gpu-arch sm75"
        fi
    fi
fi

models_sha="$(file_sha "${models_onnx}")"
tensorrt_sha="$(file_sha "${tensorrt_archive}")"
builder_sha="$(file_sha neural-demo/build-models.sh)"
cache_key="${gpu_arch}-${models_sha:0:16}-${tensorrt_sha:0:16}-${builder_sha:0:16}"
if [[ "${cache_dir}" = /* ]]; then
    cache_abs="${cache_dir}/${cache_key}"
else
    cache_abs="${repo_root}/${cache_dir}/${cache_key}"
fi
models_cache_abs="${cache_abs}/models"
timing_cache_abs="${cache_abs}/tensorrt.timing.cache"

build_inputs_rel="neural-demo/.docker-build-inputs"
build_inputs_abs="${repo_root}/${build_inputs_rel}"
tensorrt_staged_rel="${build_inputs_rel}/tensorrt-${tensorrt_sha:0:16}.tgz"
tensorrt_staged_abs="${repo_root}/${tensorrt_staged_rel}"
models_staged_rel="${build_inputs_rel}/models-${cache_key}"
models_staged_abs="${repo_root}/${models_staged_rel}"
host_uid="${SUDO_UID:-$(id -u)}"
host_gid="${SUDO_GID:-$(id -g)}"
cleanup() {
    rm -f "${tensorrt_staged_abs}" "${tensorrt_staged_abs}.tmp.$$"
    rm -rf "${models_staged_abs}"
    rmdir "${build_inputs_abs}" 2>/dev/null || true
}
trap cleanup EXIT

build_plan_cmd=(
    docker build -f neural-demo/Dockerfile --target plan-builder
    --build-arg "TENSORRT_ARCHIVE=${tensorrt_staged_rel}"
    -t "${planner_image}"
)
build_plan_cmd+=("${docker_build_extra[@]}" .)

run_plan_cmd=(
    docker run --rm --gpus all
    --user "${host_uid}:${host_gid}"
    -v "${models_onnx}:/run/avp/models_onnx.tgz:ro"
    -v "${models_cache_abs}:/home/tensorrt"
    -v "${cache_abs}:/run/avp/model-cache"
    -e "TENSORRT_TIMING_CACHE=/run/avp/model-cache/tensorrt.timing.cache"
    -e "AVP_REBUILD_MODELS=${rebuild_models}"
)
run_plan_cmd+=("${planner_image}" /run/avp/models_onnx.tgz)

build_runtime_cmd=(
    docker build -f neural-demo/Dockerfile --target runtime
    --build-arg "TENSORRT_ARCHIVE=${tensorrt_staged_rel}"
    --build-arg "PREBUILT_MODELS_DIR=${models_staged_rel}"
    -t "${image}"
)
build_runtime_cmd+=("${docker_build_extra[@]}" .)

printf 'Plan builder build:'
printf ' %q' "${build_plan_cmd[@]}"
printf '\n'
printf 'Plan generation run:'
printf ' %q' "${run_plan_cmd[@]}"
printf '\n'
printf 'Runtime build:'
printf ' %q' "${build_runtime_cmd[@]}"
printf '\n'
printf 'Model cache: %q\n' "${cache_abs}"
printf 'Timing cache: %q\n' "${timing_cache_abs}"

if [[ "${dry_run}" -eq 1 ]]; then
    exit 0
fi

rm -rf "${build_inputs_abs}"
mkdir -p "${models_cache_abs}" "${build_inputs_abs}" "${models_staged_abs}"
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R "${host_uid}:${host_gid}" "${cache_abs}"
fi
cp -p "${tensorrt_archive}" "${tensorrt_staged_abs}.tmp.$$"
mv -f "${tensorrt_staged_abs}.tmp.$$" "${tensorrt_staged_abs}"
chmod a+r "${tensorrt_staged_abs}"

"${build_plan_cmd[@]}"
"${run_plan_cmd[@]}"
rm -rf "${models_staged_abs}"
mkdir -p "${models_staged_abs}"
cp -a "${models_cache_abs}/." "${models_staged_abs}/"
"${build_runtime_cmd[@]}"

echo "image=${image}"
