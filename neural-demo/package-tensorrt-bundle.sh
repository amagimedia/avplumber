#!/usr/bin/env bash
set -euo pipefail

source_dir="/opt/tensorrt"
output="tensorrt-minimal-with-tools.tgz"
builder_resources=()

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  neural-demo/package-tensorrt-bundle.sh \
    [--source /opt/tensorrt] \
    [--builder-resource sm75] \
    [--output /path/to/tensorrt-minimal-with-tools.tgz]

Creates the TensorRT archive consumed by the neural demo Docker build. It keeps
the archive limited to headers, runtime libs, the ONNX parser, trtexec, and
the TensorRT builder resource libraries needed for the target GPU architecture.
Pass --builder-resource all only when one archive must support multiple GPU
architectures.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            source_dir="${2:-}"
            shift 2
            ;;
        --output)
            output="${2:-}"
            shift 2
            ;;
        --builder-resource)
            builder_resources+=("${2:-}")
            shift 2
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

[[ -n "${source_dir}" ]] || die "--source must not be empty"
[[ -n "${output}" ]] || die "--output must not be empty"
[[ -d "${source_dir}/include" ]] || die "missing directory: ${source_dir}/include"
[[ -x "${source_dir}/bin/trtexec" ]] || die "missing executable: ${source_dir}/bin/trtexec"
compgen -G "${source_dir}/lib/libnvinfer.so*" >/dev/null || die "missing ${source_dir}/lib/libnvinfer.so*"
compgen -G "${source_dir}/lib/libnvinfer_plugin.so*" >/dev/null || die "missing ${source_dir}/lib/libnvinfer_plugin.so*"
compgen -G "${source_dir}/lib/libnvonnxparser.so*" >/dev/null || die "missing ${source_dir}/lib/libnvonnxparser.so*"
[[ "${#builder_resources[@]}" -gt 0 ]] || die "--builder-resource is required, for example --builder-resource sm75"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

bundle="${tmpdir}/TensorRT-minimal"
mkdir -p "${bundle}/bin" "${bundle}/include" "${bundle}/lib"
cp -a "${source_dir}/bin/trtexec" "${bundle}/bin/"
cp -a "${source_dir}/include/." "${bundle}/include/"
cp -a "${source_dir}"/lib/libnvinfer.so* "${bundle}/lib/"
cp -a "${source_dir}"/lib/libnvinfer_plugin.so* "${bundle}/lib/"
cp -a "${source_dir}"/lib/libnvonnxparser.so* "${bundle}/lib/"

shopt -s nullglob
for resource in "${builder_resources[@]}"; do
    [[ -n "${resource}" ]] || die "--builder-resource must not be empty"
    if [[ "${resource}" == "all" ]]; then
        matches=(
            "${source_dir}"/lib/libnvinfer_builder_resource_ptx.so*
            "${source_dir}"/lib/libnvinfer_builder_resource_sm*.so*
        )
    else
        matches=("${source_dir}"/lib/libnvinfer_builder_resource_"${resource}".so*)
    fi
    [[ "${#matches[@]}" -gt 0 ]] || die "missing TensorRT builder resource for ${resource}"
    cp -a "${matches[@]}" "${bundle}/lib/"
done

mkdir -p "$(dirname "$(realpath -m "${output}")")"
tar -C "${tmpdir}" -czf "${output}" TensorRT-minimal
ls -lh "${output}"
