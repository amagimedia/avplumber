#!/usr/bin/env bash
set -euo pipefail

image="avplumber-neural-demo:latest"
example=""
mode=""
input=""
output=""
models_tar_url="https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/models.tar.gz"
dry_run=0
docker_extra=()

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  neural-demo/run-neural-demo.sh \
    --example tracker|tracker-cropped \
    --mode vod|live \
    --input ... \
    --output ... \
    [--models-tar-url ...] \
    [--image IMAGE] \
    [--docker-extra ARG] \
    [--dry-run]
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)
            example="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --input)
            input="${2:-}"
            shift 2
            ;;
        --output)
            output="${2:-}"
            shift 2
            ;;
        --models-tar-url)
            models_tar_url="${2:-}"
            shift 2
            ;;
        --image)
            image="${2:-}"
            shift 2
            ;;
        --docker-extra)
            docker_extra+=("${2:-}")
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

[[ -n "${example}" ]] || die "--example is required"
[[ -n "${mode}" ]] || die "--mode is required"
[[ -n "${input}" ]] || die "--input is required"
[[ -n "${output}" ]] || die "--output is required"
case "${example}" in
    tracker|tracker-cropped) ;;
    *) die "--example must be tracker or tracker-cropped" ;;
esac

case "${mode}" in
    vod|live) ;;
    *) die "--mode must be vod or live" ;;
esac

docker_cmd=(docker run --rm --gpus all)
docker_cmd+=("${docker_extra[@]}")

if [[ "${mode}" == "vod" ]]; then
    input_host="$(abspath "${input}")"
    output_host="$(abspath "${output}")"
    [[ -f "${input_host}" ]] || die "input file does not exist: ${input_host}"
    output_dir="$(dirname "${output_host}")"
    mkdir -p "${output_dir}"

    input_container="/run/avp/input/$(basename "${input_host}")"
    output_container="/run/avp/output/$(basename "${output_host}")"

    docker_cmd+=(
        -v "${input_host}:${input_container}:ro"
        -v "${output_dir}:/run/avp/output"
        -e "AVP_INPUT=${input_container}"
        -e "AVP_OUTPUT=${output_container}"
    )
else
    case "${input}" in
        rtmp://*|rtmps://*|srt://*) ;;
        *) die "live --input must be an rtmp://, rtmps://, or srt:// URL" ;;
    esac
    case "${output}" in
        rtmp://*|rtmps://*|srt://*) ;;
        *) die "live --output must be an rtmp://, rtmps://, or srt:// URL" ;;
    esac

    docker_cmd+=(
        -e "AVP_INPUT=${input}"
        -e "AVP_OUTPUT=${output}"
    )
fi

docker_cmd+=(
    -e "AVP_EXAMPLE=${example}"
    -e "AVP_MODE=${mode}"
    -e "AVP_MODELS_TAR_URL=${models_tar_url}"
    "${image}"
)

printf 'Running:'
printf ' %q' "${docker_cmd[@]}"
printf '\n'

if [[ "${dry_run}" -eq 1 ]]; then
    exit 0
fi

exec "${docker_cmd[@]}"
