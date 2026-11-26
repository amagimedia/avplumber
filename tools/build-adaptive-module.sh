#!/bin/bash
# Build the adaptive studio-engine audio module with every feature the runtime
# needs, then (optionally) layer it over an engine image.
#
#   DYREMIX_DIR=/path/to/dyremix \
#   NODES_DIR=/path/to/avplumber-nodes \
#   ENGINE_BASE=studio-engine:avp-04f9ff8f-x264-aux-before-engui5 \
#   ENGINE_TAG=studio-engine:avp-04f9ff8f-x264-aux-adaptive \
#   tools/build-adaptive-module.sh
#
# ENGINE_BASE must already carry the adaptive Python config
# (pyplumber/mixer/inputs.py compensation=-1). Without ENGINE_BASE only the
# module image is produced.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${repo}/.." && pwd)"
dyremix_dir="${DYREMIX_DIR:-${workspace}/dyremix}"
nodes_dir="${NODES_DIR:-${workspace}/avplumber-nodes}"
node_dir="${nodes_dir}/amagi_avplumber_nodes/dyremix"
# dyremix 815ff67 adds dyremix_param_curve_t, dyremix_set_connection_gain_ex and
# the ramp/curve API the amagi dyremix_mixer node compiles against. The RC
# branch does not have it.
dyremix_rev="${DYREMIX_REV:-815ff67}"
# avplumber-nodes 88d6bb6 is the dyremix_mixer with runtime ports and named feeds.
nodes_rev="${NODES_REV:-88d6bb6}"
engine_base="${ENGINE_BASE:-}"
engine_tag="${ENGINE_TAG:-studio-engine:avp-04f9ff8f-x264-aux-adaptive}"

git -C "${dyremix_dir}" rev-parse --verify "${dyremix_rev}^{commit}" >/dev/null
git -C "${nodes_dir}" rev-parse --verify "${nodes_rev}^{commit}" >/dev/null

context="$(mktemp -d)"
trap 'rm -rf "${context}"' EXIT

cp "${repo}/tools/Dockerfile.cuda-dyremix-module" "${context}/Dockerfile"
cp "${repo}/src/nodes/resample_audio.cpp" "${context}/resample_audio.cpp"
cp "${repo}/tools/build-python-module-cuda.sh" "${context}/build-python-module-cuda.sh"
mkdir -p "${context}/dyremix"
git -C "${dyremix_dir}" archive "${dyremix_rev}" | tar -x -C "${context}/dyremix"
mkdir -p "${context}/dyremix-node"
git -C "${nodes_dir}" archive "${nodes_rev}" amagi_avplumber_nodes/dyremix \
    | tar -x -C "${context}/dyremix-node" --strip-components=2

module_image="avplumber-module-adaptive:$(date +%Y%m%dT%H%M%S)"
docker build -t "${module_image}" "${context}"

suffix="$(docker run --rm --entrypoint python3 "${module_image}" \
    -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
container="$(docker create "${module_image}")"
trap 'docker rm -f "${container}" >/dev/null 2>&1 || true; rm -rf "${context}"' EXIT
docker cp "${container}:/build/_avplumber${suffix}" "${context}/_avplumber.so"
docker rm -f "${container}" >/dev/null

if [[ -n "${engine_base}" ]]; then
    layer="$(mktemp -d)"
    trap 'rm -rf "${context}" "${layer}"' EXIT
    cp "${context}/_avplumber.so" "${layer}/_avplumber.so"
    cat > "${layer}/Dockerfile" <<EOF
FROM ${engine_base}
COPY _avplumber.so /build/_avplumber${suffix}
EOF
    docker build -t "${engine_tag}" "${layer}"
    printf 'engine image: %s\n' "${engine_tag}"
fi

printf 'module image: %s\n' "${module_image}"
