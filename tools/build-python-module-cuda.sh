#!/bin/bash
# Build the Python module exactly as the CUDA mixer image does, including the
# amagi dyremix_mixer node and the CUDA replay nodes.
#
# The Makefile defaults HAVE_NVCC=0 and HAVE_CUDA_VMM_IPC=0, because most builds
# do not need them. This engine does:
#   - HAVE_NVCC=1          -> cuda_rect_overlay is a real compositor, not a stub
#   - HAVE_CUDA_VMM_IPC=1  -> cuda_ipc_source for CUDA VMM replay slots
#   - dyremix_mixer         -> amagi avplumber-nodes, linked against statically
#                             built dyremix
#
# A module missing any of them makes engine.configure fail and the project never
# initializes. Keep these in step with docker-compose/mixer/Dockerfile.fedora44
# and tools/Dockerfile.cuda-dyremix-module.
#
# Expected layout (the Dockerfile creates it):
#   /build                       avplumber source + deps + pyplumber
#   /build/dyremix-node          amagi_avplumber_nodes/dyremix (Makefile.inc + nodes)
#   /dyremix-install             dyremix static install (dyremix-static.pc)
#
# Run inside avplumber-builder:cuda-vmm-x264 after `make deps` and after the
# patched src/nodes/resample_audio.cpp has been copied in.
set -euo pipefail

cd /build

jobs="${BUILD_JOBS:-$(nproc)}"
hardening="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fcf-protection"
dyremix_node_mk="${DYREMIX_NODE_MK:-/build/dyremix-node/Makefile.inc}"
dyremix_pc_dir="${DYREMIX_PKG_CONFIG_DIR:-/dyremix-install/lib64/pkgconfig}"
export PKG_CONFIG_PATH="${dyremix_pc_dir}:${PKG_CONFIG_PATH:-}"

if [[ ! -f "${dyremix_node_mk}" ]]; then
    echo "missing ${dyremix_node_mk}: add avplumber-nodes/amagi_avplumber_nodes/dyremix to the build" >&2
    exit 2
fi
if ! pkg-config --exists dyremix-static; then
    echo "dyremix-static not found in ${dyremix_pc_dir}; build dyremix with BUILD_STATIC=ON first" >&2
    exit 2
fi

make clean
make -j"${jobs}" python_module \
    BUILD_TYPE=Release \
    HAVE_CUDA=1 \
    HAVE_DRM=1 \
    HAVE_GL=1 \
    HAVE_NVOF_FRUC=0 \
    HAVE_NVCC=1 \
    HAVE_CUDA_VMM_IPC=1 \
    HAVE_NVJPEG=0 \
    HAVE_MXL=0 \
    HAVE_TENSORRT=0 \
    NEURAL_NET=0 \
    TENSORRT_ROOT=/opt/tensorrt \
    HAVE_LIBFABRIC=1 \
    EXTRA_NODES_MK="${dyremix_node_mk}" \
    DYREMIX_PKG_CONFIG_PATH="${dyremix_pc_dir}" \
    PKG_CONFIG_PATH="${PKG_CONFIG_PATH}" \
    CXXFLAGS+=" ${hardening}" \
    LFLAGS+=' -Wl,--disable-new-dtags'

module="_avplumber$(python3-config --extension-suffix)"
test -f "${module}"
for node in cuda_rect_overlay cuda_ipc_source dyremix_mixer; do
    grep -q "\"${node}\"" graph_factory.python.generated.cpp \
        || { echo "missing node ${node} in the graph factory" >&2; exit 3; }
done
printf 'built %s\n' "$(readlink -f "${module}")"
