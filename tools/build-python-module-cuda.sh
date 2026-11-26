#!/bin/bash
# Build the Python module exactly as the CUDA mixer image does.
#
# The Makefile defaults HAVE_NVCC=0, because most builds do not have nvcc. The CUDA
# image must opt in; without it `cuda_rect_overlay` is a stub and engine.configure
# fails with "cuda_rect_overlay: compositing requires a build with HAVE_NVCC=1".
# Keep these flags in step with docker-compose/mixer/Dockerfile.fedora44.
#
# Run inside avplumber-builder:cuda-vmm-x264 (or another image with nvcc and the
# avplumber build dependencies):
#
#   docker run --rm -v "$PWD:/src" --entrypoint /bin/bash \
#     avplumber-builder:cuda-vmm-x264 -lc \
#     'cp /src/src/nodes/resample_audio.cpp /build/src/nodes/ && /src/tools/build-python-module-cuda.sh'
#
set -euo pipefail

cd /build

jobs="${BUILD_JOBS:-$(nproc)}"
hardening="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fcf-protection"

make clean
make -j"${jobs}" python_module \
    BUILD_TYPE=Release \
    HAVE_CUDA=1 \
    HAVE_DRM=1 \
    HAVE_GL=1 \
    HAVE_NVOF_FRUC=0 \
    HAVE_NVCC=1 \
    HAVE_NVJPEG=0 \
    HAVE_MXL=0 \
    HAVE_TENSORRT=0 \
    HAVE_CUDA_VMM_IPC=1 \
    NEURAL_NET=0 \
    TENSORRT_ROOT=/opt/tensorrt \
    HAVE_LIBFABRIC=1 \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
    CXXFLAGS+=" ${hardening}" \
    LFLAGS+=' -Wl,--disable-new-dtags'

module="_avplumber$(python3-config --extension-suffix)"
test -f "${module}"
grep -q '"cuda_rect_overlay"' graph_factory.python.generated.cpp
printf 'built %s\n' "$(readlink -f "${module}")"
