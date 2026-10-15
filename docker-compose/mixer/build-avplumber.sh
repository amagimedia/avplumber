#!/bin/bash
# Build the Rust avplumber binary against the FFmpeg this image just installed.
# libavcodec 62 (FFmpeg 8) selects the ffmpeg8 feature, 63 (FFmpeg 9) selects ffmpeg9.
set -euo pipefail
cd /build
export PATH="/usr/local/cuda/bin:${PATH}"
export NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
export FFMPEG_PREFIX=/usr/local
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export FFMPEG_PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
export CARGO_TERM_COLOR=never

major="$(python3 - <<'PY'
import re
from pathlib import Path
text = Path("/usr/local/include/libavcodec/version_major.h").read_text()
match = re.search(r"#define LIBAVCODEC_VERSION_MAJOR\s+(\d+)", text)
if not match:
    raise SystemExit("LIBAVCODEC_VERSION_MAJOR missing")
print(match.group(1))
PY
)"
case "${major}" in
    62) feature=ffmpeg8 ;;
    63) feature=ffmpeg9 ;;
    *)
        echo "libavcodec major ${major} has no avplumber feature; this image builds 62 (ffmpeg8) or 63 (ffmpeg9)" >&2
        exit 1
        ;;
esac

cmake_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DAV_BUILD_EXAMPLES=OFF
    -DAV_ENABLE_SHARED=OFF
)
if [[ -n "${CXXFLAGS:-}" ]]; then
    cmake_flags+=("-DCMAKE_CXX_FLAGS=${CXXFLAGS}")
fi
rm -rf deps/avcpp/build
cmake -S deps/avcpp -B deps/avcpp/build "${cmake_flags[@]}"
cmake --build deps/avcpp/build --target avcpp -j"${BUILD_JOBS:-4}"
test -f deps/avcpp/build/src/libavcpp.a

cargo build --release -p avplumber_nodes --features "${feature},cpp_mixer" --bin avplumber --jobs "${BUILD_JOBS:-4}"
install -m 0755 target/release/avplumber /usr/local/bin/avplumber
ldd /usr/local/bin/avplumber | grep -q '/usr/local/lib/libavcodec'
if ldd /usr/local/bin/avplumber | grep -q 'not found'; then
    ldd /usr/local/bin/avplumber
    exit 1
fi
