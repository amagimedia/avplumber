#!/usr/bin/env bash
# Apply the matching patch series to a supported upstream FFmpeg checkout.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 /path/to/FFmpeg" >&2; exit 2; }
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
src=$1
source "$dir/9/bases.env"
series=8
base=$(git -C "$src" rev-parse HEAD)
if [[ $base == "$upstream_commit" ]]; then
    series=9
elif ! grep -Eq '^#define LIBAVCODEC_VERSION_MAJOR[[:space:]]+62$' "$src/libavcodec/version_major.h"; then
    echo "unsupported FFmpeg base: $base (use n8.0, n8.1, or $upstream_ref)" >&2
    exit 2
fi
git -C "$src" -c user.name="avplumber patches" -c user.email="patches@local" \
    am --whitespace=nowarn "$dir"/"$series"/*.patch
[[ $series == 8 ]] || exit 0
# FFmpeg 8.1 added a libnpp configure check that fails on CUDA 13 (the legacy
# nppiYCbCr420_8u_P2P3R symbol is gone); probe the stream-context API instead.
# 8.0 has no such check, so this is a no-op there.
sed -i.bak \
    -e 's/check_func_headers "nppi.h" nppiYCbCr420_8u_P2P3R \$libnpp_extralibs/check_func_headers "nppi.h" nppiYCbCr420_8u_P2P3R_Ctx $libnpp_extralibs/' \
    -e 's/libnpp support is deprecated, version 13.0 and up are not supported/libnpp stream context APIs not found/' \
    "$src/configure" && rm -f "$src/configure.bak"
if ! git -C "$src" diff --quiet -- configure; then
    git -C "$src" -c user.name="avplumber patches" -c user.email="patches@local" \
        commit -qam "configure: probe the libnpp stream-context API (CUDA 13)"
fi
