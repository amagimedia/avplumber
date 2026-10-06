#!/usr/bin/env bash
# Build a custom FFmpeg and install it where the Rust crates can link it.
#
# Source defaults to deps/ffmpeg (the submodule, branch master).
# Override with FFMPEG_SRC. Install prefix defaults to target/ffmpeg.
# Override with FFMPEG_PREFIX. Extra arguments are passed to ./configure
# after the defaults, so a patched tree or a different feature set is:
#
#   scripts/build-ffmpeg.sh --enable-gpl --enable-libx264
#   FFMPEG_SRC=/path/to/ffmpeg scripts/build-ffmpeg.sh
#
# The script writes target/ffmpeg.env. A matching selector is required:
#
#   source target/ffmpeg.env
#   cargo build -p avplumber_nodes --features ffmpeg9,async
#
# Rebuilds are skipped when the source commit and the configure arguments
# are unchanged. Pass --reconfigure to force a new configure+install.
# (--reconfigure is consumed by this script and not forwarded.)

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${FFMPEG_SRC:-$ROOT/deps/ffmpeg}
PREFIX=${FFMPEG_PREFIX:-$ROOT/target/ffmpeg}
BUILD=${FFMPEG_BUILD_DIR:-$ROOT/target/ffmpeg-build}

reconfigure=0
configure_args=()
for arg in "$@"; do
    if [[ "$arg" == "--reconfigure" ]]; then
        reconfigure=1
    else
        configure_args+=("$arg")
    fi
done

# FFmpeg refuses to configure without nasm unless x86 assembly is disabled.
# A host without nasm still gets a working build; install nasm and re-run for
# the hand-written kernels.
if ! command -v nasm >/dev/null 2>&1; then
    have_asm_flag=0
    for arg in "${configure_args[@]}"; do
        case "$arg" in
            --disable-x86asm|--enable-x86asm) have_asm_flag=1 ;;
        esac
    done
    if [[ "$have_asm_flag" -eq 0 ]]; then
        echo "nasm not found; configuring with --disable-x86asm" >&2
        configure_args+=(--disable-x86asm)
    fi
fi

if [[ ! -f "$SRC/configure" ]]; then
    echo "No FFmpeg source at $SRC." >&2
    echo "Expected the deps/ffmpeg submodule (git submodule update --init deps/ffmpeg)." >&2
    exit 1
fi

SRC=$(cd "$SRC" && pwd)
mkdir -p "$BUILD" "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
BUILD=$(cd "$BUILD" && pwd)

config_id=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo no-git)
config_id+=" ${configure_args[*]-}"
stamp="$PREFIX/avplumber-build-stamp"
pc="$PREFIX/lib/pkgconfig/libavcodec.pc"

if [[ "$reconfigure" -eq 0 && -f "$pc" && -f "$stamp" && "$(cat "$stamp")" == "$config_id" ]]; then
    echo "FFmpeg already installed at $PREFIX"
else
    (
        cd "$BUILD"
        "$SRC/configure" \
            --prefix="$PREFIX" \
            --enable-shared \
            --disable-static \
            --disable-programs \
            --disable-doc \
            --disable-debug \
            --disable-autodetect \
            --enable-rpath \
            "${configure_args[@]}"
        make -j"$(nproc)"
        make install
    )
    printf '%s\n' "$config_id" > "$stamp"
fi

mkdir -p "$ROOT/target"
cat > "$ROOT/target/ffmpeg.env" << EOF
FFMPEG_PREFIX=$PREFIX
PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}
LD_LIBRARY_PATH=$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export FFMPEG_PREFIX PKG_CONFIG_PATH LD_LIBRARY_PATH
EOF

echo "Installed FFmpeg at $PREFIX"
echo "Source: $SRC"
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --modversion libavcodec libavformat libavutil
echo
echo "source target/ffmpeg.env"
echo "cargo build -p avplumber_nodes --features ffmpeg9,async"
