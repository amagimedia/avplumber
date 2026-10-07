#!/usr/bin/env bash
# Optionally build a custom FFmpeg and install it where the Rust crates can link it.
#
# System FFmpeg needs none of this. With FFMPEG_PREFIX unset, cargo uses
# pkg-config:
#
#   cargo build -p avplumber_nodes --features ffmpeg6,async
#
# A custom tree is used only when a source directory is given, as the first
# non-option argument or as FFMPEG_SRC. There is no default source path.
# Install prefix defaults to target/ffmpeg. Override with FFMPEG_PREFIX.
# If the checkout is a known patch base (deps/ffmpeg/8 and deps/ffmpeg/9),
# the matching series in deps/ffmpeg is applied before configure. A tree that
# is already patched, or not one of those bases, is left unchanged.
# Remaining arguments are passed to ./configure:
#
#   scripts/build-ffmpeg.sh /path/to/ffmpeg --enable-gpl --enable-libx264
#   FFMPEG_SRC=/path/to/ffmpeg scripts/build-ffmpeg.sh
#
# The script writes target/ffmpeg.env. Source it so later cargo builds link
# that install instead of the distro libraries, and pick the matching feature:
#
#   source target/ffmpeg.env
#   cargo build -p avplumber_nodes --features ffmpeg9,async
#
# Rebuilds are skipped when the source commit and the configure arguments
# are unchanged. Pass --reconfigure to force a new configure+install.
# (--reconfigure is consumed by this script and not forwarded.)

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

# git am the series only when HEAD is still the unpatched base. apply.sh
# exits 2 for every other commit, including a tree the series was already
# applied to.
apply_ffmpeg_patches() {
    local src=$1
    local apply=$ROOT/deps/ffmpeg/apply.sh
    local head
    [[ -f $apply ]] || return 0
    head=$(git -C "$src" rev-parse HEAD 2>/dev/null || true)
    [[ -n $head ]] || return 0
    local upstream9 n80 n81
    upstream9=$(sed -n 's/^upstream_commit=//p' "$ROOT/deps/ffmpeg/9/bases.env")
    n80=$(sed -n 's/^n80_commit=//p' "$ROOT/deps/ffmpeg/8/bases.env")
    n81=$(sed -n 's/^n81_commit=//p' "$ROOT/deps/ffmpeg/8/bases.env")
    if [[ $head != "$upstream9" && $head != "$n80" && $head != "$n81" ]]; then
        echo "FFmpeg $head is not a patch base; leaving the tree unchanged" >&2
        return 0
    fi
    echo "Applying deps/ffmpeg patches to $head"
    "$apply" "$src"
}
PREFIX=${FFMPEG_PREFIX:-$ROOT/target/ffmpeg}
BUILD=${FFMPEG_BUILD_DIR:-$ROOT/target/ffmpeg-build}

reconfigure=0
configure_args=()
src_arg=
for arg in "$@"; do
    if [[ "$arg" == "--reconfigure" ]]; then
        reconfigure=1
    elif [[ -z "$src_arg" && "$arg" != -* ]]; then
        src_arg=$arg
    else
        configure_args+=("$arg")
    fi
done

SRC=${src_arg:-${FFMPEG_SRC:-}}
if [[ -z "$SRC" ]]; then
    if [[ ${#configure_args[@]} -gt 0 ]]; then
        echo "Configure flags need a FFmpeg source directory." >&2
        echo "  scripts/build-ffmpeg.sh /path/to/ffmpeg ${configure_args[*]}" >&2
        exit 1
    fi
    echo "No FFmpeg source given; nothing was built."
    echo "System FFmpeg is used as-is:"
    echo "  cargo build -p avplumber_nodes --features ffmpeg6,async"
    echo "Pass a source directory to build a custom FFmpeg instead:"
    echo "  scripts/build-ffmpeg.sh /path/to/ffmpeg"
    exit 0
fi

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
    echo "No FFmpeg source at $SRC (expected a checkout with ./configure)." >&2
    exit 1
fi

SRC=$(cd "$SRC" && pwd)
mkdir -p "$BUILD" "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
BUILD=$(cd "$BUILD" && pwd)

apply_ffmpeg_patches "$SRC"

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
