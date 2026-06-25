#!/usr/bin/env bash
# Build the MediaPipe autoflip bridge library (CPU-only, no GPU/EGL) via Bazel.
# Called by the Makefile when HAVE_MEDIAPIPE_AUTOFLIP=1.
# Required env vars (set by Makefile):
#   MEDIAPIPE_REPO_URL, MEDIAPIPE_REV
#   MEDIAPIPE_BRIDGE_SOURCE_DIR  (abs path to deps/mediapipe-bridge/)
#   MEDIAPIPE_SRC_DIR            (abs path to deps/mediapipe-src/ — clone target)
#   MEDIAPIPE_INSTALL_DIR        (abs path to deps/mediapipe-autoflip/ — install target)

set -euo pipefail

MEDIAPIPE_REPO_URL=${MEDIAPIPE_REPO_URL:-https://github.com/google-ai-edge/mediapipe.git}
MEDIAPIPE_REV=${MEDIAPIPE_REV:-v0.10.35}
MEDIAPIPE_BRIDGE_SOURCE_DIR=${MEDIAPIPE_BRIDGE_SOURCE_DIR:-$(pwd)/deps/mediapipe-bridge}
MEDIAPIPE_SRC_DIR=${MEDIAPIPE_SRC_DIR:-$(pwd)/deps/mediapipe-src}
MEDIAPIPE_INSTALL_DIR=${MEDIAPIPE_INSTALL_DIR:-$(pwd)/deps/mediapipe-autoflip}

echo "=== MediaPipe autoflip bridge build ==="
echo "  rev=$MEDIAPIPE_REV"
echo "  src=$MEDIAPIPE_SRC_DIR"
echo "  install=$MEDIAPIPE_INSTALL_DIR"

# Clone if needed (shared with face mesh build)
if [ ! -d "$MEDIAPIPE_SRC_DIR/.git" ]; then
    echo "Cloning MediaPipe $MEDIAPIPE_REV ..."
    git clone --depth=1 --branch "$MEDIAPIPE_REV" "$MEDIAPIPE_REPO_URL" "$MEDIAPIPE_SRC_DIR"
fi

# Copy bridge sources into the mediapipe tree
BRIDGE_PKG="$MEDIAPIPE_SRC_DIR/avp_bridge"
mkdir -p "$BRIDGE_PKG"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_mesh_bridge.cc"      "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_mesh_bridge.h"       "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_detection_bridge.cc" "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_detection_bridge.h"  "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_autoflip_bridge.cc"       "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_autoflip_bridge.h"        "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/BUILD.bazel"                            "$BRIDGE_PKG/BUILD"

# Write .bazelrc additions if not already present
BAZELRC="$MEDIAPIPE_SRC_DIR/.bazelrc"
if ! grep -q 'avp_bazelrc_patch' "$BAZELRC" 2>/dev/null; then
cat >> "$BAZELRC" <<'BAZELRC_PATCH'
# avp_bazelrc_patch — added by build_mediapipe_autoflip.sh
build --repo_env=HERMETIC_PYTHON_VERSION=3.12
BAZELRC_PATCH
fi

cd "$MEDIAPIPE_SRC_DIR"

# Find OpenCV include dir (Fedora: /usr/include/opencv4)
OPENCV_INC=""
for d in /usr/include/opencv4 /usr/local/include/opencv4 /usr/include /usr/local/include; do
    if [ -f "$d/opencv2/core/version.hpp" ]; then
        OPENCV_INC="$d"; break
    fi
done
[ -n "$OPENCV_INC" ] && echo "OpenCV include: $OPENCV_INC"

BAZEL_BUILD_FLAGS=(
    --compilation_mode=opt
    --nocheck_visibility
)
[ -n "$OPENCV_INC" ] && BAZEL_BUILD_FLAGS+=(--copt="-I$OPENCV_INC")

echo "Building //avp_bridge:libavp_mediapipe_autoflip.so ..."
bazel build "${BAZEL_BUILD_FLAGS[@]}" \
    //avp_bridge:libavp_mediapipe_autoflip.so

BAZEL_BIN=$(bazel info --compilation_mode=opt bazel-bin)
mkdir -p "$MEDIAPIPE_INSTALL_DIR/lib"
cp -v "$BAZEL_BIN/avp_bridge/libavp_mediapipe_autoflip.so" \
      "$MEDIAPIPE_INSTALL_DIR/lib/"

echo "=== MediaPipe autoflip bridge build complete ==="
echo "  Library: $MEDIAPIPE_INSTALL_DIR/lib/libavp_mediapipe_autoflip.so"

touch "$MEDIAPIPE_INSTALL_DIR/.avp-autoflip-built"
