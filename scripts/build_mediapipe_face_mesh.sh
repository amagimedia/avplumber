#!/usr/bin/env bash
# Build the MediaPipe face mesh bridge library via Bazel.
# Called by the Makefile when HAVE_MEDIAPIPE=1.
# Required env vars (set by Makefile):
#   MEDIAPIPE_REPO_URL, MEDIAPIPE_REV
#   MEDIAPIPE_BRIDGE_SOURCE_DIR  (abs path to deps/mediapipe-bridge/)
#   MEDIAPIPE_SRC_DIR            (abs path to deps/mediapipe-src/ — clone target)
#   MEDIAPIPE_INSTALL_DIR        (abs path to deps/mediapipe-face-mesh/ — install target)

set -euo pipefail

MEDIAPIPE_REPO_URL=${MEDIAPIPE_REPO_URL:-https://github.com/google-ai-edge/mediapipe.git}
MEDIAPIPE_REV=${MEDIAPIPE_REV:-v0.10.35}
MEDIAPIPE_BRIDGE_SOURCE_DIR=${MEDIAPIPE_BRIDGE_SOURCE_DIR:-$(pwd)/deps/mediapipe-bridge}
MEDIAPIPE_SRC_DIR=${MEDIAPIPE_SRC_DIR:-$(pwd)/deps/mediapipe-src}
MEDIAPIPE_INSTALL_DIR=${MEDIAPIPE_INSTALL_DIR:-$(pwd)/deps/mediapipe-face-mesh}

echo "=== MediaPipe bridge build ==="
echo "  rev=$MEDIAPIPE_REV"
echo "  src=$MEDIAPIPE_SRC_DIR"
echo "  install=$MEDIAPIPE_INSTALL_DIR"

# Clone if needed
if [ ! -d "$MEDIAPIPE_SRC_DIR/.git" ]; then
    echo "Cloning MediaPipe $MEDIAPIPE_REV ..."
    git clone --depth=1 --branch "$MEDIAPIPE_REV" "$MEDIAPIPE_REPO_URL" "$MEDIAPIPE_SRC_DIR"
fi

# Copy bridge sources into the mediapipe tree as package avp_bridge/
BRIDGE_PKG="$MEDIAPIPE_SRC_DIR/avp_bridge"
mkdir -p "$BRIDGE_PKG"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_mesh_bridge.cc"      "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_mesh_bridge.h"       "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_detection_bridge.cc" "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/avp_mediapipe_face_detection_bridge.h"  "$BRIDGE_PKG/"
cp "$MEDIAPIPE_BRIDGE_SOURCE_DIR/BUILD.bazel"                            "$BRIDGE_PKG/BUILD"

# Write .bazelrc additions — pin Python to 3.12 (v0.10.35 has no 3.13 lock file)
# and configure headless GPU (EGL, no X11)
BAZELRC="$MEDIAPIPE_SRC_DIR/.bazelrc"
if ! grep -q 'avp_bazelrc_patch' "$BAZELRC" 2>/dev/null; then
cat >> "$BAZELRC" <<'BAZELRC_PATCH'
# avp_bazelrc_patch — added by build_mediapipe_face_mesh.sh
build --repo_env=HERMETIC_PYTHON_VERSION=3.12
build:linux_gpu --define=MEDIAPIPE_DISABLE_GPU=0
build:linux_gpu --copt=-DMESA_EGL_NO_X11_HEADERS
build:linux_gpu --copt=-DEGL_NO_X11
BAZELRC_PATCH
fi

# Find OpenCV include dir (Fedora: /usr/include/opencv4)
OPENCV_INC=""
for d in /usr/include/opencv4 /usr/local/include/opencv4 /usr/include /usr/local/include; do
    if [ -f "$d/opencv2/core/version.hpp" ]; then
        OPENCV_INC="$d"; break
    fi
done
[ -n "$OPENCV_INC" ] && echo "OpenCV include: $OPENCV_INC"

cd "$MEDIAPIPE_SRC_DIR"

# Patch MediaPipe's gl_context_egl.cc for headless NVIDIA (all in one Python pass):
#   1. Add #include <EGL/eglext.h> and <vector>
#   2. EGL_DEPTH_SIZE 16 → 0  (NVIDIA headless EGL has no depth configs)
#   3. GetInitializedEglDisplay() tries EGL device platform before default
python3 - "$MEDIAPIPE_SRC_DIR/mediapipe/gpu/gl_context_egl.cc" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
if 'avp_headless_patch' in src:
    print("avp_headless_patch: already applied, skipping")
    sys.exit(0)

# 1. Insert extra headers after the EGL include
src = src.replace(
    '#include <EGL/egl.h>',
    '#include <EGL/egl.h>\n#include <EGL/eglext.h>  // avp_headless_patch\n#include <vector>         // avp_headless_patch',
    1
)

# 2. Drop depth and alpha requirements — headless NVIDIA device EGL has no D16/RGBA8888 configs
import re
d_before = src.count('EGL_DEPTH_SIZE, 16,')
src = re.sub(r'EGL_DEPTH_SIZE,\s*16,', 'EGL_DEPTH_SIZE, 0,  // avp_headless_patch', src)
d_after = src.count('EGL_DEPTH_SIZE, 16,')
print(f"avp_headless_patch: replaced {d_before - d_after} EGL_DEPTH_SIZE occurrences")

a_before = src.count('EGL_ALPHA_SIZE, 8,')
src = re.sub(r'EGL_ALPHA_SIZE,\s*8,', 'EGL_ALPHA_SIZE, 0,  // avp_headless_patch', src)
a_after = src.count('EGL_ALPHA_SIZE, 8,')
print(f"avp_headless_patch: replaced {a_before - a_after} EGL_ALPHA_SIZE occurrences")

# 3. Replace GetInitializedEglDisplay to try device platform first
# Use void* for device handles — EGLDeviceEXT may not be defined in hermetic sysroot
old_fn = """static absl::StatusOr<EGLDisplay> GetInitializedEglDisplay() {
  auto status_or_display = GetInitializedDefaultEglDisplay();
  return status_or_display;
}"""
new_fn = """static absl::StatusOr<EGLDisplay> GetInitializedEglDisplay() {
  // avp_headless_patch: try EGL_PLATFORM_DEVICE_EXT (NVIDIA headless) before default
  // Use void* for device handles to avoid EGLDeviceEXT dependency in hermetic build
#ifndef EGL_PLATFORM_DEVICE_EXT
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif
  using QueryDevicesFn = EGLBoolean (*)(EGLint, void**, EGLint*);
  using GetPlatformDisplayFn = EGLDisplay (*)(EGLenum, void*, const EGLint*);
  auto query_fn = reinterpret_cast<QueryDevicesFn>(
      eglGetProcAddress("eglQueryDevicesEXT"));
  auto platform_fn = reinterpret_cast<GetPlatformDisplayFn>(
      eglGetProcAddress("eglGetPlatformDisplayEXT"));
  if (query_fn && platform_fn) {
    EGLint num = 0;
    query_fn(0, nullptr, &num);
    if (num > 0) {
      std::vector<void*> devs(static_cast<size_t>(num));
      query_fn(num, devs.data(), &num);
      for (EGLint i = 0; i < num; ++i) {
        EGLDisplay dpy = platform_fn(EGL_PLATFORM_DEVICE_EXT, devs[static_cast<size_t>(i)], nullptr);
        if (dpy == EGL_NO_DISPLAY) continue;
        EGLint maj = 0, min2 = 0;
        if (eglInitialize(dpy, &maj, &min2)) {
          ABSL_LOG(INFO) << "avp_headless_patch: EGL device " << i << " ok";
          return dpy;
        }
      }
    }
  }
  auto status_or_display = GetInitializedDefaultEglDisplay();
  return status_or_display;
}"""
if old_fn not in src:
    print("WARNING: GetInitializedEglDisplay not found, skipping fn patch", file=sys.stderr)
else:
    src = src.replace(old_fn, new_fn, 1)
    print("avp_headless_patch: GetInitializedEglDisplay patched")

open(path, "w").write(src)
print("avp_headless_patch: gl_context_egl.cc patched ok")
PYEOF

echo "Building //avp_bridge:libavp_mediapipe_face_mesh.so ..."
BAZEL_COPTS=""
[ -n "$OPENCV_INC" ] && BAZEL_COPTS="--copt=-I$OPENCV_INC"

BAZEL_BUILD_FLAGS=(
    --config=linux_gpu
    --compilation_mode=opt
    --copt=-DEGL_NO_X11
    --copt=-DMESA_EGL_NO_X11_HEADERS
    --copt=-DMEDIAPIPE_OMIT_EGL_WINDOW_BIT
    --nocheck_visibility
)
[ -n "$BAZEL_COPTS" ] && BAZEL_BUILD_FLAGS+=($BAZEL_COPTS)

# Build combined GPU bridge (face mesh + face detection in one .so)
bazel build "${BAZEL_BUILD_FLAGS[@]}" \
    //avp_bridge:libavp_mediapipe_gpu.so

# Also build the face_landmark + face_detection filegroups explicitly so their
# tflite files materialise in bazel-bin at mediapipe/modules/... paths.
# cc_binary(linkshared=1) does NOT generate runfiles, so data deps are not
# automatically placed in bazel-bin unless we explicitly build them.
echo "Building face landmark + detection model targets..."
bazel build "${BAZEL_BUILD_FLAGS[@]}" \
    //mediapipe/modules/face_landmark:face_landmark_model \
    //mediapipe/modules/face_landmark:face_landmark_with_attention_model \
    //mediapipe/modules/face_detection:face_detection_short_range_common \
    2>/dev/null \
|| bazel build "${BAZEL_BUILD_FLAGS[@]}" \
    //mediapipe/modules/face_landmark/... \
    //mediapipe/modules/face_detection/... \
    2>/dev/null \
|| echo "Note: explicit model target build failed — falling back to search-based install"

# Install — query bazel-bin for the same compilation_mode used above (opt)
BAZEL_BIN=$(bazel info --compilation_mode=opt bazel-bin)
mkdir -p "$MEDIAPIPE_INSTALL_DIR/lib"
cp -v "$BAZEL_BIN/avp_bridge/libavp_mediapipe_gpu.so" \
      "$MEDIAPIPE_INSTALL_DIR/lib/"

# Install model files.
# After explicitly building the model filegroup targets above, Bazel places the
# tflite files in bazel-bin at their workspace-relative paths:
#   bazel-bin/mediapipe/modules/face_landmark/face_landmark.tflite
#   bazel-bin/mediapipe/modules/face_detection/face_detection_short_range.tflite
# Fallback: search output_base/external (downloaded http_file repos) and
# bazel-mediapipe/external (source-tree symlinks).
mkdir -p "$MEDIAPIPE_INSTALL_DIR/share"
BAZEL_OUTPUT_BASE=$(bazel info --compilation_mode=opt output_base 2>/dev/null || true)
echo "Bazel output_base: $BAZEL_OUTPUT_BASE"
echo "Bazel bin:         $BAZEL_BIN"

install_tflite() {
    local src="$1"
    # First: try to extract mediapipe/modules/... from the path
    local rel
    rel=$(echo "$src" | grep -oP 'mediapipe/modules/[^/]+/[^/]+\.tflite')
    # Fallback: use the basename in the appropriate module dir (derived from dirname keywords)
    if [ -z "$rel" ]; then
        local name
        name=$(basename "$src")
        case "$src" in
            *face_landmark*)
                rel="mediapipe/modules/face_landmark/$name" ;;
            *face_detection*)
                rel="mediapipe/modules/face_detection/$name" ;;
            *)
                return ;;
        esac
    fi
    local dst="$MEDIAPIPE_INSTALL_DIR/share/$rel"
    [ -f "$dst" ] && return
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst" && echo "installed: $rel"
}

for search_root in \
    "$BAZEL_BIN/mediapipe/modules" \
    "$BAZEL_OUTPUT_BASE/external" \
    "$MEDIAPIPE_SRC_DIR/bazel-mediapipe/external" \
    "$MEDIAPIPE_SRC_DIR/bazel-out" \
    "$MEDIAPIPE_SRC_DIR" \
; do
    [ -d "$search_root" ] || continue
    while IFS= read -r f; do
        install_tflite "$f"
    done < <(find -L "$search_root" -name "*.tflite" \
        \( -path "*face_landmark*" -o -path "*face_detection*" \) 2>/dev/null)
done

COUNT=$(find "$MEDIAPIPE_INSTALL_DIR/share" -name "*.tflite" | wc -l)
echo "Installed tflite models: $COUNT"
if [ "$COUNT" -eq 0 ]; then
    echo "NOTE: tflite models not found via Bazel search." >&2
    echo "Searched bazel-bin:" >&2
    find "$BAZEL_BIN/mediapipe" -name "*.tflite" 2>/dev/null | head -30 >&2
    echo "Searched output_base:" >&2
    find "$BAZEL_OUTPUT_BASE" -name "*.tflite" 2>/dev/null | head -30 >&2
    echo "Downloading models directly from storage.googleapis.com ..."
    LM_DIR="$MEDIAPIPE_INSTALL_DIR/share/mediapipe/modules/face_landmark"
    DET_DIR="$MEDIAPIPE_INSTALL_DIR/share/mediapipe/modules/face_detection"
    mkdir -p "$LM_DIR" "$DET_DIR"
    BASE="https://storage.googleapis.com/mediapipe-assets"
    curl -fL -o "$LM_DIR/face_landmark.tflite"                    "$BASE/face_landmark.tflite"
    curl -fL -o "$LM_DIR/face_landmark_with_attention.tflite"     "$BASE/face_landmark_with_attention.tflite"
    curl -fL -o "$DET_DIR/face_detection_short_range.tflite"      "$BASE/face_detection_short_range.tflite"
    COUNT=$(find "$MEDIAPIPE_INSTALL_DIR/share" -name "*.tflite" | wc -l)
    echo "Models after direct download: $COUNT"
fi
if [ "$COUNT" -eq 0 ]; then
    echo "ERROR: still no tflite models after direct download." >&2
    exit 1
fi

echo "=== MediaPipe bridge build complete ==="
echo "  Library: $MEDIAPIPE_INSTALL_DIR/lib/libavp_mediapipe_gpu.so"

touch "$MEDIAPIPE_INSTALL_DIR/.avp-face-mesh-built"
