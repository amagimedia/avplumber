#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AVP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MEDIAPIPE_REPO_URL="${MEDIAPIPE_REPO_URL:-https://github.com/google-ai-edge/mediapipe.git}"
MEDIAPIPE_REV="${MEDIAPIPE_REV:-v0.10.35}"
MEDIAPIPE_BRIDGE_SOURCE_DIR="${MEDIAPIPE_BRIDGE_SOURCE_DIR:-${AVP_ROOT}/deps/mediapipe-bridge}"
MEDIAPIPE_SRC_DIR="${MEDIAPIPE_SRC_DIR:-${AVP_ROOT}/deps/mediapipe-src}"
MEDIAPIPE_INSTALL_DIR="${MEDIAPIPE_INSTALL_DIR:-${AVP_ROOT}/deps/mediapipe-face-mesh}"
HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}"
MEDIAPIPE_OPENCV_PKG_CONFIG="${MEDIAPIPE_OPENCV_PKG_CONFIG:-opencv4}"

if [[ -n "${BAZEL:-}" ]]; then
	BAZEL_BIN="${BAZEL}"
elif command -v bazelisk >/dev/null 2>&1; then
	BAZEL_BIN="bazelisk"
elif command -v bazel >/dev/null 2>&1; then
	BAZEL_BIN="bazel"
else
	cat >&2 <<'EOF'
MediaPipe requires Bazel/Bazelisk, but neither command was found.
Install Bazelisk or set BAZEL=/path/to/bazelisk, then rerun with HAVE_MEDIAPIPE=1.
EOF
	exit 1
fi

mkdir -p "$(dirname -- "${MEDIAPIPE_SRC_DIR}")" "${MEDIAPIPE_INSTALL_DIR}"

if [[ ! -d "${MEDIAPIPE_SRC_DIR}/.git" ]]; then
	git clone "${MEDIAPIPE_REPO_URL}" "${MEDIAPIPE_SRC_DIR}"
fi

git -C "${MEDIAPIPE_SRC_DIR}" fetch --tags origin
git -C "${MEDIAPIPE_SRC_DIR}" checkout --detach "${MEDIAPIPE_REV}"

BRIDGE_DST="${MEDIAPIPE_SRC_DIR}/mediapipe/avp_bridge"
mkdir -p "${BRIDGE_DST}"
cp "${MEDIAPIPE_BRIDGE_SOURCE_DIR}/BUILD.bazel" "${BRIDGE_DST}/BUILD.bazel"
cp "${MEDIAPIPE_BRIDGE_SOURCE_DIR}/avp_mediapipe_face_mesh_bridge.h" "${BRIDGE_DST}/"
cp "${MEDIAPIPE_BRIDGE_SOURCE_DIR}/avp_mediapipe_face_mesh_bridge.cc" "${BRIDGE_DST}/"

BAZEL_OPTS=(
	--repo_env=HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION}"
	--copt=-DMESA_EGL_NO_X11_HEADERS
	--copt=-DEGL_NO_X11
)

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "${MEDIAPIPE_OPENCV_PKG_CONFIG}"; then
	read -r -a OPENCV_CFLAGS <<< "$(pkg-config --cflags "${MEDIAPIPE_OPENCV_PKG_CONFIG}")"
	for flag in "${OPENCV_CFLAGS[@]}"; do
		BAZEL_OPTS+=(--copt="${flag}")
	done
fi

(
	cd "${MEDIAPIPE_SRC_DIR}"
	"${BAZEL_BIN}" build -c opt \
		"${BAZEL_OPTS[@]}" \
		//mediapipe/avp_bridge:libavp_mediapipe_face_mesh.so \
		//mediapipe/modules/face_detection:face_detection_short_range.tflite \
		//mediapipe/modules/face_landmark:face_landmark.tflite \
		//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite
)

install_mediapipe_file() {
	local relpath="$1"
	local src="${MEDIAPIPE_SRC_DIR}/${relpath}"
	if [[ ! -f "${src}" ]]; then
		src="${MEDIAPIPE_SRC_DIR}/bazel-bin/${relpath}"
	fi
	if [[ ! -f "${src}" ]]; then
		echo "MediaPipe build did not produce ${relpath}" >&2
		exit 1
	fi
	install -D -m 0644 "${src}" "${MEDIAPIPE_INSTALL_DIR}/share/${relpath}"
}

install -D -m 0755 \
	"${MEDIAPIPE_SRC_DIR}/bazel-bin/mediapipe/avp_bridge/libavp_mediapipe_face_mesh.so" \
	"${MEDIAPIPE_INSTALL_DIR}/lib/libavp_mediapipe_face_mesh.so"
install -D -m 0644 \
	"${MEDIAPIPE_BRIDGE_SOURCE_DIR}/avp_mediapipe_face_mesh_bridge.h" \
	"${MEDIAPIPE_INSTALL_DIR}/include/avp_mediapipe_face_mesh_bridge.h"

install_mediapipe_file "mediapipe/modules/face_detection/face_detection_short_range.tflite"
install_mediapipe_file "mediapipe/modules/face_landmark/face_landmark.tflite"
install_mediapipe_file "mediapipe/modules/face_landmark/face_landmark_with_attention.tflite"

{
	echo "repo=${MEDIAPIPE_REPO_URL}"
	echo "rev=${MEDIAPIPE_REV}"
	echo "src=${MEDIAPIPE_SRC_DIR}"
	echo "install=${MEDIAPIPE_INSTALL_DIR}"
	date -u '+built_utc=%Y-%m-%dT%H:%M:%SZ'
} > "${MEDIAPIPE_INSTALL_DIR}/.avp-face-mesh-built"
