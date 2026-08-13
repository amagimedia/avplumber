#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PATCH_FILE="${SCRIPT_DIR}/electron-offscreen-native-handle.patch"

usage() {
  cat <<'EOF'
Build Electron with the avplumber NVIDIA DMA-BUF native-handle opt-in.

Usage:
  build-electron.sh [ELECTRON_VERSION]

The default is Electron 41.3.0 (Chromium 146), matching deps/dma-browser.
Electron 43.x builds Chromium 150; for example:

  build-electron.sh 43.4.0

Environment overrides:
  ELECTRON_BUILD_ROOT     Checkout and build directory.
  ELECTRON_ARTIFACT_DIR  Destination for the dist zip and checksum.
  ELECTRON_DEPOT_TOOLS   Existing depot_tools checkout to reuse.
  ELECTRON_OUT_DIR       Chromium output directory (default: out/Release).
  BUILD_JOBS             Parallel build jobs passed to autoninja.

The patch changes Electron's offscreen shared-texture consumer and leaves the
embedded Chromium source untouched. It is disabled by default at runtime.

The script never resets an existing checkout. It refuses a mismatched or dirty
checkout so that local Chromium/Electron work is not overwritten.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

ELECTRON_VERSION="${1:-${ELECTRON_VERSION:-41.3.0}}"
if [[ ! "${ELECTRON_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid Electron version: ${ELECTRON_VERSION}" >&2
  exit 2
fi

ELECTRON_TAG="v${ELECTRON_VERSION}"
BUILD_ROOT="${ELECTRON_BUILD_ROOT:-${SCRIPT_DIR}/work/electron-${ELECTRON_VERSION}}"
ARTIFACT_DIR="${ELECTRON_ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
DEPOT_TOOLS="${ELECTRON_DEPOT_TOOLS:-${SCRIPT_DIR}/work/depot_tools}"
OUT_DIR="${ELECTRON_OUT_DIR:-out/Release}"
CHROMIUM_SRC="${BUILD_ROOT}/src"
ELECTRON_SRC="${CHROMIUM_SRC}/electron"

if [[ "${OUT_DIR}" == /* || "${OUT_DIR}" == *..* ]]; then
  echo "ELECTRON_OUT_DIR must be a relative path without '..': ${OUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Missing Electron patch: ${PATCH_FILE}" >&2
  exit 2
fi

if [[ ! -d "${DEPOT_TOOLS}/.git" ]]; then
  mkdir -p "$(dirname -- "${DEPOT_TOOLS}")"
  git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git \
    "${DEPOT_TOOLS}"
fi

export PATH="${DEPOT_TOOLS}:${PATH}"
export DEPOT_TOOLS_UPDATE="${DEPOT_TOOLS_UPDATE:-0}"
export CHROMIUM_BUILDTOOLS_PATH="${CHROMIUM_SRC}/buildtools"

checkout_ready=0
if [[ -d "${ELECTRON_SRC}/.git" && -f "${CHROMIUM_SRC}/chrome/VERSION" ]]; then
  current_tag="$(git -C "${ELECTRON_SRC}" describe --tags --exact-match 2>/dev/null || true)"
  if [[ "${current_tag}" != "${ELECTRON_TAG}" ]]; then
    echo "Existing checkout is ${current_tag:-not at a tag}, expected ${ELECTRON_TAG}." >&2
    echo "Choose a different ELECTRON_BUILD_ROOT; this script will not retarget it." >&2
    exit 3
  fi
  checkout_ready=1
fi

if [[ ${checkout_ready} -eq 0 ]]; then
  mkdir -p "${BUILD_ROOT}"
  if [[ ! -f "${BUILD_ROOT}/.gclient" ]]; then
    (
      cd "${BUILD_ROOT}"
      gclient config --name "src/electron" --unmanaged \
        https://github.com/electron/electron.git
    )
  fi

  (
    cd "${BUILD_ROOT}"
    gclient sync -f --with_branch_heads --with_tags
    git -C "${ELECTRON_SRC}" fetch origin tag "${ELECTRON_TAG}"
    git -C "${ELECTRON_SRC}" checkout --detach "${ELECTRON_TAG}"
    gclient sync -f --with_branch_heads --with_tags
  )
fi

current_tag="$(git -C "${ELECTRON_SRC}" describe --tags --exact-match 2>/dev/null || true)"
if [[ "${current_tag}" != "${ELECTRON_TAG}" ]]; then
  echo "Electron checkout verification failed: ${current_tag:-no tag}" >&2
  exit 3
fi

if git -C "${ELECTRON_SRC}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  unexpected_changes="$(
    git -C "${ELECTRON_SRC}" status --porcelain | \
      awk 'substr($0, 4) != "shell/browser/osr/osr_video_consumer.cc"'
  )"
  if [[ -n "${unexpected_changes}" ]]; then
    echo "Electron checkout has changes besides the DMA-BUF patch; refusing to build." >&2
    printf '%s\n' "${unexpected_changes}" >&2
    exit 4
  fi
  echo "Electron native-handle patch is already applied."
else
  if [[ -n "$(git -C "${ELECTRON_SRC}" status --porcelain)" ]]; then
    echo "Electron checkout is dirty; refusing to apply the patch." >&2
    git -C "${ELECTRON_SRC}" status --short >&2
    exit 4
  fi
  git -C "${ELECTRON_SRC}" apply --check "${PATCH_FILE}"
  git -C "${ELECTRON_SRC}" apply "${PATCH_FILE}"
  echo "Applied $(basename -- "${PATCH_FILE}")."
fi

if [[ -n "$(git -C "${CHROMIUM_SRC}" status --porcelain --untracked-files=no)" ]]; then
  echo "Chromium checkout is dirty; refusing to build unrelated changes." >&2
  git -C "${CHROMIUM_SRC}" status --short >&2
  exit 4
fi

chromium_version="$(awk -F= '
  /^MAJOR=/{major=$2}
  /^MINOR=/{minor=$2}
  /^BUILD=/{build=$2}
  /^PATCH=/{patch=$2}
  END{printf "%s.%s.%s.%s", major, minor, build, patch}
' "${CHROMIUM_SRC}/chrome/VERSION")"

(
  cd "${CHROMIUM_SRC}"
  gn gen "${OUT_DIR}" --args='import("//electron/build/args/release.gn")'
  if [[ -n "${BUILD_JOBS:-}" ]]; then
    autoninja -C "${OUT_DIR}" -j "${BUILD_JOBS}" electron:electron_dist_zip
  else
    autoninja -C "${OUT_DIR}" electron:electron_dist_zip
  fi
)

case "$(uname -m)" in
  x86_64) electron_arch=x64 ;;
  aarch64|arm64) electron_arch=arm64 ;;
  *)
    echo "Unsupported host architecture for artifact naming: $(uname -m)" >&2
    exit 5
    ;;
esac

mkdir -p "${ARTIFACT_DIR}"
artifact="${ARTIFACT_DIR}/electron-v${ELECTRON_VERSION}-linux-${electron_arch}.zip"
cp "${CHROMIUM_SRC}/${OUT_DIR}/dist.zip" "${artifact}"
(
  cd "${ARTIFACT_DIR}"
  sha256sum -b "$(basename -- "${artifact}")" > SHASUMS256.txt
)

electron_binary="${CHROMIUM_SRC}/${OUT_DIR}/electron"
echo
echo "Built Electron ${ELECTRON_VERSION} / Chromium ${chromium_version}."
echo "Binary:   ${electron_binary}"
echo "Artifact: ${artifact}"
echo
echo "Run the demo browser with:"
printf '  ELECTRON_BIN=%q %q/deps/dma-browser/bin/run.sh\n' \
  "${electron_binary}" "${REPO_ROOT}"
echo
echo "Keep deps/dma-browser's Electron version and native addon build aligned with ${ELECTRON_VERSION}."
