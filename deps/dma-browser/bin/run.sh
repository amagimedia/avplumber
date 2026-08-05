#!/usr/bin/env bash
# Launcher for dma-browser.
#
# The default NVIDIA path expects the patched Electron 41 / Chrome 146 build
# from the project .npmrc. No GBM LD_PRELOAD shim is applied.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

electron_args=(--no-sandbox)

detect_nvidia() {
  if [[ "${DMA_BROWSER_FORCE_NVIDIA:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${DMA_BROWSER_DISABLE_NVIDIA_RUNTIME:-0}" == "1" ]]; then
    return 1
  fi
  if command -v lspci >/dev/null 2>&1; then
    if lspci 2>/dev/null | grep -Eqi '(vga|3d|display).*nvidia'; then
      return 0
    fi
  fi
  if compgen -G "/sys/class/drm/card*/device/vendor" >/dev/null; then
    for f in /sys/class/drm/card*/device/vendor; do
      [[ -r "$f" ]] || continue
      if [[ "$(cat "$f")" == "0x10de" ]]; then
        return 0
      fi
    done
  fi
  return 1
}

detect_wayland_display() {
  local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  local socket
  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    printf '%s\n' "$WAYLAND_DISPLAY"
    return 0
  fi
  if compgen -G "$runtime_dir/wayland-*" >/dev/null; then
    for socket in "$runtime_dir"/wayland-*; do
      [[ -S "$socket" ]] || continue
      printf '%s\n' "$(basename "$socket")"
      return 0
    done
  fi
  return 1
}

detect_render_node() {
  local node
  if [[ -n "${DMA_BROWSER_RENDER_NODE:-}" ]]; then
    printf '%s\n' "$DMA_BROWSER_RENDER_NODE"
    return 0
  fi
  if compgen -G "/dev/dri/renderD*" >/dev/null; then
    for node in /dev/dri/renderD*; do
      [[ -e "$node" ]] || continue
      printf '%s\n' "$node"
      return 0
    done
  fi
  return 1
}

apply_nvidia_runtime() {
  local wayland_display
  local render_node
  local gl_backend="${DMA_BROWSER_GL_BACKEND:-angle}"
  local angle_backend="${DMA_BROWSER_ANGLE_BACKEND:-gl-egl}"

  # GBM shim for the stock-Electron shared-texture capture path on NVIDIA, so no
  # patched Chromium is needed (replaces the custom feature
  # RenderableMappableSharedImageForceScanout). Only preloaded on the NVIDIA path
  # (rewriting GBM flags can harm Intel/AMD). Two mutually exclusive modes:
  #
  #   FORCE_LINEAR (default): allocate LINEAR *and* RENDERING and pin the modifier
  #     list to [LINEAR]. The buffer is a valid render target (SkSurface works ->
  #     frames flow) AND CPU-linear, so a downstream consumer using a plain DRM
  #     hwdownload (no CUDA/EGL detiling) reads a correct image. This is what the
  #     minimal dmabuf demo needs.
  #   ADD_SCANOUT (fallback, set FORCE_LINEAR=0 to use): strip LINEAR and add
  #     SCANOUT|RENDERING. Also captures on stock Chromium, but the buffer is
  #     tiled/block-linear -> only a CUDA/EGL import can detile it; a plain DRM
  #     hwdownload yields a sheared image.
  local shim="$PROJECT_ROOT/native/gbm-linear-shim/libgbm_linear_shim.so"
  if [[ ! -f "$shim" && -x "$PROJECT_ROOT/native/gbm-linear-shim/build.sh" ]] && command -v cc >/dev/null 2>&1; then
    shim="$("$PROJECT_ROOT/native/gbm-linear-shim/build.sh" 2>/dev/null | tail -1)"
  fi
  if [[ -f "$shim" ]]; then
    export LD_PRELOAD="$shim${LD_PRELOAD:+:$LD_PRELOAD}"
    export GBM_LINEAR_SHIM="${GBM_LINEAR_SHIM:-1}"
    export GBM_LINEAR_SHIM_FORCE_LINEAR="${GBM_LINEAR_SHIM_FORCE_LINEAR:-1}"
    export GBM_LINEAR_SHIM_ADD_SCANOUT="${GBM_LINEAR_SHIM_ADD_SCANOUT:-1}"
    export GBM_LINEAR_SHIM_LOG="${GBM_LINEAR_SHIM_LOG:-0}"
    echo "dma-browser: GBM linear shim preloaded ($shim)" >&2
  else
    echo "dma-browser: WARNING GBM linear shim missing; NVIDIA dmabuf capture may fail" >&2
  fi

  export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-nvidia}"
  export NVD_BACKEND="${NVD_BACKEND:-direct}"
  export NVD_LOG="${NVD_LOG:-0}"

  if [[ -d /opt/libva-2.17/lib/x86_64-linux-gnu ]]; then
    export LD_LIBRARY_PATH="/opt/libva-2.17/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -d /usr/lib64/dri ]]; then
    export LIBVA_DRIVERS_PATH="${LIBVA_DRIVERS_PATH:-/usr/lib64/dri}"
  elif [[ -d /usr/lib/x86_64-linux-gnu/dri ]]; then
    export LIBVA_DRIVERS_PATH="${LIBVA_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri}"
  fi

  if wayland_display="$(detect_wayland_display)"; then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    export WAYLAND_DISPLAY="$wayland_display"
    export XDG_SESSION_TYPE="wayland"
    export EGL_PLATFORM="${EGL_PLATFORM:-wayland}"
    if [[ -S "$XDG_RUNTIME_DIR/bus" ]]; then
      export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
    fi
    electron_args+=(--ozone-platform=wayland)
  elif [[ -n "${DISPLAY:-}" ]]; then
    electron_args+=(--ozone-platform=x11)
  fi

  electron_args+=(--disable-vulkan "--use-gl=$gl_backend")
  if [[ "$gl_backend" == "angle" || "$gl_backend" == "egl-angle" ]]; then
    electron_args+=("--use-angle=$angle_backend")
  fi

  if render_node="$(detect_render_node)"; then
    electron_args+=(
      "--hardware-video-device-path=$render_node"
      "--render-node-override=$render_node"
    )
  fi

  echo "dma-browser: NVIDIA runtime enabled, electron_args=${electron_args[*]}" >&2
}

if detect_nvidia; then
  apply_nvidia_runtime
fi

ELECTRON_BIN="${ELECTRON_BIN:-$PROJECT_ROOT/node_modules/.bin/electron}"
if [[ ! -x "$ELECTRON_BIN" ]]; then
  echo "dma-browser: electron not installed (run 'npm install' first)" >&2
  exit 1
fi

exec "$ELECTRON_BIN" "${electron_args[@]}" "$PROJECT_ROOT/dist/main/index.js" "$@"
