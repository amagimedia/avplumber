#!/usr/bin/env bash
# Launcher for dma-browser.
#
# The default NVIDIA path expects an Electron build with the demo's
# runtime-gated native-handle patch. No GBM LD_PRELOAD shim is applied.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${DMA_BROWSER_WORKER_INDEX:-}" && -z "${DMA_BROWSER_PROCESS_COUNT:-}" ]]; then
  grouping_total_raw="${DMA_BROWSER_MAX_WINDOWS:-8}"
  grouping_size_raw="${DMA_BROWSER_WINDOWS_PER_PROCESS:-8}"
  if [[ ! "$grouping_total_raw" =~ ^[0-9]+$ || ! "$grouping_size_raw" =~ ^[0-9]+$ ]] ||
    (( 10#$grouping_total_raw < 1 || 10#$grouping_size_raw < 1 )); then
    echo "dma-browser: window grouping values must be positive integers" >&2
    exit 1
  fi
  grouping_total=$((10#$grouping_total_raw))
  grouping_size=$((10#$grouping_size_raw))
  export DMA_BROWSER_PROCESS_COUNT=$(((grouping_total + grouping_size - 1) / grouping_size))
fi

if [[ -z "${DMA_BROWSER_WORKER_INDEX:-}" && "${DMA_BROWSER_PROCESS_COUNT:-1}" != "1" ]]; then
  node_bin="${DMA_BROWSER_NODE_BIN:-node}"
  if ! command -v "$node_bin" >/dev/null 2>&1; then
    echo "dma-browser: Node.js is required for multiprocess supervision" >&2
    exit 1
  fi
  exec "$node_bin" "$PROJECT_ROOT/dist/main/supervisor/index.js"
fi

electron_args=(--no-sandbox)

if [[ -n "${DMA_BROWSER_USER_DATA_DIR:-}" ]]; then
  mkdir -p -- "$DMA_BROWSER_USER_DATA_DIR"
  electron_args+=("--user-data-dir=$DMA_BROWSER_USER_DATA_DIR")
fi

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
  local scanout_feature="RenderableMappableSharedImageForceScanout"

  export DMA_BROWSER_CHROMIUM_EXTRA_FEATURES="${DMA_BROWSER_CHROMIUM_EXTRA_FEATURES:+${DMA_BROWSER_CHROMIUM_EXTRA_FEATURES},}${scanout_feature}"

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
