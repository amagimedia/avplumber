#!/bin/sh
# Launcher for dma-browser.
#
# Official Electron; on NVIDIA always with the bundled GBM shim preloaded.
set -e

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

positive_decimal() {
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  # Normalize leading zeroes before shell arithmetic. expr returns failure for zero.
  expr "$1" + 0
}

if [ -z "${DMA_BROWSER_WORKER_INDEX:-}" ] && [ -z "${DMA_BROWSER_PROCESS_COUNT:-}" ]; then
  if ! grouping_total=$(positive_decimal "${DMA_BROWSER_MAX_WINDOWS:-8}") ||
     ! grouping_size=$(positive_decimal "${DMA_BROWSER_WINDOWS_PER_PROCESS:-8}"); then
    echo "dma-browser: window grouping values must be positive integers" >&2
    exit 1
  fi
  export DMA_BROWSER_PROCESS_COUNT=$(((grouping_total + grouping_size - 1) / grouping_size))
fi

if [ -n "${DMA_BROWSER_ISOLATED_SCENE_URL:-}" ]; then
  if [ "${DMA_BROWSER_PROCESS_COUNT:-1}" != 1 ]; then
    echo "dma-browser: isolated scenes require a single browser process" >&2
    exit 1
  fi
  if [ "$#" -ne 0 ]; then
    echo "dma-browser: isolated scenes do not accept extra launch arguments" >&2
    exit 1
  fi
fi

if [ -z "${DMA_BROWSER_WORKER_INDEX:-}" ] && [ "${DMA_BROWSER_PROCESS_COUNT:-1}" != 1 ]; then
  node_bin="${DMA_BROWSER_NODE_BIN:-node}"
  if ! command -v "$node_bin" >/dev/null 2>&1; then
    echo "dma-browser: Node.js is required for multiprocess supervision" >&2
    exit 1
  fi
  exec "$node_bin" "$PROJECT_ROOT/dist/main/supervisor/index.js"
fi

detect_nvidia() {
  if [ "${DMA_BROWSER_FORCE_NVIDIA:-0}" = 1 ]; then
    return 0
  fi
  if [ "${DMA_BROWSER_DISABLE_NVIDIA_RUNTIME:-0}" = 1 ]; then
    return 1
  fi
  if pci_devices=$(lspci 2>/dev/null) &&
     printf '%s\n' "$pci_devices" | grep -Eqi '(vga|3d|display).*nvidia'; then
    return 0
  fi
  for f in /sys/class/drm/card*/device/vendor; do
    [ -r "$f" ] || continue
    if [ "$(cat "$f")" = "0x10de" ]; then
      return 0
    fi
  done
  return 1
}

detect_wayland_display() {
  runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    printf '%s\n' "$WAYLAND_DISPLAY"
    return 0
  fi
  for socket in "$runtime_dir"/wayland-*; do
    [ -S "$socket" ] || continue
    basename "$socket"
    return 0
  done
  return 1
}

detect_render_node() {
  if [ -n "${DMA_BROWSER_RENDER_NODE:-}" ]; then
    printf '%s\n' "$DMA_BROWSER_RENDER_NODE"
    return 0
  fi
  for node in /dev/dri/renderD*; do
    [ -e "$node" ] || continue
    printf '%s\n' "$node"
    return 0
  done
  return 1
}

# NVIDIA's GBM allocates Chromium's capture buffers renderable only with the shim's usage flags.
apply_gbm_shim() {
  shim="$PROJECT_ROOT/native/gbm-linear-shim/libgbm_linear_shim.so"
  if [ ! -f "$shim" ]; then
    echo "dma-browser: GBM shim not built (run 'npm run rebuild:shim')" >&2
    exit 1
  fi
  case ":${LD_PRELOAD:-}:" in
    *":$shim:"*) ;;
    *) export LD_PRELOAD="$shim${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
}

# Older Bash releases treat an empty "$@" as unset under nounset.
set -- "$PROJECT_ROOT/dist/main/index.js" "$@"
set -u

if detect_nvidia; then
  apply_gbm_shim
  if render_node=$(detect_render_node); then
    set -- "--hardware-video-device-path=$render_node" "--render-node-override=$render_node" "$@"
  fi

  gl_backend="${DMA_BROWSER_GL_BACKEND:-angle}"
  case "$gl_backend" in
    angle|egl-angle) set -- "--use-angle=${DMA_BROWSER_ANGLE_BACKEND:-gl-egl}" "$@" ;;
  esac
  set -- "--use-gl=$gl_backend" "$@"

  if wayland_display=$(detect_wayland_display); then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    export WAYLAND_DISPLAY="$wayland_display"
    export XDG_SESSION_TYPE="wayland"
    export EGL_PLATFORM="${EGL_PLATFORM:-wayland}"
    if [ -S "$XDG_RUNTIME_DIR/bus" ]; then
      export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
    fi
    set -- --ozone-platform=wayland "$@"
  elif [ -n "${DISPLAY:-}" ]; then
    set -- --ozone-platform=x11 "$@"
  fi
  echo "dma-browser: NVIDIA runtime enabled, GL backend=$gl_backend" >&2
fi

if [ -n "${DMA_BROWSER_USER_DATA_DIR:-}" ]; then
  mkdir -p -- "$DMA_BROWSER_USER_DATA_DIR"
  set -- "--user-data-dir=$DMA_BROWSER_USER_DATA_DIR" "$@"
fi
if [ -z "${DMA_BROWSER_ISOLATED_SCENE_URL:-}" ]; then
  set -- --no-sandbox "$@"
fi

ELECTRON_BIN="$PROJECT_ROOT/node_modules/.bin/electron"
if [ ! -x "$ELECTRON_BIN" ]; then
  echo "dma-browser: electron not installed (run 'npm install' first)" >&2
  exit 1
fi

exec "$ELECTRON_BIN" "$@"
