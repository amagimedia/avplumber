#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

SUPPORT_SERVICES=(janus janus-preview web-ui wayland dma-browser)
HEALTH_SERVICES=(web-ui wayland dma-browser)
REQUIRED_COMPOSE_ENV=(
  TENSORRT_CONTEXT
  TENSORRT_ARCHIVE
  FACE_ONNX
  FACE_ENGINE_CACHE_DIR
  MEDIA_INPUT_DIR
  MEDIA_WIPE_DIR
)
PRESERVE_ENV=(
  AUTO_MIXER_IMAGE_TAG
  AUTO_MIXER_ARGS
  AUTO_MIXER_CONTAINER
  AUTO_MIXER_INSTANCE_NAME
  AUTO_MIXER_JANUS_HOST
  AUTO_MIXER_LOGFILE
  AUTO_MIXER_SERVICE
  COMPOSE_FILE
  COMPOSE_PROJECT_NAME
  DMA_BROWSER_ALLOWED_DIMS
  DMA_BROWSER_CHROMIUM_EXTRA_FEATURES
  DMA_BROWSER_DMABUF_POOL_SIZE
  DMA_BROWSER_IMAGE_TAG
  DMA_BROWSER_REST_PORT
  ENV_FILE
  FACE_ENGINE_CACHE_DIR
  FACE_ONNX
  HTML_OVERLAY_DRM_DEVICE
  HTML_OVERLAY_FPS
  HTML_OVERLAY_HEIGHT
  HTML_OVERLAY_URL
  HTML_OVERLAY_WIDTH
  JANUS_DEBUG_LEVEL
  JANUS_HOST_IP
  JANUS_HTTP_PORT
  JANUS_AUDIO_ENABLED
  JANUS_AUDIO_PORT
  JANUS_AUDIO_RTCP_PORT
  JANUS_IMAGE_TAG
  JANUS_PREVIEW_IMAGE_TAG
  JANUS_PREVIEW_PORT
  JANUS_REF
  JANUS_RTP_PORT_RANGE
  JANUS_VIDEO_PORT
  JANUS_VIDEO_RTCP_PORT
  JANUS_VIDEO_PT
  JANUS_VIDEO_SSRC
  MEDIA_INPUT_DIR
  MEDIA_WIPE_DIR
  NEURAL_DEMO_ARTIFACT_DIR
  NEURAL_DEMO_EXAMPLE
  NEURAL_DEMO_IMAGE
  NEURAL_DEMO_INPUT
  NEURAL_DEMO_INSTANCE_NAME
  NEURAL_DEMO_LOGFILE
  NEURAL_DEMO_METADATA_DUMPS
  NEURAL_DEMO_MODE
  NEURAL_DEMO_OUTPUT
  REMOTE_CONTROL_PORT
  TENSORRT_ARCHIVE
  TENSORRT_CONTEXT
  WAYLAND_DISPLAY
  WAYLAND_IMAGE_TAG
  WEBUI_IMAGE_TAG
  WEBUI_PORT
  WLR_RENDER_DRM_DEVICE
)

DOCKER_CMD=()

usage() {
  cat <<'EOF'
Usage:
  mixer-stack.sh start [--build] [--force-recreate] [--name NAME] [--] <auto-mixer args...>
  mixer-stack.sh stop [--down]
  mixer-stack.sh restart [--build] [--force-recreate] [--name NAME] [--] <auto-mixer args...>
  mixer-stack.sh status
  mixer-stack.sh logs [SERVICE|auto-mixer]
  mixer-stack.sh help

Environment:
  Copy .env.example to .env and fill host-local paths before starting.
  AUTO_MIXER_ARGS may hold the auto-mixer CLI when no args are passed after --.
  AUTO_MIXER_CONTAINER defaults to avp-auto-mixer.
  AUTO_MIXER_STOP_TIMEOUT defaults to 20 seconds.
  HEALTH_TIMEOUT defaults to 120 seconds.

Examples:
  ./mixer-stack.sh start -- \
    --inputs /media-inputs/cam0.ts /media-inputs/cam1.ts \
    --janus-output --janus-host 127.0.0.1 \
    --media-wipe-dir /media-wipes \
    --html-overlay-url "$HTML_OVERLAY_URL"

  ./mixer-stack.sh stop
  ./mixer-stack.sh logs auto-mixer
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

load_env_file() {
  [[ -f "$ENV_FILE" ]] || return 0

  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*($|#) ]] && continue
    line="${line#export }"

    if [[ "$line" != *=* ]]; then
      warn "ignoring malformed env line in $ENV_FILE: $line"
      continue
    fi

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"

    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      warn "ignoring invalid env key in $ENV_FILE: $key"
      continue
    fi

    if [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]] || [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    fi

    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done <"$ENV_FILE"
}

select_docker_cmd() {
  if [[ -n "${DOCKER_CMD_OVERRIDE:-}" ]]; then
    read -r -a DOCKER_CMD <<<"$DOCKER_CMD_OVERRIDE"
    return 0
  fi

  if docker info >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return 0
  fi

  if sudo -n docker info >/dev/null 2>&1; then
    local preserve
    preserve="$(IFS=,; printf '%s' "${PRESERVE_ENV[*]}")"
    DOCKER_CMD=(sudo "--preserve-env=$preserve" docker)
    return 0
  fi

  die "docker is not available; set DOCKER_CMD_OVERRIDE or enable passwordless docker access"
}

docker_cmd() {
  "${DOCKER_CMD[@]}" "$@"
}

compose() {
  docker_cmd compose --project-directory "$SCRIPT_DIR" -f "$COMPOSE_FILE" "$@"
}

require_compose_env() {
  local missing=()
  local name
  for name in "${REQUIRED_COMPOSE_ENV[@]}"; do
    if [[ -z "${!name:-}" || "${!name}" == \<* ]]; then
      missing+=("$name")
    fi
  done

  if ((${#missing[@]})); then
    printf 'error: missing required compose environment:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    printf 'Copy %s to %s and fill host-local paths, or export these variables.\n' \
      "$SCRIPT_DIR/.env.example" "$ENV_FILE" >&2
    exit 1
  fi
}

container_exists() {
  docker_cmd inspect "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker_cmd inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]
}

stop_auto_mixer_container() {
  if ! container_exists "$AUTO_MIXER_CONTAINER"; then
    return 0
  fi

  if container_running "$AUTO_MIXER_CONTAINER"; then
    docker_cmd stop -t "$AUTO_MIXER_STOP_TIMEOUT" "$AUTO_MIXER_CONTAINER" >/dev/null
  fi

  docker_cmd rm "$AUTO_MIXER_CONTAINER" >/dev/null
}

health_status() {
  local service="$1"
  local container_id
  container_id="$(compose ps -q "$service")"
  [[ -n "$container_id" ]] || return 1
  docker_cmd inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id"
}

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  local service status all_ready

  while ((SECONDS < deadline)); do
    all_ready=1
    for service in "${HEALTH_SERVICES[@]}"; do
      status="$(health_status "$service" 2>/dev/null || true)"
      if [[ "$status" != healthy && "$status" != running ]]; then
        all_ready=0
        printf '%s health=%s\n' "$service" "${status:-missing}"
      fi
    done

    if ((all_ready)); then
      return 0
    fi

    sleep "$HEALTH_INTERVAL"
  done

  printf 'Health check timed out after %s seconds.\n' "$HEALTH_TIMEOUT" >&2
  compose ps "${SUPPORT_SERVICES[@]}" >&2 || true
  return 1
}

auto_mixer_args_from_env() {
  local -n out="$1"
  out=()

  [[ -n "${AUTO_MIXER_ARGS:-}" ]] || return 0
  command -v python3 >/dev/null 2>&1 || die "python3 is required to parse AUTO_MIXER_ARGS"

  mapfile -d '' -t out < <(
    AUTO_MIXER_ARGS="$AUTO_MIXER_ARGS" python3 - <<'PY'
import os
import shlex
import sys

for arg in shlex.split(os.environ["AUTO_MIXER_ARGS"]):
    sys.stdout.write(arg)
    sys.stdout.write("\0")
PY
  )
}

start_support_services() {
  local -a up_flags=("-d")
  if [[ "$START_BUILD" == 1 ]]; then
    up_flags+=("--build")
  fi
  if [[ "$START_FORCE_RECREATE" == 1 ]]; then
    up_flags+=("--force-recreate")
  fi

  compose up "${up_flags[@]}" "${SUPPORT_SERVICES[@]}"
  wait_for_health
}

start_auto_mixer() {
  local -a args=("$@")
  local -a run_flags=("--no-deps" "-d" "--name" "$AUTO_MIXER_CONTAINER")

  if ((${#args[@]} == 0)); then
    auto_mixer_args_from_env args
  fi

  if ((${#args[@]} == 0)); then
    die "auto-mixer args are required; pass them after -- or set AUTO_MIXER_ARGS"
  fi

  if container_exists "$AUTO_MIXER_CONTAINER"; then
    if container_running "$AUTO_MIXER_CONTAINER" && [[ "$START_FORCE_RECREATE" != 1 ]]; then
      printf '%s is already running. Use restart or start --force-recreate to replace it.\n' "$AUTO_MIXER_CONTAINER"
      return 0
    fi
    stop_auto_mixer_container
  fi

  if [[ "$START_BUILD" == 1 ]]; then
    run_flags=("--build" "${run_flags[@]}")
  fi

  compose --profile run run "${run_flags[@]}" "$AUTO_MIXER_SERVICE" "${args[@]}"
  docker_cmd inspect "$AUTO_MIXER_CONTAINER" --format 'auto-mixer={{.Name}} status={{.State.Status}} image={{.Config.Image}}'
}

start_stack() {
  require_compose_env
  start_support_services
  start_auto_mixer "$@"
}

stop_stack() {
  local down=0
  while (($#)); do
    case "$1" in
      --down)
        down=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown stop option: $1"
        ;;
    esac
  done

  stop_auto_mixer_container

  if ((down)); then
    require_compose_env
    compose down
  else
    require_compose_env
    compose stop "${SUPPORT_SERVICES[@]}"
  fi
}

status_stack() {
  require_compose_env
  compose ps "${SUPPORT_SERVICES[@]}"

  if container_exists "$AUTO_MIXER_CONTAINER"; then
    docker_cmd inspect "$AUTO_MIXER_CONTAINER" \
      --format 'auto-mixer={{.Name}} status={{.State.Status}} image={{.Config.Image}} started={{.State.StartedAt}}'
  else
    printf 'auto-mixer=%s status=missing\n' "$AUTO_MIXER_CONTAINER"
  fi
}

logs_stack() {
  local target="${1:-}"
  if [[ "$target" == auto-mixer || "$target" == "$AUTO_MIXER_CONTAINER" ]]; then
    container_exists "$AUTO_MIXER_CONTAINER" || die "$AUTO_MIXER_CONTAINER does not exist"
    docker_cmd logs -f --tail "${LOG_TAIL:-200}" "$AUTO_MIXER_CONTAINER"
  elif [[ -n "$target" ]]; then
    require_compose_env
    compose logs -f --tail "${LOG_TAIL:-200}" "$target"
  else
    require_compose_env
    compose logs -f --tail "${LOG_TAIL:-200}" "${SUPPORT_SERVICES[@]}"
  fi
}

parse_start_options() {
  START_BUILD=0
  START_FORCE_RECREATE=0
  START_ARGS=()

  while (($#)); do
    case "$1" in
      --build)
        START_BUILD=1
        shift
        ;;
      --force-recreate)
        START_FORCE_RECREATE=1
        shift
        ;;
      --name)
        [[ $# -ge 2 ]] || die "--name requires a value"
        AUTO_MIXER_CONTAINER="$2"
        shift 2
        ;;
      --)
        shift
        START_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown start option before --: $1"
        ;;
    esac
  done
}

main() {
  local command="${1:-help}"
  [[ $# -gt 0 ]] && shift || true

  load_env_file

  : "${AUTO_MIXER_CONTAINER:=avp-auto-mixer}"
  : "${AUTO_MIXER_SERVICE:=auto-mixer}"
  : "${AUTO_MIXER_STOP_TIMEOUT:=20}"
  : "${HEALTH_TIMEOUT:=120}"
  : "${HEALTH_INTERVAL:=2}"

  case "$command" in
    help|-h|--help)
      usage
      exit 0
      ;;
  esac

  select_docker_cmd

  case "$command" in
    start)
      parse_start_options "$@"
      start_stack "${START_ARGS[@]}"
      ;;
    stop)
      stop_stack "$@"
      ;;
    restart)
      parse_start_options "$@"
      START_FORCE_RECREATE=1
      stop_stack
      start_stack "${START_ARGS[@]}"
      ;;
    status)
      status_stack
      ;;
    logs)
      logs_stack "$@"
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
