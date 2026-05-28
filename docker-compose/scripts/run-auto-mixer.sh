#!/usr/bin/env bash
set -euo pipefail

parse_auto_mixer_args() {
  python3 - <<'PY'
import os
import shlex
import sys

for arg in shlex.split(os.environ.get("AUTO_MIXER_ARGS", "")):
    sys.stdout.buffer.write(arg.encode())
    sys.stdout.buffer.write(b"\0")
PY
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

args=()

if [[ -n "${AUTO_MIXER_ARGS:-}" ]]; then
  while IFS= read -r -d '' arg; do
    args+=("$arg")
  done < <(parse_auto_mixer_args)
else
  shopt -s nullglob
  inputs=()
  for pattern in ${AUTO_MIXER_INPUTS:-/media-inputs/*}; do
    matches=( $pattern )
    inputs+=("${matches[@]}")
  done

  if [[ "${#inputs[@]}" -lt 2 ]]; then
    echo "AUTO_MIXER_INPUTS matched fewer than two inputs: ${AUTO_MIXER_INPUTS:-/media-inputs/*}" >&2
    exit 2
  fi

  args+=(--inputs "${inputs[@]}")

  case "${AUTO_MIXER_OUTPUT:-janus}" in
    ""|janus)
      args+=(--janus-output)
      ;;
    *)
      args+=(--output "${AUTO_MIXER_OUTPUT}")
      ;;
  esac

  if is_true "${AUTO_MIXER_TALKSHOW_PROFILE:-1}"; then
    args+=(--talkshow-profile)
  fi

  if [[ -n "${AUTO_MIXER_INPUT_START_TS:-}" ]]; then
    args+=(--input-start-ts "${AUTO_MIXER_INPUT_START_TS}")
  fi

  if [[ -n "${AUTO_MIXER_HTML_OVERLAY_URL:-}" ]]; then
    args+=(--html-overlay-url "${AUTO_MIXER_HTML_OVERLAY_URL}")
  fi
fi

exec /usr/local/bin/avp-auto-mixer-entrypoint "${args[@]}"
