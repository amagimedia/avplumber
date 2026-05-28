#!/usr/bin/env bash
set -euo pipefail

: "${AVP_SCRIPT:?AVP_SCRIPT must point to the mounted .avplumber script}"

if [[ -z "${AVP_SCRIPT_FILE:-}" ]]; then
  echo "AVP_SCRIPT_FILE must be set to the host .avplumber script path" >&2
  exit 2
fi

if [[ ! -f "$AVP_SCRIPT" ]]; then
  echo "mounted avplumber script is not a file: $AVP_SCRIPT" >&2
  exit 2
fi

args=(/usr/local/bin/avplumber)

if [[ -n "${AVP_LOGFILE:-}" ]]; then
  args+=(--logfile "$AVP_LOGFILE")
fi

if [[ -n "${AVP_WEBUI_API:-}" ]]; then
  args+=(--webui-api "$AVP_WEBUI_API")
fi

if [[ -n "${AVP_REMOTE_CONTROL_PORT:-}" && "${AVP_REMOTE_CONTROL_PORT}" != "0" ]]; then
  args+=(--port "$AVP_REMOTE_CONTROL_PORT")
fi

if [[ -n "${AVP_INSTANCE_NAME:-}" ]]; then
  args+=(--instance-name "$AVP_INSTANCE_NAME")
fi

if [[ -n "${AVP_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$AVP_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

exec "${args[@]}" -s "$AVP_SCRIPT"
