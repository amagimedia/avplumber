#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${script_dir}/.env"

usage() {
  cat <<'EOF'
Usage:
  docker-compose/run-demo.sh [--env FILE] [docker compose args...]

Examples:
  docker-compose/run-demo.sh --env docker-compose/demos/custom-script.env --profile script up
  docker-compose/run-demo.sh --env docker-compose/.env up -d
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env-file)
      env_file="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -f "$env_file" ]]; then
  echo "env file not found: $env_file" >&2
  echo "copy docker-compose/.env.example or one of docker-compose/demos/*.env.example first" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  set -- --profile script up
fi

compose_cmd=()
if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "docker compose is not available" >&2
  exit 127
fi

exec "${compose_cmd[@]}" --env-file "$env_file" -f "${script_dir}/docker-compose.yml" "$@"
