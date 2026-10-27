# Shared config/helpers for scripts/*.sh. Source with `. "$(dirname "$0")/_lib.sh"`.
#
# Everything here talks to the demo host (AWS, Fedora) over plain
# `ssh`/`rsync` and a `docker compose` toolchain -- no git on that host.
# Config comes from .dev.env (gitignored) next to this scripts/ dir; copy it
# from .dev.env.example and edit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> demos/mixer_audio
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# demos/mixer_audio -> avplumber checkout root
REPO_ROOT="$(cd "${DEMO_DIR}/../.." && pwd)"

DEV_ENV="${SCRIPT_DIR}/.dev.env"
[[ -f "$DEV_ENV" ]] || { echo "[mixer-audio] ERROR: $DEV_ENV not found. Copy .dev.env.example and edit." >&2; exit 1; }
source "$DEV_ENV"

: "${EC2_USER:?missing in .dev.env}"
: "${EC2_HOST:?missing in .dev.env}"
: "${EC2_PEM:?missing in .dev.env}"
: "${EC2_REPO:?missing in .dev.env}"

COMPOSE_REL="docker-compose/mixer_audio/compose.yaml"
EC2_AVPLUMBER="${EC2_REPO}/avplumber"

log()  { echo "[mixer-audio] $*"; }
die()  { echo "[mixer-audio] ERROR: $*" >&2; exit 1; }

ssh_run() {
  ssh -i "$EC2_PEM" "$EC2_USER@$EC2_HOST" "$1"
}
