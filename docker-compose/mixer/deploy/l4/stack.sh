#!/usr/bin/env bash
# The portable CUarray stack; every Compose command uses the same pins and paths.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
if [[ -f "$here/.env" ]]; then
    set -a
    source "$here/.env"
    set +a
fi
source "$repo/deps/ffmpeg/9/bases.env"
export FFMPEG_TAG="$upstream_ref"
if [[ -z "${AVPLUMBER_REVISION:-}" && -f "$repo/SOURCE_REVISION" ]]; then
    read -r source_project source_revision < "$repo/SOURCE_REVISION"
    [[ "$source_project" = avplumber && "$source_revision" =~ ^[0-9a-f]{40}$ ]] || {
        echo "Invalid SOURCE_REVISION" >&2; exit 2;
    }
    export AVPLUMBER_REVISION="$source_revision"
fi
export AVPLUMBER_REVISION="${AVPLUMBER_REVISION:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo workspace)}"
export MIXER_RELEASE="${MIXER_RELEASE:-${AVPLUMBER_REVISION:0:12}}"
export MIXER_INSTANCE_TYPE=nvidia_l4_cuarray
export MIXER_HTTP_PORT="${MIXER_HTTP_PORT:-17681}"
# Only the proxy reaches the page, so MIXER_HTPASSWD cannot be bypassed on MIXER_HTTP_PORT.
export MIXER_BIND="${MIXER_BIND:-127.0.0.1}"
export JANUS_PREVIEW_PORT="${JANUS_PREVIEW_PORT:-18080}"
export MIXER_PREVIEW_BASE=/preview/
export MIXER_WEBUI_URL=http://127.0.0.1:22222
export MIXER_RECIPE=/media/demo.json
export MIXER_MEDIA_DIR="${MIXER_MEDIA_DIR:-$repo/media}"
export MIXER_INITIAL_SETTINGS="${MIXER_INITIAL_SETTINGS:-$here/settings.json}"
export MIXER_AUTH_REALM=off
if [[ -n "${MIXER_HTPASSWD:-}" ]]; then
    [[ -f "$MIXER_HTPASSWD" ]] || { echo "MIXER_HTPASSWD must name an existing file" >&2; exit 2; }
    export MIXER_AUTH_REALM=Mixer
fi
if [[ $# == 0 ]]; then
    echo "Usage: $0 {build|up -d|down|logs -f|config|...}" >&2
    exit 2
fi
if [[ $1 == up ]] && command -v nvidia-smi >/dev/null; then
    # Without persistence mode the driver tears the GPU down whenever no process holds it,
    # so every container start pays its initialisation again.
    if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | grep -q Disabled; then
        echo "stack.sh: GPU persistence mode is off; enable it with" \
            "'sudo systemctl enable --now nvidia-persistenced' (README.md)" >&2
    fi
    # The compute time-slice resets on every boot; gcp/host-setup.sh sets it on GCE.
    if ! nvidia-smi compute-policy -l 2>/dev/null | grep -q 'Timeslice *Long'; then
        echo "stack.sh: the GPU compute time-slice is not Long; set it with" \
            "'sudo nvidia-smi compute-policy --set-timeslice=3' (README.md)" >&2
    fi
fi
exec docker compose --env-file /dev/null --project-directory "$repo/docker-compose/mixer" \
    -f "$repo/docker-compose/mixer/compose.yaml" -f "$here/compose.yaml" "$@"
