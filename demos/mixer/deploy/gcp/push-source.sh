#!/usr/bin/env bash
# Copy the committed source tree and its submodules to a host, where the mixer
# image is built from exactly that commit:
#   PROJECT=<project> demos/mixer/deploy/gcp/push-source.sh <host> [remote-dir]
# Uncommitted changes are not sent. The commit is recorded in SOURCE_REVISION.
set -euo pipefail
host="${1:?usage: PROJECT=<project> push-source.sh <host> [remote-dir]}"
remote="${2:-/srv/avplumber}"
PROJECT="${PROJECT:?set PROJECT to the GCP project}"
repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
zone="$(gcloud compute instances list --project "$PROJECT" --filter="name=($host)" --format='value(zone.basename())')"
[ -n "$zone" ] || { echo "ERROR: no instance named $host in $PROJECT" >&2; exit 1; }
[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ] || echo "WARNING: uncommitted changes are not copied to the host" >&2
revision="$(git -C "$repo" rev-parse HEAD)"
{
  git -C "$repo" archive --format=tar HEAD
  # Only checked-out submodules: in an empty path git archive would archive the parent again.
  git -C "$repo" submodule --quiet foreach 'echo "$sm_path"' | while read -r sub; do
    git -C "$repo/$sub" archive --format=tar --prefix="$sub/" HEAD
  done
} | gcloud compute ssh "$host" --project "$PROJECT" --zone "$zone" --quiet --command "
  set -e; sudo rm -rf '$remote.new' && sudo mkdir -p '$remote.new' '$remote' && sudo chown \$(id -u):\$(id -g) '$remote.new' '$remote'
  tar -x -i -C '$remote.new' -f -
  # The host's own data and settings stay: the generated media and the stack's .env.
  rsync -a --delete --exclude /media/ --exclude /demos/mixer/.env '$remote.new/' '$remote/' && rm -rf '$remote.new'
  printf 'avplumber %s\n' '$revision' > '$remote/SOURCE_REVISION' && cat '$remote/SOURCE_REVISION'"
