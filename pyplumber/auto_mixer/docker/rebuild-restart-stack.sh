#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
MIXER_STACK="${MIXER_STACK:-$SCRIPT_DIR/mixer-stack.sh}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_SYNC=1
DISCARD_LOCAL_CHANGES=0
PRUNE_OLD_DOCKER=1
PRUNE_BUILD_CACHE=0
BUILD_SERVICES=(dma-browser auto-mixer)
BRANCH="${AVP_GIT_BRANCH:-}"
MIXER_ARGS=()
DOCKER_CMD=()

usage() {
  cat <<'EOF'
Usage:
  rebuild-restart-stack.sh [options] [-- <auto-mixer args...>]

Options:
  --branch BRANCH      Git branch to fetch and fast-forward before rebuilding.
                       Defaults to AVP_GIT_BRANCH, then the current branch.
  --remote REMOTE      Git remote to fetch from. Defaults to origin.
  --no-git-sync        Do not fetch or fast-forward the working tree.
  --discard-local-changes
                       Reset tracked changes and remove untracked files before
                       checking out the target branch. The env file is kept.
  --no-prune           Do not remove old project containers/images or dangling
                       image layers after a successful restart.
  --prune-build-cache  Also remove Docker build cache after a successful
                       restart. This saves disk but makes the next rebuild slow.
  --env-file PATH      Env file for docker compose and mixer-stack.sh.
                       Defaults to pyplumber/auto_mixer/docker/.env.
  -h, --help           Show this help.

Auto-mixer args:
  Pass args after --, or set AUTO_MIXER_ARGS in the env file. The script
  rebuilds dma-browser and auto-mixer, then calls mixer-stack.sh restart -- ...

Examples:
  ./rebuild-restart-stack.sh --branch <branch>
  AVP_GIT_BRANCH=<branch> ./rebuild-restart-stack.sh
  ./rebuild-restart-stack.sh --no-git-sync -- --inputs /media-inputs/cam0.ts /media-inputs/cam1.ts
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

current_branch() {
  git -C "$REPO_DIR" branch --show-current
}

env_file_relative_to_repo() {
  local env_dir env_abs
  env_dir="$(cd -- "$(dirname -- "$ENV_FILE")" && pwd)"
  env_abs="$env_dir/$(basename -- "$ENV_FILE")"

  case "$env_abs" in
    "$REPO_DIR"/*)
      printf '%s\n' "${env_abs#"$REPO_DIR"/}"
      ;;
    *)
      return 1
      ;;
  esac
}

discard_worktree_changes() {
  local env_rel
  git -C "$REPO_DIR" reset --hard

  if env_rel="$(env_file_relative_to_repo)" && [[ -f "$ENV_FILE" ]]; then
    git -C "$REPO_DIR" clean -fd -e "$env_rel"
  else
    git -C "$REPO_DIR" clean -fd
  fi

  git -C "$REPO_DIR" submodule update --init --recursive
  git -C "$REPO_DIR" submodule foreach --recursive 'git reset --hard && git clean -fd'
}

require_branch() {
  if [[ -z "$BRANCH" ]]; then
    BRANCH="$(current_branch)"
  fi

  [[ -n "$BRANCH" ]] || die "branch is required outside a named branch; pass --branch or set AVP_GIT_BRANCH"
  [[ "$BRANCH" != *".."* && "$BRANCH" != /* && "$BRANCH" != *" "* ]] || die "invalid branch name: $BRANCH"
}

git_sync() {
  require_branch

  git -C "$REPO_DIR" fetch "$GIT_REMOTE" "$BRANCH"

  if ((DISCARD_LOCAL_CHANGES)); then
    discard_worktree_changes
    git -C "$REPO_DIR" checkout -B "$BRANCH" "$GIT_REMOTE/$BRANCH"
    git -C "$REPO_DIR" reset --hard "$GIT_REMOTE/$BRANCH"
    git -C "$REPO_DIR" submodule update --init --recursive
    return 0
  fi

  if [[ "$(current_branch)" != "$BRANCH" ]]; then
    if git -C "$REPO_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
      git -C "$REPO_DIR" checkout "$BRANCH"
    else
      git -C "$REPO_DIR" checkout -b "$BRANCH" "$GIT_REMOTE/$BRANCH"
    fi
  fi

  git -C "$REPO_DIR" merge --ff-only "$GIT_REMOTE/$BRANCH"
  git -C "$REPO_DIR" submodule update --init --recursive
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
    DOCKER_CMD=(sudo docker)
    return 0
  fi

  die "docker is not available; set DOCKER_CMD_OVERRIDE or enable passwordless docker access"
}

docker_cmd() {
  "${DOCKER_CMD[@]}" "$@"
}

compose() {
  docker_cmd compose --project-directory "$SCRIPT_DIR" --env-file "$ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yml" "$@"
}

remove_old_project_containers() {
  local project="$1"
  local -a containers=()
  local status

  [[ -n "$project" ]] || return 0

  for status in created exited dead; do
    while IFS= read -r container_id; do
      [[ -n "$container_id" ]] && containers+=("$container_id")
    done < <(docker_cmd ps -a -q --filter "label=com.docker.compose.project=$project" --filter "status=$status")
  done

  ((${#containers[@]})) || return 0
  mapfile -t containers < <(printf '%s\n' "${containers[@]}" | sort -u)
  printf 'removing old project containers: %s\n' "${containers[*]}"
  docker_cmd rm "${containers[@]}" >/dev/null || true
}

remove_old_project_images() {
  local keep_file candidates_file image image_id repo tag
  keep_file="$(mktemp)"
  candidates_file="$(mktemp)"

  while IFS= read -r image; do
    [[ -n "$image" ]] || continue
    docker_cmd image inspect -f '{{.Id}}' "$image" 2>/dev/null || true
  done < <(compose config --images | sort -u) >>"$keep_file"

  while IFS= read -r container_id; do
    [[ -n "$container_id" ]] || continue
    docker_cmd inspect -f '{{.Image}}' "$container_id" 2>/dev/null || true
  done < <(docker_cmd ps -q) >>"$keep_file"

  sort -u -o "$keep_file" "$keep_file"

  while read -r repo tag; do
    [[ "$repo" == avplumber-auto-mixer* ]] || continue
    docker_cmd image inspect -f '{{.Id}}' "$repo:$tag" 2>/dev/null || true
  done < <(docker_cmd image ls --format '{{.Repository}} {{.Tag}}') | sort -u >"$candidates_file"

  while IFS= read -r image_id; do
    [[ -n "$image_id" ]] || continue
    if grep -qx "$image_id" "$keep_file"; then
      continue
    fi
    printf 'removing old project image: %s\n' "$image_id"
    docker_cmd image rm "$image_id" >/dev/null || true
  done <"$candidates_file"

  rm -f "$keep_file" "$candidates_file"
}

prune_old_docker_state() {
  local project
  project="${COMPOSE_PROJECT_NAME:-avplumber-auto-mixer}"

  remove_old_project_containers "$project"
  remove_old_project_images
  docker_cmd image prune -f >/dev/null || true
  if ((PRUNE_BUILD_CACHE)); then
    docker_cmd builder prune -f >/dev/null || true
  fi
}

parse_args() {
  while (($#)); do
    case "$1" in
      --branch)
        [[ $# -ge 2 ]] || die "--branch requires a value"
        BRANCH="$2"
        shift 2
        ;;
      --remote)
        [[ $# -ge 2 ]] || die "--remote requires a value"
        GIT_REMOTE="$2"
        shift 2
        ;;
      --no-git-sync)
        GIT_SYNC=0
        shift
        ;;
      --discard-local-changes)
        DISCARD_LOCAL_CHANGES=1
        shift
        ;;
      --no-prune)
        PRUNE_OLD_DOCKER=0
        shift
        ;;
      --prune-build-cache)
        PRUNE_BUILD_CACHE=1
        shift
        ;;
      --env-file)
        [[ $# -ge 2 ]] || die "--env-file requires a value"
        ENV_FILE="$2"
        shift 2
        ;;
      --)
        shift
        MIXER_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option before --: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  if ((GIT_SYNC)); then
    git_sync
  else
    require_branch
  fi

  [[ -x "$MIXER_STACK" ]] || die "mixer-stack.sh is not executable: $MIXER_STACK"
  [[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE"

  select_docker_cmd

  printf 'rebuild branch=%s remote=%s repo=%s\n' "$BRANCH" "$GIT_REMOTE" "$REPO_DIR"
  compose build "${BUILD_SERVICES[@]}"

  if ((${#MIXER_ARGS[@]})); then
    ENV_FILE="$ENV_FILE" AVP_GIT_BRANCH="$BRANCH" "$MIXER_STACK" restart --force-recreate -- "${MIXER_ARGS[@]}"
  else
    ENV_FILE="$ENV_FILE" AVP_GIT_BRANCH="$BRANCH" "$MIXER_STACK" restart --force-recreate
  fi

  if ((PRUNE_OLD_DOCKER)); then
    prune_old_docker_state
  fi
}

main "$@"
