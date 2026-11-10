#!/usr/bin/env bash
#
# NVIDIA Nsight Systems for an avplumber image: which CUDA kernels the GPU spends its time on.
#
#   tools/nsight/nsight.sh image <base-image> [<tag>]
#       Build <tag> (default: <base-image>-nsys): the base image plus the Nsight Systems CLI from
#       NVIDIA's CUDA package repository for the base image's distribution. The base is not changed.
#       Fedora and RHEL-family images (the ones avplumber's Dockerfiles build) are supported.
#
#   tools/nsight/nsight.sh summary <trace.nsys-rep> [<nsys-image>]
#       Print the trace's GPU time per kernel and grid: launches per second, mean duration and the
#       share of the traced window, with the totals. <nsys-image> is one built by `image`
#       (default: $NSIGHT_IMAGE).
#
# Taking a trace is the deployment's own command with `nsys profile` in front of it, in the image
# built above, for example for a compose service:
#
#   docker compose run --rm --no-deps --entrypoint nsys <service> \
#     profile -t cuda -s none --cpuctxsw=none --delay=60 --duration=40 \
#     -o <mounted dir>/trace --force-overwrite true <the service's command and arguments>
#
# --delay skips start-up, --duration ends the trace and then stops the process with SIGTERM.
# Stop the regular service first: a second copy beside it shares the GPU and measures neither.
# Tracing costs some CPU per CUDA call; compare the GPU load during the trace with the usual one
# before trusting the shares. TensorRT inference runs as CUDA graphs, which this summary counts
# as launches only: their GPU time is what the GPU's utilisation leaves over the kernels' share.
# Adding --cuda-graph-trace=node lists the kernels inside the graphs too, at a higher tracing cost.
#
# DOCKER overrides the docker command, e.g. DOCKER="sudo docker".
set -euo pipefail

DOCKER=${DOCKER:-docker}
REPOS=https://developer.download.nvidia.com/compute/cuda/repos
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "nsight.sh: $*" >&2; exit 1; }

build_image() {
    local base=$1 tag=${2:-$1-nsys} id version repo package
    read -r id version < <($DOCKER run --rm --entrypoint sh "$base" -c '. /etc/os-release; echo "$ID $VERSION_ID"') \
        || die "cannot read /etc/os-release of $base"
    case $id in
        fedora) repo=fedora${version} ;;
        rhel|rocky|almalinux|centos) repo=rhel${version%%.*} ;;
        *) die "no rule for distribution '$id' of $base: add its repository under $REPOS" ;;
    esac
    # The newest nsight-systems package of that repository; it carries the CLI and its CUPTI.
    package=$(curl -fsSL "$REPOS/$repo/x86_64/" | grep -o 'nsight-systems-20[0-9][0-9][^"<>]*\.x86_64\.rpm' | sort -uV | tail -1) \
        || die "cannot list $REPOS/$repo/x86_64/"
    [ -n "$package" ] || die "no nsight-systems package in $REPOS/$repo/x86_64/"
    echo "nsight.sh: $base is $id $version; installing $package as $tag" >&2
    # --nodeps: the package also asks for the libraries of its GUI, which a trace does not need.
    $DOCKER build -t "$tag" - <<DOCKERFILE
FROM $base
RUN rpm -ivh --nodeps $REPOS/$repo/x86_64/$package && nsys --version
DOCKERFILE
}

summary() {
    local trace=$1 image=${2:-${NSIGHT_IMAGE:-}}
    [ -f "$trace" ] || die "no trace at $trace"
    [ -n "$image" ] || die "name the image built by 'nsight.sh image', or set NSIGHT_IMAGE"
    local dir name
    dir="$(cd "$(dirname "$trace")" && pwd)"; name="$(basename "$trace")"
    $DOCKER run --rm -v "$dir":/trace -v "$HERE/summary.py":/summary.py:ro --entrypoint sh "$image" -c \
        "nsys export --type sqlite --force-overwrite true --output /tmp/trace.sqlite /trace/'$name' >/tmp/export.log 2>&1 \\
         || { cat /tmp/export.log >&2; exit 1; }; python3 /summary.py /tmp/trace.sqlite"
}

case ${1:-} in
    image) [ $# -ge 2 ] || die "usage: nsight.sh image <base-image> [<tag>]"; build_image "$2" "${3:-}" ;;
    summary) [ $# -ge 2 ] || die "usage: nsight.sh summary <trace.nsys-rep> [<nsys-image>]"; summary "$2" "${3:-}" ;;
    *) sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
