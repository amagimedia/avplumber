#!/usr/bin/env bash
# The CUDA v210 pack has to produce exactly what the libavcodec v210 encoder
# produces. The frames reaching it are the same either way (scale_cuda
# converts in both runs), so the published grains must match byte for byte —
# including the row padding to a 128-byte multiple and, at widths that are not
# a multiple of 6, the partial block at the end of every row.
#
#   pack_bitexact.sh [options] [WxH...]
#
# Default geometries are 1920x1080 (whole blocks), 1280x720 and 1918x1080
# (both ending a row mid-block). Exits nonzero unless every geometry matches.
set -uo pipefail
source "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# One fixed flow id: the dumping ffmpeg has to name the flow it reads.
flow_id=${AVP_MXL_VIDEO_ID:-11111111-2222-3333-4444-555555555555}
frames=${AVP_BENCH_PACK_FRAMES:-2}
settle=${AVP_BENCH_PACK_SETTLE:-12}
geometries=()
while [[ $# -gt 0 ]]; do
    bench_common_arg "$@"; consumed=$?
    if [[ $consumed -gt 0 ]]; then shift "$consumed"; continue; fi
    case $1 in
        --frames) frames=${2:?missing value for --frames}; shift 2 ;;
        --settle) settle=${2:?missing value for --settle}; shift 2 ;;
        -h|--help)
            awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
            echo; bench_common_usage
            cat <<'EOF'
  --frames N         grains to dump per run (default 2)
  --settle N         seconds to let the writer publish first (default 12)
EOF
            exit 0 ;;
        *) geometries+=("$1"); shift ;;
    esac
done
[[ ${#geometries[@]} -gt 0 ]] || geometries=(1920x1080 1280x720 1918x1080)

bench_setup
dump_dir=$bench_out/pack
trap 'rm -rf -- "$domain"' EXIT

# Publish with one pack and dump the first grains of the ring. Writer and
# ffmpeg share a container: the reader has to see the same /dev/shm, and in
# exec mode there is only one container to begin with.
dump() {  # $1 = WxH, $2 = cpu|gpu
    local geom=$1 pack=$2 w=${1%x*} h=${1#*x} base
    base=$dump_dir/$1.$2
    bench_domain_reset
    bench_in_container_bash "
        mkdir -p '$dump_dir'
        rm -f '$base.v210'
        python3 $demo_in_container --domain '$domain' --video-flow-id $flow_id \
            --writer-only --width $w --height $h --fps '$fps' \
            --input 'lavfi:smptehdbars=size=${w}x${h}:rate=$fps' \
            --gpu-scale writer --writer-pack $pack >'$base.writer.log' 2>&1 &
        writer=\$!
        sleep $settle
        # -grain_index_init head starts at the oldest grain still in the ring,
        # so both runs dump the same frames of a static pattern; -c copy keeps
        # the v210 bytes as the muxer wrote them.
        ffmpeg -nostdin -loglevel warning -blocking 1 -grain_index_init head \
            -f mxl -i 'mxl://$domain?id=$flow_id' -frames:v $frames \
            -c copy -f rawvideo -y '$base.v210' >'$base.ffmpeg.log' 2>&1
        rc=\$?
        kill \$writer 2>/dev/null
        wait \$writer 2>/dev/null
        [ \$rc = 0 ] || { echo \"  $geom $pack: ffmpeg exit \$rc\"; tail -3 '$base.ffmpeg.log'; }
        [ -s '$base.v210' ] || { echo \"  $geom $pack: nothing dumped\"; tail -5 '$base.writer.log'; }
        exit \$rc" 2>&1
}

compare() {  # $1 = WxH
    bench_in_container_bash "
        cd '$dump_dir' || exit 1
        ls -l '$1.cpu.v210' '$1.gpu.v210' | awk '{print \"  \" \$5, \$9}'
        if cmp '$1.cpu.v210' '$1.gpu.v210'; then
            md5sum '$1.cpu.v210' | sed 's/^/  /'
            exit 0
        fi
        echo \"  differing bytes: \$(cmp -l '$1.cpu.v210' '$1.gpu.v210' | wc -l)\"
        exit 1" 2>&1
}

fail=0
for geom in "${geometries[@]}"; do
    echo "=== $geom"
    dump "$geom" cpu
    dump "$geom" gpu
    if compare "$geom"; then
        echo "  $geom: IDENTICAL"
    else
        echo "  $geom: DIFFERS"
        fail=1
    fi
    echo
done
if [[ $mode == exec ]]; then
    echo "dumps and logs in $container:$dump_dir"
else
    echo "dumps and logs: $dump_dir"
fi
exit "$fail"
