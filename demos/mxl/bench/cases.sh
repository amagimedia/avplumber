#!/usr/bin/env bash
# Groups of bench.sh cases. Each group reproduces one table or claim in
# ../README.md; `cases.sh list` says which.
#
#   cases.sh <group> [bench.sh or mxl_demo.py args...]
#
# Trailing arguments are appended to every case, so they beat the group's own
# defaults: `cases.sh smoke --exec avpbuild`, `cases.sh pack --seconds 10`.
set -uo pipefail
bench_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# 1080p59.94 bars everywhere. The cases that publish a 1080p source into a
# smaller flow name it explicitly, since bench.sh otherwise derives the
# source geometry from the flow's.
src_1080=${AVP_BENCH_SRC_1080:-lavfi:smptehdbars=size=1920x1080:rate=60000/1001}
src_1080_10bit="$src_1080,format=yuv422p10le"

groups=(
    "throughput|Measured throughput: the 11-row table, 60 s per case"
    "nodecpu|Where the CPU goes: per-node ms/frame, tail by tail"
    "conversion|Conversion on the GPU: swscale against scale_cuda, both legs"
    "pack|Packing on the GPU: the CPU and GPU v210 packs, paced and unpaced"
    "zerocopy|The r_input rows: zero-copy on and off, three runs each"
    "swsflags|The --sws-flags rows: flag sweep on the conversions"
    "srcfmt|Whether a 10-bit 4:2:2 source removes the writer's swscale cost"
    "tails|Where the unpaced CPU reader caps, tail by tail"
    "smoke|Every graph shape builds and runs; short, unsampled"
)

group_defaults=()
defaults() { group_defaults=("$@"); }

run() {
    local label=$1; shift
    "$bench_dir/bench.sh" --label "$label" \
        "${group_defaults[@]+"${group_defaults[@]}"}" "$@" \
        "${passthrough[@]+"${passthrough[@]}"}"
    echo
}

# ../README.md "Measured throughput". One container per case so the cgroup
# CPU figure covers this run and nothing else.
group_throughput() {
    defaults --seconds 60 --warmup 10
    run gpu-zc-paced                  --gpu-unpack on
    run gpu-nozc-paced                --gpu-unpack on  --no-zero-copy
    run cpu-zc-paced                  --gpu-unpack off
    run gpu-zc-unpaced                --gpu-unpack on  --writer-pace off
    run gpu-nozc-unpaced              --gpu-unpack on  --writer-pace off --no-zero-copy
    run cpu-zc-unpaced                --gpu-unpack off --writer-pace off
    run writer-only-unpaced           --writer-only --writer-pace off
    run writer-only-unpaced-fastbl    --writer-only --writer-pace off --sws-flags fast_bilinear
    run writer-only-unpaced-gpu-scale --writer-only --writer-pace off --gpu-scale writer
    run writer-only-unpaced-gpu-pack  --writer-only --writer-pace off \
                                      --gpu-scale writer --writer-pack gpu
    run roundtrip-unpaced-gpu-pack    --writer-pace off --gpu-unpack on \
                                      --gpu-scale writer --writer-pack gpu
}

# ../README.md "Where the CPU goes". The per-node table comes from the demo's
# own /proc/self/task sampling, so exec mode reports it just as well; the
# first five cases also show that the CPU path needs no GPU at all.
group_nodecpu() {
    defaults --seconds 30 --warmup 10 --output /dev/null
    run nodecpu-demux         --no-gpu --gpu-unpack off --reader-tail demux
    run nodecpu-unpack        --no-gpu --gpu-unpack off --reader-tail unpack
    run nodecpu-encode        --no-gpu --gpu-unpack off --reader-tail encode
    run nodecpu-encode-fastbl --no-gpu --gpu-unpack off --reader-tail encode \
                              --sws-flags fast_bilinear
    run nodecpu-encode-nozc   --no-gpu --gpu-unpack off --reader-tail encode --no-zero-copy
    run nodecpu-unpack-gpu    --gpu-unpack on --reader-tail unpack
}

# ../README.md "Conversion on the GPU".
group_conversion() {
    defaults --seconds 40 --warmup 10 --output /dev/null
    run w1-sws-default            --writer-only
    run w2-sws-fast-bilinear      --writer-only --sws-flags fast_bilinear
    run w3-scale-cuda             --writer-only --gpu-scale writer

    run w4-sws-default-unpaced    --writer-only --writer-pace off
    run w5-sws-fastbl-unpaced     --writer-only --writer-pace off --sws-flags fast_bilinear
    run w6-scale-cuda-unpaced     --writer-only --writer-pace off --gpu-scale writer

    # A real resize: 1080p source into a 720p flow.
    run w7-720p-sws-default       --width 1280 --height 720 --input "$src_1080" \
                                  --writer-only --writer-pace off
    run w8-720p-sws-lanczos       --width 1280 --height 720 --input "$src_1080" \
                                  --writer-only --writer-pace off --sws-flags lanczos
    run w9-720p-scale-cuda-lanczos --width 1280 --height 720 --input "$src_1080" \
                                  --writer-only --writer-pace off \
                                  --gpu-scale writer --cuda-interp lanczos

    # The reader's conversion alone, with the output encoder out of the way.
    run r1-sws                    --reader-tail scale --gpu-unpack off
    run r2-scale-cuda-up-down     --reader-tail scale --gpu-unpack off --gpu-scale reader
    run r3-scale-cuda-down-only   --reader-tail scale --gpu-unpack on --reader-encoder mpeg4
    run r4-scale-cuda-no-copy     --reader-tail scale --gpu-unpack on

    run t1-all-cpu                --gpu-unpack off
    run t2-cpu-scale-cuda-both    --gpu-unpack off --gpu-scale both
    run t3-gpu-unpack-mpeg4       --gpu-unpack on --reader-encoder mpeg4
    run t4-gpu-unpack-nvenc       --gpu-unpack on

    run t5-all-cpu-unpaced        --gpu-unpack off --writer-pace off
    run t6-cpu-scale-cuda-both-unpaced --gpu-unpack off --gpu-scale both --writer-pace off
    run t7-gpu-unpack-mpeg4-unpaced --gpu-unpack on --reader-encoder mpeg4 --writer-pace off
}

# ../README.md "Packing on the GPU". The round trips keep their output file:
# a GPU-packed grain still has to come back as a decodable 1800-frame mp4.
group_pack() {
    defaults --seconds 30 --warmup 10
    run pack-cpu-paced   --output /dev/null --writer-only --gpu-scale writer --writer-pack cpu
    run pack-gpu-paced   --output /dev/null --writer-only --gpu-scale writer --writer-pack gpu
    run pack-cpu-unpaced --output /dev/null --writer-only --writer-pace off \
                         --gpu-scale writer --writer-pack cpu
    run pack-gpu-unpaced --output /dev/null --writer-only --writer-pace off \
                         --gpu-scale writer --writer-pack gpu
    run pack-gpu-roundtrip-cpu-reader --gpu-scale writer --writer-pack gpu --gpu-unpack off
    run pack-gpu-roundtrip-gpu-reader --gpu-scale writer --writer-pack gpu --gpu-unpack on
    run pack-cpu-roundtrip-gpu-reader --gpu-scale writer --writer-pack cpu --gpu-unpack on
}

# ../README.md "Where the CPU goes" / "Zero-copy grains": the difference is
# one copy of 5.5 MB, which is small enough to want three runs each way.
group_zerocopy() {
    defaults --seconds 25 --warmup 8 --output /dev/null \
             --gpu-unpack off --reader-tail demux
    local i
    for i in 1 2 3; do
        run "zc-on-$i"
        run "zc-off-$i" --no-zero-copy
    done
}

group_swsflags() {
    defaults --seconds 30 --warmup 10 --output /dev/null
    run sws-roundtrip-default      --gpu-unpack off --reader-tail encode
    run sws-roundtrip-fastbl       --gpu-unpack off --reader-tail encode \
                                   --sws-flags fast_bilinear
    run sws-roundtrip-point        --gpu-unpack off --reader-tail encode --sws-flags point
    run sws-roundtrip-bilinear     --gpu-unpack off --reader-tail encode --sws-flags bilinear
    run sws-writer-default         --writer-only --writer-pace off
    run sws-writer-fast-bilinear   --writer-only --writer-pace off --sws-flags fast_bilinear
    run sws-writer-point           --writer-only --writer-pace off --sws-flags point
}

group_srcfmt() {
    defaults --seconds 30 --warmup 10 --output /dev/null --gpu-unpack off
    run srcfmt-8bit-demux        --input "$src_1080"       --reader-tail demux
    run srcfmt-10bit-demux       --input "$src_1080_10bit" --reader-tail demux
    run srcfmt-10bit-encode      --input "$src_1080_10bit" --reader-tail encode
    run srcfmt-10bit-demux-point --input "$src_1080_10bit" --reader-tail demux \
                                 --sws-flags point
}

# Writer unpaced too, so the ring is always ahead of the reader: drops are
# expected here and do not affect the rate.
group_tails() {
    defaults --seconds 30 --warmup 10 --output /dev/null --writer-pace off
    run tail-cpu-demux        --gpu-unpack off --reader-tail demux
    run tail-cpu-unpack       --gpu-unpack off --reader-tail unpack
    run tail-cpu-encode       --gpu-unpack off --reader-tail encode
    run tail-cpu-encode-point --gpu-unpack off --reader-tail encode --sws-flags point
    run tail-gpu-unpack       --gpu-unpack on  --reader-tail unpack
}

# Does every shape still build? Numbers from 14 s windows are worthless, so
# read the exit status, the log counts and the output probe instead.
group_smoke() {
    defaults --seconds 14 --warmup 6 --no-sample
    run smoke-writer-gpu-scale    --writer-only --gpu-scale writer
    run smoke-writer-gpu-pack     --writer-only --gpu-scale writer --writer-pack gpu
    run smoke-cpu-unpack-gpu-scale --gpu-unpack off --gpu-scale reader
    run smoke-gpu-unpack-mpeg4    --gpu-unpack on  --reader-encoder mpeg4
    run smoke-cpu-unpack-nvenc    --gpu-unpack off --reader-encoder nvenc
    run smoke-gpu-full-tail       --gpu-unpack on
    run smoke-cpu-full-tail       --gpu-unpack off
    run smoke-scale-tail-cpu-sws  --gpu-unpack off --reader-tail scale
    run smoke-scale-tail-cpu-gpu  --gpu-unpack off --reader-tail scale --gpu-scale reader
    run smoke-scale-tail-gpu-mpeg4 --gpu-unpack on --reader-tail scale --reader-encoder mpeg4
    run smoke-scale-tail-gpu-nvenc --gpu-unpack on --reader-tail scale
}

usage() {
    echo "usage: ${0##*/} <group> [bench.sh or mxl_demo.py args...]"
    echo "groups:"
    printf '  %-12s %s\n' "list" "this list"
    local entry
    for entry in "${groups[@]}"; do
        printf '  %-12s %s\n' "${entry%%|*}" "${entry#*|}"
    done
}

group=${1:-}
[[ -n $group && $group != -h && $group != --help ]] || { usage; exit 2; }
[[ $group == list ]] && { usage; exit 0; }
declare -F "group_$group" >/dev/null || { echo "unknown group: $group" >&2; usage >&2; exit 2; }
shift
passthrough=("$@")
"group_$group"
