#!/usr/bin/env bash
# What the GPU conversion does to the pixels, and what the GPU pays for taking
# it — the checks behind ../README.md's "Conversion on the GPU" that are not
# an fps number.
#
#   pixel_compare.sh [options] [psnr] [pcie] [formats]     (default: all three)
#
#   psnr     the same flow published with swscale and with scale_cuda, read
#            back through the CPU reader both times, compared by PSNR and SSIM
#   pcie     nvidia-smi dmon over a paced run: SM time and PCIe traffic
#   formats  which pixel formats this build's scale_cuda accepts
set -uo pipefail
source "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

seconds=${AVP_BENCH_SECONDS:-40}
warmup=${AVP_BENCH_WARMUP:-8}
dmon_delay=${AVP_BENCH_DMON_DELAY:-20}
dmon_count=${AVP_BENCH_DMON_COUNT:-8}
sections=()
while [[ $# -gt 0 ]]; do
    bench_common_arg "$@"; consumed=$?
    if [[ $consumed -gt 0 ]]; then shift "$consumed"; continue; fi
    case $1 in
        --dmon-delay) dmon_delay=${2:?missing value for --dmon-delay}; shift 2 ;;
        --dmon-count) dmon_count=${2:?missing value for --dmon-count}; shift 2 ;;
        -h|--help)
            awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
            echo; bench_common_usage
            cat <<'EOF'
  --dmon-delay N     seconds into the run before dmon starts (default 20)
  --dmon-count N     dmon samples, one a second (default 8)
EOF
            exit 0 ;;
        psnr|pcie|formats) sections+=("$1"); shift ;;
        *) bench_die "unknown argument: $1" ;;
    esac
done
[[ ${#sections[@]} -gt 0 ]] || sections=(psnr pcie formats)

bench_setup
sample=0  # these sections watch the GPU with dmon instead
trap 'rm -rf -- "$domain"' EXIT

demo() {  # $1 = label, $2 = output, rest: demo args
    local label=$1 out=$2 rc; shift 2
    bench_domain_reset
    [[ $mode == run ]] && bench_cgroup_name=mxlbench-$label
    bench_run_demo "$results/$label.log" \
        --domain "$domain" --width "$width" --height "$height" --fps "$fps" \
        --input "$input" --output "$out" \
        --bench-seconds "$seconds" --bench-warmup "$warmup" "$@"
    rc=$?
    [[ $rc == 0 ]] || {
        echo "  $label: exit $rc — $results/$label.log"
        bench_summary "$results/$label.log" | tail -3
    }
    return $rc
}

section_psnr() {
    # Not named gpu/cpu: the library's $gpu is the --gpus all switch.
    local out_cpu=$bench_out/pixel-cpu.mp4 out_gpu=$bench_out/pixel-gpu.mp4
    echo "=== psnr: writer conversion on the CPU against the GPU, CPU reader both times"
    demo pixel-cpu "$out_cpu" --gpu-unpack off || return 1
    demo pixel-gpu "$out_gpu" --gpu-unpack off --gpu-scale writer || return 1
    # smptehdbars is static, so the comparison needs no frame alignment: any
    # real difference in the conversion shows up as a PSNR drop.
    bench_in_container_bash "
        for f in '$out_cpu' '$out_gpu'; do
            printf '  %s: ' \"\$f\"
            ffprobe -v error -select_streams v:0 -count_frames \
                -show_entries stream=codec_name,width,height,pix_fmt,nb_read_frames \
                -of csv=p=0 \"\$f\"
        done
        # Both filters print their summary at INFO level, and -nostats plus
        # the \r-to-newline pass keeps it off the progress line.
        for filter in psnr ssim; do
            ffmpeg -hide_banner -nostats -v info -i '$out_cpu' -i '$out_gpu' \
                -lavfi \"[0:v][1:v]\$filter\" -f null - 2>&1 \
                | tr '\r' '\n' | grep '^\[Parsed' | tail -1 | sed 's/^[^]]*] /  /'
        done" 2>&1
}

section_pcie() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo "=== pcie: no nvidia-smi, skipped"; return 0; }
    [[ $seconds -ge $((dmon_delay + dmon_count + 5)) ]] \
        || echo "warning: --seconds $seconds leaves dmon outside the measured window" >&2
    local label
    for label in pcie-gpu-scale-both pcie-gpu-reader; do
        case $label in
            pcie-gpu-scale-both) set -- --gpu-unpack off --gpu-scale both ;;
            pcie-gpu-reader)     set -- --gpu-unpack on ;;
        esac
        echo "=== $label: $*"
        demo "$label" /dev/null "$@" &
        local runner=$!
        sleep "$dmon_delay"
        nvidia-smi dmon -s put -c "$dmon_count"
        wait "$runner"
        bench_summary "$results/$label.log"
        echo
    done
}

section_formats() {
    echo "=== formats: scale_cuda output formats in this build"
    bench_in_container_bash '
        for f in p010le p210le yuv422p10le nv16 yuv420p nv12 p216le; do
            out=$(ffmpeg -hide_banner -loglevel error \
                -init_hw_device cuda=g -filter_hw_device g \
                -f lavfi -i testsrc2=s=320x240:d=0.1 \
                -vf "hwupload,scale_cuda=format=$f,hwdownload,format=$f" \
                -frames:v 1 -f null - 2>&1 | head -1)
            echo "  $f: ${out:-OK}"
        done' 2>&1
}

for section in "${sections[@]}"; do
    "section_$section"
    echo
done
