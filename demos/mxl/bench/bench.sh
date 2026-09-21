#!/usr/bin/env bash
# One measured run of demos/mxl/mxl_demo.py: fresh MXL domain, host-side GPU
# and CPU sampling, and a digest of what the demo printed.
#
#   bench.sh [options] [mxl_demo.py args...]
#
# Anything this script does not recognise goes to the demo, so its own flags
# work here unchanged and win over the defaults below:
#
#   bench.sh --label gpu-pack --gpu-scale writer --writer-pack gpu
#   bench.sh --exec avpbuild --seconds 30 --reader-tail demux --gpu-unpack off
#
# cases.sh drives this for the groups of cases behind ../README.md's tables.
set -uo pipefail
source "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

label=
output=
extra=()
while [[ $# -gt 0 ]]; do
    bench_common_arg "$@"; consumed=$?
    if [[ $consumed -gt 0 ]]; then shift "$consumed"; continue; fi
    case $1 in
        --label)  label=${2:?missing value for --label}; shift 2 ;;
        --output) output=${2:?missing value for --output}; shift 2 ;;
        -h|--help)
            awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
            echo; bench_common_usage
            cat <<'EOF'
  --label NAME       names the log and sample files (default case-HHMMSS)
  --output PATH      as the demo sees it; /dev/null skips the output probe
EOF
            exit 0 ;;
        --) shift; extra+=("$@"); break ;;
        *)  extra+=("$1"); shift ;;
    esac
done

[[ -n $label ]] || label=case-$(date +%H%M%S)
bench_setup
[[ $mode == run ]] && bench_cgroup_name=mxlbench-$label
[[ -n $output ]] || output=$bench_out/$label.mp4
prefix=$results/$label
log=$prefix.log

bench_domain_reset
[[ $output == /dev/null ]] || bench_in_container_bash "rm -f '$output'" >/dev/null 2>&1

bench_sampler_start "$prefix"
start=$(date +%s.%N)
bench_run_demo "$log" \
    --domain "$domain" --width "$width" --height "$height" --fps "$fps" \
    --input "$input" --output "$output" \
    --bench-seconds "$seconds" --bench-warmup "$warmup" \
    "${extra[@]+"${extra[@]}"}"
rc=$?
end=$(date +%s.%N)
bench_sampler_stop

printf '=== %s (exit %s, wall %.1fs)%s\n' "$label" "$rc" \
    "$(awk "BEGIN{print $end - $start}")" \
    "${extra[*]+ args: ${extra[*]}}"
bench_summary "$log"
bench_sampler_report "$prefix"
bench_probe_output "$output"
echo "--- log: $log"
exit "$rc"
