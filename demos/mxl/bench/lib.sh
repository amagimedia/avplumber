#!/usr/bin/env bash
# Shared plumbing for the demos/mxl measurement scripts: how to reach a
# container, how to hand it a clean MXL domain, how to sample the host while
# the demo runs, and how to digest what it printed. Sourced, not executed.

[[ -n ${bench_lib_sourced:-} ]] && return 0
bench_lib_sourced=1

bench_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
demos_mxl_dir=$(dirname -- "$bench_dir")
demo_in_container=/build/demos/mxl/mxl_demo.py

# Two ways to reach the demo, both ending in the same python3 command line:
#   run    a fresh container per case. The only mode whose cgroup holds
#          nothing but the run, so whole-process CPU means something.
#   exec   an already-running container: the incremental build loop, i.e.
#          whatever /build holds right now, bind-mounted sources included.
mode=${AVP_BENCH_MODE:-run}
container=${AVP_BENCH_CONTAINER:-avpbuild}
image=${AVP_BENCH_IMAGE:-avplumber-mixer:local}
results=${AVP_BENCH_RESULTS:-${TMPDIR:-/tmp}/mxl-bench}
domain=${AVP_MXL_DOMAIN:-/dev/shm/mxlbench}
history_ns=${AVP_MXL_HISTORY_NS:-1000000000}
width=${AVP_WIDTH:-1920}
height=${AVP_HEIGHT:-1080}
fps=${AVP_FPS:-60000/1001}
input=${AVP_INPUT:-}
seconds=${AVP_BENCH_SECONDS:-60}
warmup=${AVP_BENCH_WARMUP:-10}
gpu=${AVP_BENCH_GPU:-1}
sample=${AVP_BENCH_SAMPLE:-1}
mount_demos=${AVP_BENCH_MOUNT_DEMOS:-0}
timeout_slack=${AVP_BENCH_TIMEOUT_SLACK:-180}

bench_die() { echo "${0##*/}: $*" >&2; exit 2; }

# Options every script here takes. Returns how many arguments it consumed, 0
# for "not mine" — callers add their own cases and pass the rest through to
# mxl_demo.py.
bench_common_arg() {
    case ${1:-} in
        --exec)        mode=exec; container=${2:?missing value for --exec}; return 2 ;;
        --image)       mode=run; image=${2:?missing value for --image}; return 2 ;;
        --domain)      domain=${2:?missing value for --domain}; return 2 ;;
        --history-ns)  history_ns=${2:?missing value for --history-ns}; return 2 ;;
        --width)       width=${2:?missing value for --width}; return 2 ;;
        --height)      height=${2:?missing value for --height}; return 2 ;;
        --fps)         fps=${2:?missing value for --fps}; return 2 ;;
        --input)       input=${2:?missing value for --input}; return 2 ;;
        --seconds)     seconds=${2:?missing value for --seconds}; return 2 ;;
        --warmup)      warmup=${2:?missing value for --warmup}; return 2 ;;
        --results)     results=${2:?missing value for --results}; return 2 ;;
        --no-gpu)      gpu=0; return 1 ;;
        --no-sample)   sample=0; return 1 ;;
        --mount-demos) mount_demos=1; return 1 ;;
        *)             return 0 ;;
    esac
}

bench_common_usage() {
    cat <<'EOF'
common options:
  --image IMAGE      fresh container per case (default avplumber-mixer:local)
  --exec CONTAINER   run through `docker exec` in a container that is already up
  --mount-demos      run mode: bind this checkout's demos/mxl over the image's
  --domain PATH      MXL domain directory (default /dev/shm/mxlbench)
  --history-ns N     the domain's history_duration (default 1000000000)
  --width N --height N --fps R --input URI    flow geometry and source
  --seconds N --warmup N                      measured window and its warmup
  --results DIR      logs, samples and outputs (default $TMPDIR/mxl-bench)
  --no-gpu           run mode: leave out --gpus all
  --no-sample        skip the host-side GPU and cgroup CPU samplers
EOF
}

# The default source matches the flow geometry; the cases that publish a
# 1080p source into a smaller flow pass --input themselves.
bench_default_input() {
    [[ -n $input ]] || input="lavfi:smptehdbars=size=${width}x${height}:rate=${fps}"
}

# Derive what the options imply, once the caller has parsed them all.
bench_setup() {
    command -v docker >/dev/null 2>&1 || bench_die "docker not found"
    bench_default_input
    mkdir -p -- "$results"
    if [[ $mode == exec ]]; then
        [[ $(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null) == true ]] \
            || bench_die "container $container is not running (start it, or use --image)"
        # Its own filesystem: the host sees these only through `docker cp`.
        bench_out=/tmp/mxl-bench
        docker exec "$container" mkdir -p "$bench_out"
        bench_cgroup_name=$container
    else
        docker image inspect "$image" >/dev/null 2>&1 \
            || bench_die "no image $image — build demos/mixer/Dockerfile first"
        bench_out=$results/media
        mkdir -p -- "$bench_out"
        bench_cgroup_name=
    fi
}

# A stale flow directory keeps the old geometry and grain indices, so every
# case starts from an empty domain.
bench_domain_reset() {
    local cid name
    for cid in $(docker ps -q); do
        docker inspect -f '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$cid" 2>/dev/null \
            | grep -qx -- "$domain" || continue
        # A container sharing the host's /dev/shm (--ipc=host, which is how
        # the demo runs) picks the recreated directory up. One holding only a
        # bind mount of it would be left publishing into an unlinked
        # directory that no reader can find, so refuse that.
        [[ $domain == /dev/shm/* \
           && $(docker inspect -f '{{.HostConfig.IpcMode}}' "$cid") == host ]] && continue
        name=$(docker inspect -f '{{.Name}}' "$cid")
        bench_die "refusing to reset $domain: container ${name#/} has it bind-mounted"
    done
    rm -rf -- "$domain"
    mkdir -p -- "$domain"
    printf '{"urn:x-mxl:option:history_duration/v1.0": %s}\n' "$history_ns" \
        >"$domain/options.json"
}

# Extra `docker run` arguments a caller needs (mounts, -e); appended last.
bench_extra_run_args=()

bench_run_args() {
    bench_args=( --rm --ipc=host )
    [[ $gpu == 1 ]] && bench_args+=( --gpus all )
    [[ -n $bench_cgroup_name ]] && bench_args+=( --name "$bench_cgroup_name" )
    bench_args+=( -v "$domain:$domain" -v "$results:$results" )
    [[ $mount_demos == 1 ]] && bench_args+=( -v "$demos_mxl_dir:/build/demos/mxl:ro" )
    bench_args+=( "${bench_extra_run_args[@]+"${bench_extra_run_args[@]}"}" )
}

# Run the demo, log to $1, and return its exit status.
bench_run_demo() {
    local log=$1 rc; shift
    if [[ $mode == exec ]]; then
        timeout --signal=KILL "$((seconds + timeout_slack))" \
            docker exec "$container" python3 "$demo_in_container" "$@" >"$log" 2>&1
        rc=$?
        # `timeout` only kills the docker client; the demo keeps running.
        [[ $rc == 124 || $rc == 137 ]] \
            && docker exec "$container" pkill -f mxl_demo.py >/dev/null 2>&1
        return $rc
    fi
    docker rm -f "$bench_cgroup_name" >/dev/null 2>&1
    bench_run_args
    timeout --signal=KILL "$((seconds + timeout_slack))" \
        docker run "${bench_args[@]}" --entrypoint python3 "$image" \
        "$demo_in_container" "$@" >"$log" 2>&1
}

# Run a shell script inside the container: ffprobe, ffmpeg and a backgrounded
# writer all live there. Same mounts as bench_run_demo in run mode.
bench_in_container_bash() {
    if [[ $mode == exec ]]; then
        docker exec "$container" bash -c "$1"
        return $?
    fi
    bench_run_args
    docker run "${bench_args[@]}" --entrypoint bash "$image" -c "$1"
}

bench_gpu_sampler=
bench_cpu_sampler=

# Cumulative cgroup CPU time over a known window is an exact mean core count;
# docker stats' own percentages are sampled too coarsely to trust here.
bench_cpu_stat_path() {
    local cid path
    [[ -n $bench_cgroup_name ]] || return 0
    cid=$(docker inspect -f '{{.Id}}' "$bench_cgroup_name" 2>/dev/null) || return 0
    [[ -n $cid ]] || return 0
    for path in "/sys/fs/cgroup/system.slice/docker-$cid.scope/cpu.stat" \
                "/sys/fs/cgroup/docker/$cid/cpu.stat"; do
        [[ -r $path ]] && { printf '%s\n' "$path"; return 0; }
    done
}

bench_sampler_start() {  # $1 = result prefix
    bench_gpu_sampler= bench_cpu_sampler=
    [[ $sample == 1 ]] || return 0
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,memory.used \
            --format=csv,noheader,nounits -l 1 >"$1.gpu.csv" 2>/dev/null &
        bench_gpu_sampler=$!
    fi
    ( while :; do
          path=$(bench_cpu_stat_path)
          [[ -n $path ]] && awk -v t="$(date +%s.%N)" \
              '/^usage_usec/ {print t "," $2}' "$path"
          sleep 1
      done ) >"$1.cpu.csv" 2>/dev/null &
    bench_cpu_sampler=$!
}

bench_sampler_stop() {
    local pid
    for pid in $bench_gpu_sampler $bench_cpu_sampler; do
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    done
    bench_gpu_sampler= bench_cpu_sampler=
}

bench_sampler_report() {  # $1 = result prefix
    if [[ -s $1.gpu.csv ]]; then
        echo "--- gpu (mean/max)"
        awk -F', *' 'NF==3 {n++; s1+=$1; s2+=$2; s3+=$3
            if($1>m1)m1=$1; if($2>m2)m2=$2; if($3>m3)m3=$3}
            END {if(n) printf "  sm %.1f%%/%d%%  enc %.1f%%/%d%%  gpumem %.0f/%d MiB\n",
                s1/n, m1, s2/n, m2, s3/n, m3}' "$1.gpu.csv"
    fi
    if [[ -s $1.cpu.csv ]]; then
        echo "--- cpu (whole container)"
        awk -F, 'NR==1 {t0=$1; u0=$2} {t1=$1; u1=$2; n++}
            END {if(n>1) printf "  %.2f cores mean over %.0f s\n",
                (u1-u0)/1e6/(t1-t0), t1-t0}' "$1.cpu.csv"
    fi
    return 0
}

# Lines that mean the run was not clean: dropped or late grains, edge
# mistakes, and the usual failure words.
bench_log_patterns=('too late' 'too early' 'not consumed by libavformat'
                    'setting edge more than once' 'EventLoop negative'
                    'Traceback' 'Exception' 'ERROR' 'failed')

# The demo's own summary, minus the per-second rows and their header — those
# stay in the log.
bench_summary() {  # $1 = log
    grep -E '^bench:' -- "$1" | grep -vE '^bench: +[0-9]|^bench: elapsed_s'
    local pat n first=1
    for pat in "${bench_log_patterns[@]}"; do
        n=$(grep -c -i -- "$pat" "$1")
        [[ $n == 0 ]] && continue
        [[ $first == 1 ]] && { echo "--- log counts"; first=0; }
        printf '  %s: %s\n' "$pat" "$n"
    done
    return 0
}

# ffprobe lives in the image, so ask the container: in exec mode the file is
# inside it, and in run mode a throwaway container sees the same mounts.
bench_probe_output() {  # $1 = path as the demo saw it
    [[ -n $1 && $1 != /dev/null ]] || return 0
    echo "--- output"
    bench_in_container_bash "
        [ -s '$1' ] || { echo '  no output file'; exit 0; }
        ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames,avg_frame_rate,width,height,pix_fmt \
            -of default=noprint_wrappers=1 '$1' | tr '\n' ' ' | sed 's/^/  /'
        echo
        ls -l '$1' | awk '{print \"  \" \$5 \" bytes\"}'" 2>&1
}
