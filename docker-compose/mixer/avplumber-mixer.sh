#!/bin/bash
# Graph process, then the Python controller. Arguments are pyplumber.mixer arguments.
# Both stay in this process group: setup stops the show with SIGINT to the group,
# and the EXIT trap reaps avplumber when the controller returns.
avplumber_pid=
stop_avplumber() {
    if [[ -n "${avplumber_pid}" ]] && kill -0 "${avplumber_pid}" 2>/dev/null; then
        kill "${avplumber_pid}" 2>/dev/null || true
        wait "${avplumber_pid}" 2>/dev/null || true
    fi
    avplumber_pid=
}
trap stop_avplumber EXIT

# Ignore SIGINT in the graph process so the controller can still stop groups
# after setup signals this process group. The controller itself is not ignored.
trap '' INT
/usr/local/bin/avplumber --bind 127.0.0.1 --port 22422 &
avplumber_pid=$!
trap - INT

ready=0
for _ in $(seq 1 100); do
    if ! kill -0 "${avplumber_pid}" 2>/dev/null; then
        wait "${avplumber_pid}"
        exit 1
    fi
    if (echo >/dev/tcp/127.0.0.1/22422) >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 0.1
done
if [[ "${ready}" != 1 ]]; then
    echo "avplumber did not accept 127.0.0.1:22422" >&2
    exit 1
fi

python3 -u -m pyplumber.mixer "$@"
exit $?
