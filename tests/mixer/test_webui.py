"""The web UI bridge: routing, command building and reconnection."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from itertools import count as counter
from types import SimpleNamespace

import pytest

from demos.mixer.webui import parse_args

from pyplumber.mixer.control import mixer_command
from pyplumber.mixer.gui.web import (CONFIG_SCRIPT, CriticalNice, GpuStats, HostStats, MixerBridge, normalize_preview_base, page,
                   compute_price, serve)
from pyplumber.mixer.gui.telemetry import RecentCounts, _encoder_snapshot, _encoder_totals, delivery_totals


class FakeBridge(MixerBridge):
    """A bridge with the control connection replaced by a scripted stub."""

    def __init__(self, replies=None, fail_take=None, transition=None):
        self.mixer = "mixer"
        self.timeout = 5.0
        self.transition = transition
        self.sent: list[str] = []
        self.replies = replies or {}
        self.fail_take = fail_take
        self.gates: dict[str, threading.Event] = {}   # a command prefix waits for its event
        self._init_sharing()

    def command(self, line: str, timeout: float | None = None):
        self.sent.append(line)
        for prefix, gate in self.gates.items():
            if line.startswith(prefix):
                assert gate.wait(5)
        if self.fail_take and line.startswith(self.fail_take):
            raise RuntimeError("mixer said no")
        for prefix, reply in self.replies.items():
            if line.startswith(prefix):
                return reply
        return None


@pytest.fixture
def client(monkeypatch, http_server):
    monkeypatch.setattr(GpuStats, 'snapshot', lambda _: [])
    def start(bridge, **kwargs):
        server = http_server(bridge, **kwargs)
        return f"http://127.0.0.1:{server.server_port}", server
    return start


def mock_gpu_queries(monkeypatch, query):
    monkeypatch.setattr('pyplumber.mixer.gui.telemetry.subprocess.check_output', query)
    monkeypatch.setattr('pyplumber.mixer.gui.telemetry._encoder_snapshot', lambda counts: _encoder_totals(
        query(['nvidia-smi', 'encodersessions'], timeout=1)))


def test_gpu_samples_are_cached_and_missing_metrics_stay_unknown(monkeypatch):
    now = [0]
    calls = []
    monkeypatch.setattr('pyplumber.mixer.gui.telemetry.time.monotonic', lambda: now[0])
    def query(*args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] == 1
        if args[0] == ['nvidia-smi', 'encodersessions']:
            return '# GPU Session Process Codec H V Average Average\n# Idx Id Id Type Res Res FPS Latency(us)\n' \
                   '0 10 200 H.264 1920 1080 24.5 17000\n0 11 200 H.265 1280 720 25 12000\n' \
                   '1 12 201 AV1 3840 2160 12.5 20000\n'
        return '0, 93, [N/A], 12, 14336, 15360, 53.95, 72.00, 2\n1, 0, 99, [N/A], 15104, 24576, [N/A], [Not Supported], 1\n'
    mock_gpu_queries(monkeypatch, query)
    stats = GpuStats()
    first = stats.snapshot()
    assert first[0] == dict(index=0, gpu=93, decoder=None, encoder=12, memory_used_mib=14336, memory_total_mib=15360,
                          power_draw_w=53.95, power_limit_w=72.0, encoder_sessions=2,
                          encoder_fps=49.5, encoder_mpix_s=pytest.approx(1920 * 1080 * 24.5 / 1e6 + 1280 * 720 * 25 / 1e6))
    assert first[1]['encoder'] is None
    assert first[1]['power_draw_w'] is None and first[1]['power_limit_w'] is None
    assert first[1]['index'] == 1 and first[1]['gpu'] == 0
    assert first[1]['encoder_sessions'] == 1 and first[1]['encoder_fps'] == 12.5
    assert first[1]['encoder_mpix_s'] == pytest.approx(3840 * 2160 * 12.5 / 1e6)
    now[0] = .5
    assert stats.snapshot() == first and len(calls) == 2
    now[0] = 1
    stats.lock.acquire()
    try:
        assert stats.snapshot() == first and len(calls) == 2
    finally:
        stats.lock.release()
    stats.snapshot()
    assert len(calls) == 4
    assert 'power.draw,enforced.power.limit' in calls[0][0][1]
    assert 'encoder.stats.sessionCount' in calls[0][0][1]


@pytest.mark.parametrize('raw', ['NaN', 'inf', '-1', '[N/A]'])
def test_invalid_power_readings_do_not_hide_utilization(monkeypatch, raw):
    mock_gpu_queries(monkeypatch, lambda *a, **kw: f'0, 60, 70, 40, 100, 200, {raw}, {raw}, 1\n')
    sample = GpuStats().snapshot()[0]
    assert sample['gpu'] == 60
    assert sample['power_draw_w'] is None and sample['power_limit_w'] is None


@pytest.mark.parametrize('sessions', ['', '# GPU Session Process Codec H V Average Average\n'])
def test_encoder_zero_requires_a_driver_session_counter(monkeypatch, sessions):
    def query(command, **_):
        if command[1] == 'encodersessions':
            return sessions
        return '0, 60, 70, 40, 100, 200, 50, 72, 0\n1, 10, 20, 30, 100, 200, 50, 72, [N/A]\n'
    mock_gpu_queries(monkeypatch, query)
    first, second = GpuStats().snapshot()
    assert (first['encoder_sessions'], first['encoder_fps'], first['encoder_mpix_s']) == (0, 0, 0)
    assert all(second[key] is None for key in ('encoder_sessions', 'encoder_fps', 'encoder_mpix_s'))


@pytest.mark.parametrize('bad', [
    '0 12 200 H.264 1920 1080', '0 - - - - - - -', '0 12 200 H.264 0 1080 25 1',
    '0 12 200 H.264 1920 1080 NaN 1', '0 12 200 H.264 1920 1080 -1 1',
    '0 12 200 H.264 1920 1080 1e308 1',
])
def test_incomplete_encoder_rows_do_not_publish_partial_totals(monkeypatch, bad):
    def query(command, **_):
        if command[1] == 'encodersessions':
            return '0 10 200 H.264 1920 1080 25 1\n' + bad + '\n1 11 200 H.265 1280 720 0 0\n'
        return '0, 60, 70, 40, 100, 200, 50, 72, 2\n1, 10, 20, 30, 100, 200, 40, 72, 1\n'
    mock_gpu_queries(monkeypatch, query)
    first, second = GpuStats().snapshot()
    assert all(first[key] is None for key in ('encoder_sessions', 'encoder_fps', 'encoder_mpix_s'))
    assert first['gpu'] == 60 and first['power_draw_w'] == 50
    assert (second['encoder_sessions'], second['encoder_fps'], second['encoder_mpix_s']) == (1, 0, 0)


@pytest.mark.parametrize('result', [
    subprocess.TimeoutExpired('nvidia-smi', 1), subprocess.CalledProcessError(1, 'nvidia-smi'),
    'Encoder sessions are not supported.', '', '0 10 200 H.264 1920 1080 25 1\n',
])
def test_missing_or_changing_encoder_sample_preserves_gpu_metrics(monkeypatch, result):
    def query(command, **_):
        if command[1] == 'encodersessions':
            if isinstance(result, Exception):
                raise result
            return result
        return '0, 60, 70, 40, 100, 200, 50.25, 72, 2\n'
    mock_gpu_queries(monkeypatch, query)
    sample = GpuStats().snapshot()[0]
    assert all(sample[key] is None for key in ('encoder_sessions', 'encoder_fps', 'encoder_mpix_s'))
    assert sample['encoder'] == 40 and sample['power_draw_w'] == 50.25 and sample['memory_used_mib'] == 100


@pytest.mark.parametrize('complete', [True, False])
def test_encoder_reader_stops_and_reaps_a_continuously_running_command(monkeypatch, complete):
    real_popen, children = subprocess.Popen, []
    row = '0 10 200 H.264 1920 1080 25 1' + ('\n' if complete else '')
    def start(command, **kwargs):
        assert command == ['nvidia-smi', 'encodersessions']
        child = real_popen([sys.executable, '-c',
                           f'import sys,time;sys.stdout.write({row!r});sys.stdout.flush();time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr('pyplumber.mixer.gui.telemetry.subprocess.Popen', start)
    at = time.monotonic()
    sample = _encoder_snapshot({0: 1})
    elapsed = time.monotonic() - at
    if complete:
        assert sample[0] == (1, 25, pytest.approx(1920 * 1080 * 25 / 1e6))
        assert elapsed < .8, 'the first full table must not wait for the timeout'
    else:
        assert sample == {}, 'a partial last line is not a complete snapshot'
        assert elapsed < 2
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed


def test_compute_price_config_is_optional_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv('MIXER_COMPUTE_PRICE', '')
    assert compute_price(parse_args([]).compute_price) is None
    path = tmp_path / 'price.json'
    value = dict(hourly_usd=1.2, label='Google Cloud VM + GPU, on demand', as_of='2026-10-03',
                 source_url='https://cloud.google.com/products/compute/pricing/accelerator-optimized')
    path.write_text(json.dumps(value))
    monkeypatch.setenv('MIXER_COMPUTE_PRICE', str(path))
    assert compute_price(parse_args([]).compute_price) == value
    for rate in [0, -1, True, '1.2', float('nan'), float('inf')]:
        path.write_text(json.dumps(dict(value, hourly_usd=rate)))
        with pytest.raises(ValueError, match='hourly_usd'):
            compute_price(path)
    path.write_text(json.dumps(dict(value, source_url='javascript:alert(1)')))
    with pytest.raises(ValueError, match='HTTPS'):
        compute_price(path)


@pytest.mark.parametrize('error', [FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 1)])
def test_gpu_query_failure_clears_old_sample_and_is_cached(monkeypatch, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise error
    monkeypatch.setattr('pyplumber.mixer.gui.telemetry.subprocess.check_output', fail)
    stats = GpuStats()
    stats.values = [{'gpu': 90}]
    assert stats.snapshot() == []
    assert stats.snapshot() == [] and len(calls) == 1


def fake_proc(root, load1, cpu, threads=None, pid=4242, process_ticks=None, started=5):
    """A /proc with one host of two vCPUs and, given {tid: (comm, utime, stime)}, the mixer's threads."""
    (root / "loadavg").write_text(f"{load1} 20.1 18.7 3/1234 56789\n")
    (root / "stat").write_text(f"cpu {' '.join(map(str, cpu))} 0 0\ncpu0 1 0 0 1 0 0 0 0 0 0\ncpu1 1 0 0 1 0 0 0 0 0 0\n"
                               "intr 5 1 2\nctxt 9\n")
    def stat(tid, comm, utime, stime):
        return f"{tid} ({comm}) S 1 1 1 0 -1 4194560 0 0 0 0 {utime} {stime} 0 0 20 0 1 0 {started} 0 0\n"
    for tid, (comm, utime, stime) in (threads or {}).items():
        task = root / str(pid) / "task" / str(tid)
        task.mkdir(parents=True, exist_ok=True)
        task.joinpath("stat").write_text(stat(tid, comm, utime, stime))
    if threads is not None:
        process = root / str(pid)
        process.mkdir(exist_ok=True)
        used = sum(utime + stime for _, utime, stime in threads.values()) if process_ticks is None else process_ticks
        process.joinpath("stat").write_text(stat(pid, "mixer (process)", used, 0))


def test_host_stats_sample_load_cpu_and_the_busiest_mixer_thread(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("pyplumber.mixer.gui.telemetry.time.monotonic", lambda: now[0])
    hz = os.sysconf("SC_CLK_TCK")
    pid = [4242]
    stats = HostStats(lambda: pid[0], proc=tmp_path)
    #                       user nice system idle iowait irq softirq steal
    fake_proc(tmp_path, 3.5, [100, 0, 50, 800, 20, 0, 10, 0], {1: ("mixer.py", 500, 100), 2: ("mixer_comp_a", 10, 0)})
    stats.sample()
    assert stats.values is None   # the first sample has nothing to diff against
    now[0] += 2.0   # user+nice+system+irq+softirq+steal grew 52 of 100 ticks; thread 2 ran 76 ticks over two seconds
    fake_proc(tmp_path, 21.4, [140, 2, 58, 840, 28, 1, 10, 1], {1: ("mixer.py", 505, 105), 2: ("mixer_comp_a", 84, 2),
                                                                3: ("mixer (new)", 30, 1)})
    stats.sample()
    assert stats.values == dict(load1=21.4, vcpus=2, cpu_pct=52, mixer_cpu_pct=round(117 * 100 / hz / 2, 1),
                               thread_pct=round(76 * 100 / hz / 2), thread_name="mixer_comp_a")
    # Without a mixer, or before its first diff after a restart, the thread fields stay unknown.
    pid[0] = None
    now[0] += 1.0
    stats.sample()
    assert stats.values["thread_pct"] is None and stats.values["thread_name"] is None
    assert stats.values["mixer_cpu_pct"] is None
    pid[0] = 4243
    now[0] += 1.0
    fake_proc(tmp_path, 1.0, [150, 2, 58, 940, 28, 1, 10, 1], {1: ("mixer.py", 900, 100)}, pid=4243)
    stats.sample()
    assert stats.values["cpu_pct"] == 9 and stats.values["thread_pct"] is None
    assert stats.values["mixer_cpu_pct"] is None
    now[0] += 1.0
    fake_proc(tmp_path, 1.0, [150, 2, 58, 1040, 28, 1, 10, 1], {1: ("mixer.py", 920, 100)}, pid=4243)
    stats.sample()
    assert stats.values["cpu_pct"] == 0 and stats.values["thread_pct"] == round(20 * 100 / hz) and stats.values["thread_name"] == "mixer.py"
    assert stats.values["mixer_cpu_pct"] == round(20 * 100 / hz, 1)


def test_mixer_cpu_counts_exited_threads_and_detects_pid_reuse(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("pyplumber.mixer.gui.telemetry.time.monotonic", lambda: now[0])
    hz = os.sysconf("SC_CLK_TCK")
    stats = HostStats(lambda: 4242, proc=tmp_path)
    def sample(ticks, started=5):
        fake_proc(tmp_path, 1, [100, 0, 0, 900, 0, 0, 0, 0],
                  {1: ("mixer.py", 10, 0)}, process_ticks=ticks, started=started)
        stats.sample()
        now[0] += 2
    sample(1000)
    sample(1000 + hz * 7 + 1)
    assert stats.values['mixer_cpu_pct'] == round(350 + 50 / hz, 1)
    assert stats.values['thread_pct'] == 0   # CPU of exited threads remains in the process counters.
    sample(5000, started=6)
    assert stats.values['mixer_cpu_pct'] is None and stats.values['thread_pct'] is None
    sample(5000, started=6)
    assert stats.values['mixer_cpu_pct'] == 0
    (tmp_path / '4242' / 'stat').write_text('truncated')
    stats.sample()
    assert stats.values['mixer_cpu_pct'] is None and stats.values['cpu_pct'] is None


def test_host_stats_thread_hides_the_meters_during_an_error_and_logs_it_once(tmp_path, monkeypatch, caplog):
    fake_proc(tmp_path, 1.0, [1, 0, 1, 1, 0, 0, 0, 0], {1: ("mixer.py", 1, 0)})
    # A setup change clears setup.process on another thread: two samples in a row see None where a process was.
    processes = iter([SimpleNamespace(pid=4242), None, None, SimpleNamespace(pid=4242)])
    stats = HostStats(lambda: next(processes).pid, proc=tmp_path)
    clock, published = counter(10.0), []

    class Stop(Exception):
        pass

    def sleep(_seconds):   # the loop only ends with the process: stop it after the fourth sample
        published.append(stats.values)
        if len(published) == 4:
            raise Stop
    monkeypatch.setattr("pyplumber.mixer.gui.telemetry.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("pyplumber.mixer.gui.telemetry.time.sleep", sleep)
    with caplog.at_level(logging.WARNING, logger="webui"), pytest.raises(Stop):
        stats._run()
    assert published[:3] == [None, None, None] and published[3]["thread_name"] == "mixer.py"   # the meters return with the next good sample
    assert [r.getMessage() for r in caplog.records] == ["Host stats unavailable: AttributeError: 'NoneType' object has no attribute 'pid'"]


def test_host_stats_survive_a_missing_proc_and_a_vanished_mixer(tmp_path):
    stats = HostStats(lambda: 4242, proc=tmp_path)
    with pytest.raises(OSError):   # the sampling thread then publishes no values
        stats.sample()
    fake_proc(tmp_path, 1.0, [1, 0, 1, 1, 0, 0, 0, 0])   # a pid without a /proc entry: the mixer just exited
    stats.sample()
    fake_proc(tmp_path, 1.0, [2, 0, 2, 2, 0, 0, 0, 0])
    stats.sample()
    assert stats.values["cpu_pct"] == 67 and stats.values["thread_pct"] is None


def fake_threads(root, pid, comms):
    """/proc/<pid>/task/<tid>/comm for {tid: thread name}; the kernel keeps 15 characters of a name."""
    for tid, comm in comms.items():
        task = root / str(pid) / "task" / str(tid)
        task.mkdir(parents=True, exist_ok=True)
        task.joinpath("comm").write_text(comm[:15] + "\n")


def record_setpriority(monkeypatch, fail=None):
    """Record (tid, nice) of every os.setpriority call; `fail` maps a tid to the error raised for it."""
    calls = []

    def setpriority(which, who, nice):
        assert which == os.PRIO_PROCESS
        if (fail or {}).get(who):
            raise fail[who]
        calls.append((who, nice))
    monkeypatch.setattr("pyplumber.mixer.gui.priority.os.setpriority", setpriority)
    return calls


# The names avplumber, the CUDA driver and FFmpeg give their threads; the mixer's Python main
# thread, an NVDEC input and the aux compositor are not deadline-critical.
MIXER_THREADS = {1: "mixer.py", 2: "mixer_comp_a", 3: "mixer_comp_b", 4: "mixer_snapshot_a", 5: "mixer_snapshot_output",
                 6: "EventLoop", 7: "cuda-EvtHandlr", 8: "janus_encoder", 9: "janus_hdr_encoder", 10: "aux_mv2_encoder",
                 11: "input_12_decode", 12: "aux_mv2_comp", 13: "mixer_otm_scene_a", 14: "stats sender", 15: "dsk_comp"}
CRITICAL_TIDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 15}


def test_critical_threads_get_the_nice_level_each_period_and_are_logged_once(tmp_path, monkeypatch, caplog):
    calls = record_setpriority(monkeypatch)
    pid = [4242]
    nicer = CriticalNice(lambda: pid[0], 10, proc=tmp_path)
    fake_threads(tmp_path, 4242, MIXER_THREADS)
    with caplog.at_level(logging.INFO, logger="webui"):
        nicer.apply()
        assert sorted(calls) == [(tid, -10) for tid in sorted(CRITICAL_TIDS)]
        nicer.apply()   # thread ids are reused: every match is set again, nothing new is logged
        assert len(calls) == 2 * len(CRITICAL_TIDS)
        fake_threads(tmp_path, 4242, {16: "udp-tx"})   # the output started later
        nicer.apply()
        assert sorted(calls[2 * len(CRITICAL_TIDS):]) == [(tid, -10) for tid in sorted(CRITICAL_TIDS | {16})]
    assert [r.getMessage() for r in caplog.records] == [
        "Mixer 4242: nice -10 on 10 threads: EventLoop (6), aux_mv2_encoder (10), cuda-EvtHandlr (7), dsk_comp (15), "
        "janus_encoder (8), janus_hdr_encod (9), mixer_comp_a (2), mixer_comp_b (3), mixer_snapshot_ (4), "
        "mixer_snapshot_ (5)",
        "Mixer 4242: nice -10 on 1 thread: udp-tx (16)"]
    # A restarted mixer is logged in full again; while no mixer runs nothing is touched.
    caplog.clear()
    pid[0] = None
    nicer.apply()
    assert len(calls) == 3 * len(CRITICAL_TIDS) + 1 and not caplog.records
    pid[0] = 4243
    fake_threads(tmp_path, 4243, {2: "mixer_comp_a", 6: "EventLoop"})
    with caplog.at_level(logging.INFO, logger="webui"):
        nicer.apply()
    assert sorted(calls[-2:]) == [(2, -10), (6, -10)]
    assert [r.getMessage() for r in caplog.records] == ["Mixer 4243: nice -10 on 2 threads: EventLoop (6), mixer_comp_a (2)"]


def test_critical_nice_stops_after_eperm_and_tolerates_vanished_threads(tmp_path, monkeypatch, caplog):
    fake_threads(tmp_path, 4242, {2: "mixer_comp_a", 3: "mixer_comp_b", 6: "EventLoop"})
    calls = record_setpriority(monkeypatch, fail={3: ProcessLookupError()})
    nicer = CriticalNice(lambda: 4242, 5, proc=tmp_path)
    with caplog.at_level(logging.INFO, logger="webui"):
        nicer.apply()
    assert sorted(calls) == [(2, -5), (6, -5)]   # thread 3 exited after the listing
    assert [r.getMessage() for r in caplog.records] == ["Mixer 4242: nice -5 on 2 threads: EventLoop (6), mixer_comp_a (2)"]
    calls = record_setpriority(monkeypatch, fail={2: PermissionError(1, "Operation not permitted")})
    fake_threads(tmp_path / "unprivileged", 4242, {2: "mixer_comp_a"})
    nicer = CriticalNice(lambda: 4242, 5, proc=tmp_path / "unprivileged")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="webui"):
        nicer.apply()
        nicer.apply()   # no retry: the container will not gain CAP_SYS_NICE while it runs
    assert not nicer.enabled and calls == []
    assert [r.getMessage() for r in caplog.records] == [
        "Cannot set nice -5 on mixer thread mixer_comp_a (2): the container needs CAP_SYS_NICE; critical-thread priority is off"]
    nicer = CriticalNice(lambda: 9999, 5, proc=tmp_path)   # a pid without a /proc entry: the mixer just exited
    nicer.apply()
    assert calls == [] and nicer.enabled


MANAGE = ["--manage-setup", "--instance-type", "tesla_t4"]


def test_critical_nice_is_off_by_default_and_bounded(monkeypatch):
    monkeypatch.delenv("MIXER_CRITICAL_NICE", raising=False)
    assert parse_args([]).critical_nice == 0
    assert parse_args([*MANAGE, "--critical-nice", "10"]).critical_nice == 10
    monkeypatch.setenv("MIXER_CRITICAL_NICE", "7")
    assert parse_args(MANAGE).critical_nice == 7
    assert parse_args(["--critical-nice", "0", "--mixer-args", "--janus-output"]).mixer_args == ["--janus-output"]
    # The mixer pid comes from the managed process, so without --manage-setup the option would be inert.
    for bad in (["--critical-nice", "10"], [], [*MANAGE, "--critical-nice", "21"],
                [*MANAGE, "--critical-nice", "-1"], [*MANAGE, "--critical-nice", "high"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
    monkeypatch.setenv("MIXER_CRITICAL_NICE", "lots")
    with pytest.raises(SystemExit):
        parse_args(MANAGE)


def test_a_managed_setup_names_its_instance_type(capsys):
    assert parse_args(MANAGE).instance_type == "tesla_t4"
    with pytest.raises(SystemExit):
        parse_args(["--manage-setup"])
    assert "--manage-setup needs --instance-type" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse_args(["--manage-setup", "--instance-type", "tesla_l4"])   # no measured profile


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as response:
        return response.status, json.loads(response.read() or b"{}")


def post(url, payload):
    request = urllib.request.Request(url + "/api/command", method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_state_is_one_round_trip_of_status_scenes_and_settings(client):
    bridge = FakeBridge({
        "mixer.status": '{"pgm_scene":"a","pvw_scene":"b","transition":"idle"}',
        "mixer.scenes": '["a","b"]',
        "mixer.settings": '{"direct":true,"fade_seconds":0.8}',
    })
    url, _ = client(bridge)
    status, body = get(url, "/api/state")

    assert status == 200
    assert body["status"]["pgm_scene"] == "a"
    assert body["scenes"] == ["a", "b"]
    assert body["settings"]["direct"] is True
    assert body["host"] is None   # no host sampler: the page hides its meters
    assert bridge.sent == ["mixer.status mixer", "mixer.scenes mixer", "mixer.settings mixer"]


def test_state_carries_the_host_sample(client):
    bridge = FakeBridge(STATE_REPLIES)
    host = HostStats()
    host.values = dict(load1=21.4, vcpus=16, cpu_pct=52, thread_pct=38, thread_name="mixer_comp_a")
    url, _ = client(bridge, host_stats=host)
    assert get(url, "/api/state")[1]["host"] == host.values


def test_delivery_counts_cover_the_last_ten_minutes_not_the_time_since_the_start():
    totals, now = {"missed_deadlines": 59, "output_drops": 3}, [1000.0]
    recent = RecentCounts(lambda: dict(totals), clock=lambda: now[0])
    assert recent.values is None                      # nothing sampled: the page keeps its totals
    recent.sample()
    # What was counted before the first sample is of unknown age and is left out.
    assert recent.values == {"window_s": 600, "covered_s": 0, "missed_deadlines": 0, "output_drops": 0}
    now[0], totals["missed_deadlines"] = 1300.0, 64
    recent.sample()
    assert recent.values == {"window_s": 600, "covered_s": 300, "missed_deadlines": 5, "output_drops": 0}
    now[0] = 1700.0
    totals.update(missed_deadlines=66, output_drops=4)
    recent.sample()
    # The five misses of 1300 are now the window's start; only what came after it counts.
    assert recent.values == {"window_s": 600, "covered_s": 400, "missed_deadlines": 2, "output_drops": 1}
    now[0] = 2400.0
    recent.sample()
    assert recent.values["missed_deadlines"] == 0 and recent.values["output_drops"] == 0   # all older than ten minutes


def test_a_restarted_mixer_counts_from_zero_and_all_of_it_is_recent():
    totals, now = {"missed_deadlines": 120, "output_drops": 9}, [50.0]
    recent = RecentCounts(lambda: None if totals is None else dict(totals), clock=lambda: now[0])
    recent.sample()
    now[0], totals = 80.0, {"missed_deadlines": 48, "output_drops": 0}     # fewer than before: it started again
    recent.sample()
    assert recent.values == {"window_s": 600, "covered_s": 0, "missed_deadlines": 48, "output_drops": 0}
    now[0], totals = 90.0, None                                            # no mixer: the last values stand
    recent.sample()
    assert recent.values["missed_deadlines"] == 48
    now[0], totals = 400.0, {"missed_deadlines": 50, "output_drops": 0}
    recent.sample()
    assert recent.values["missed_deadlines"] == 50 and recent.values["covered_s"] == 320   # the start is still inside the window
    now[0], totals = 800.0, {"missed_deadlines": 51, "output_drops": 0}
    recent.sample()
    assert recent.values["missed_deadlines"] == 1 and recent.values["covered_s"] == 400    # the start's misses have aged out


def test_delivery_totals_are_the_pages_sums_or_nothing_while_a_bus_is_missing():
    state = {"status": {"playout": {"A": {"missed_deadlines": 1}, "B": {"missed_deadlines": 0}}},
             "settings": {"aux_buses": ["aux1", "aux2"]},
             "aux_buses": [{"playout": {"missed_deadlines": 2}, "output_drops": 1},
                           {"playout": {"missed_deadlines": 0}, "output_drops": 4}]}
    assert delivery_totals(state) == {"missed_deadlines": 3, "output_drops": 5}
    assert delivery_totals({**state, "aux_buses": state["aux_buses"][:1]}) is None       # one bus has not reported
    assert delivery_totals({**state, "status": {}}) is None
    assert delivery_totals({"status": state["status"], "settings": {}}) == {"missed_deadlines": 1, "output_drops": 0}


def test_state_carries_the_recent_delivery_counts(client):
    recent = RecentCounts(lambda: {"missed_deadlines": 7, "output_drops": 0})
    url, _ = client(FakeBridge(STATE_REPLIES), recent=recent)
    assert get(url, "/api/state")[1]["recent"] is None            # no sample yet
    recent.sample()
    assert get(url, "/api/state")[1]["recent"] == {"window_s": 600, "covered_s": 0, "missed_deadlines": 0, "output_drops": 0}


STATE_REPLIES = {"mixer.status": '{"pgm_scene":"a"}', "mixer.scenes": '["a","b"]', "mixer.settings": "{}"}
ONE_POLL = ["mixer.status mixer", "mixer.scenes mixer", "mixer.settings mixer"]


def test_polls_share_one_state_until_it_expires_or_a_take_lands(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("pyplumber.mixer.gui.telemetry.time.monotonic", lambda: now[0])
    bridge = FakeBridge(STATE_REPLIES)
    first = bridge.state()
    first["gpus"] = []   # a caller's own keys stay out of the shared reply
    assert "gpus" not in bridge.state()
    assert bridge.sent == ONE_POLL
    now[0] += 0.2
    bridge.state()
    assert bridge.sent == ONE_POLL * 2
    bridge.take({"command": "cut", "scene": "b"})
    bridge.state()
    assert bridge.sent == ONE_POLL * 2 + ['mixer.cut {"scene":"b","mixer":"mixer"}'] + ONE_POLL


def test_concurrent_polls_cost_one_set_of_commands():
    bridge = FakeBridge(STATE_REPLIES)
    bridge.gates["mixer.status"] = gate = threading.Event()
    polls = [threading.Thread(target=bridge.state) for _ in range(3)]
    for poll in polls:
        poll.start()
    gate.set()
    for poll in polls:
        poll.join(5)
    assert bridge.sent == ONE_POLL


def test_a_burst_of_takes_collapses_to_its_newest_in_order():
    bridge = FakeBridge()
    bridge.gates["mixer.cut"] = gate = threading.Event()
    sent = {}

    def take(scene):
        sent[scene] = bridge.take({"command": "cut", "scene": scene})
    takes = []
    for count, scene in enumerate("abcd", 1):
        takes.append(threading.Thread(target=take, args=(scene,)))
        takes[-1].start()
        while bridge._take_arrivals < count:   # each take arrives after the previous one
            time.sleep(0.001)
    gate.set()   # the mixer answers the first take
    for thread in takes:
        thread.join(5)
    assert bridge.sent == ['mixer.cut {"scene":"a","mixer":"mixer"}', 'mixer.cut {"scene":"d","mixer":"mixer"}']
    assert sent == {"a": True, "b": False, "c": False, "d": True}


def test_a_preview_never_supersedes_a_waiting_program_take():
    bridge = FakeBridge()
    bridge.gates["mixer.preview"] = gate = threading.Event()
    requests = [{"command": "preview", "scene": "b"}, {"command": "cut", "scene": "b"},
                {"command": "preview", "scene": "c"}, {"command": "interrupt"}]
    sent = [None] * len(requests)

    def take(i):
        sent[i] = bridge.take(requests[i])
    takes = []
    for i in range(len(requests)):
        takes.append(threading.Thread(target=take, args=(i,)))
        takes[-1].start()
        while bridge._take_arrivals < i + 1:
            time.sleep(0.001)
    gate.set()   # the mixer answers the first preview
    for thread in takes:
        thread.join(5)
    assert bridge.sent == [mixer_command(r.pop("command"), "mixer", **r) for r in requests]
    assert sent == [True] * len(requests)


def test_status_endpoint_sends_only_mixer_status(client):
    bridge = FakeBridge(STATE_REPLIES)
    url, _ = client(bridge)
    assert get(url, "/api/status") == (200, {"pgm_scene": "a"})
    assert bridge.sent == ["mixer.status mixer"]


def test_state_survives_a_mixer_without_settings(client):
    bridge = FakeBridge({"mixer.status": "{}", "mixer.scenes": "[]"}, fail_take="mixer.settings")
    url, _ = client(bridge)
    assert get(url, "/api/state")[1]["settings"] == {}


@pytest.mark.parametrize("override, expected", [(None, "fade"), ("cut", "cut")])
def test_transition_override_only_changes_page_settings(client, override, expected):
    bridge = FakeBridge({"mixer.status": "{}", "mixer.scenes": "[]",
                         "mixer.settings": '{"transition":"fade","fade_seconds":0.8}'},
                        transition=override)
    url, _ = client(bridge)
    settings = get(url, "/api/state")[1]["settings"]
    assert settings == {"transition": expected, "fade_seconds": 0.8}
    assert bridge.sent == ["mixer.status mixer", "mixer.scenes mixer", "mixer.settings mixer"]


def test_takes_reach_the_mixer_as_control_commands(client):
    bridge = FakeBridge()
    url, _ = client(bridge)

    assert post(url, {"command": "preview", "scene": "grid_4_page_0"})[0] == 200
    assert post(url, {"command": "cut", "scene": "grid_4_page_0"})[0] == 200
    assert post(url, {"command": "fade", "scene": "two_up", "duration_sec": 0.8})[0] == 200
    assert post(url, {"command": "fade", "scene": "two_up", "duration_sec": 0.8, "curve": "ease-in"})[0] == 200
    assert post(url, {"command": "fade", "scene": "two_up", "duration_sec": 0.8, "color": "#ffffff"})[0] == 200
    assert post(url, {"command": "wipe", "scene": "two_up", "wipe_file": "/media/w.mov"})[0] == 200
    assert post(url, {"command": "interrupt"})[0] == 200

    assert bridge.sent == [
        'mixer.preview {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.cut {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"curve":"ease-in","mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"color":"#ffffff","mixer":"mixer"}',
        'mixer.wipe {"scene":"two_up","wipe_file":"/media/w.mov","mixer":"mixer"}',
        'mixer.interrupt {"mixer":"mixer"}',
    ]


def test_key_fades_reach_the_mixer_unchanged(client):
    bridge = FakeBridge({"mixer.dsk": "[]"})
    url, _ = client(bridge)
    assert post(url, {"command": "dsk", "key": "bug", "on": True, "fade_seconds": 0.5, "curve": "ease-out"})[0] == 200
    assert bridge.sent == ['mixer.dsk {"key": "bug", "on": true, "fade_seconds": 0.5, "curve": "ease-out"}']


def test_aux_requests_reach_the_mixer_and_setup_keeps_the_bus(monkeypatch):
    monkeypatch.setattr(GpuStats, 'snapshot', lambda _: [])
    layout = {"preset": "source_pages", "page": 3}
    bridge = FakeBridge({"mixer.aux_page": json.dumps({"layout": layout, "scenes": [], "page": 3}),
                         "mixer.aux_layout": json.dumps({"layout": layout, "scenes": []}),
                         "mixer.aux ": '{"error": "Assignments changed; refresh before editing", "conflict": true}'})
    remembered = []
    setup = SimpleNamespace(remember_aux=lambda bus, fields: remembered.append((bus, fields)),
                            status=lambda: {"revision": 1})
    server = serve(bridge, "127.0.0.1", 0, setup=setup)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    assert post(url, {"command": "aux_page", "bus": "mv2", "step": 1})[0] == 200
    assert post(url, {"command": "aux_layout", "bus": "mv2", "layout": {"preset": "source_pages"}})[0] == 200
    assert post(url, {"command": "aux", "bus": "mv2", "scenes": []})[0] == 400   # a conflict changes nothing
    assert bridge.sent == ['mixer.aux_page {"bus": "mv2", "step": 1}',
                           'mixer.aux_layout {"bus": "mv2", "layout": {"preset": "source_pages"}}',
                           'mixer.aux {"bus": "mv2", "scenes": []}']
    assert [bus for bus, _ in remembered] == ["mv2", "mv2"] and remembered[0][1]["layout"] == layout
    server.shutdown()


@pytest.mark.parametrize("payload, expected", [
    ({"command": "reboot", "scene": "a"}, "command must be one of"),
    ({"command": "cut"}, "needs a scene"),
    ({"scene": "a"}, "command must be one of"),
])
def test_bad_requests_are_rejected_without_touching_the_mixer(client, payload, expected):
    bridge = FakeBridge()
    url, _ = client(bridge)
    status, body = post(url, payload)

    assert status == 400 and expected in body["error"]
    assert bridge.sent == []


def test_a_failing_mixer_is_reported_not_swallowed(client):
    bridge = FakeBridge(fail_take="mixer.cut")
    url, _ = client(bridge)
    status, body = post(url, {"command": "cut", "scene": "a"})

    assert status == 502 and "mixer said no" in body["error"]


def get_page(url, path="/", content_type="text/html"):
    with urllib.request.urlopen(url + path, timeout=5) as response:
        assert response.headers["Content-Type"].startswith(content_type)
        return response.read().decode()


def test_the_pages_and_only_the_pages_are_served(client):
    url, _ = client(FakeBridge({"mixer.status": "{}", "mixer.scenes": "[]"}))
    html = get_page(url)
    assert "AVPlumber mixer" in html and "/api/state" in html and 'href="/wall"' in html
    # The wall shows every output the mixer page lists, from the same script.
    for path in ("/wall", "/wall/"):
        wall = get_page(url, path)
        assert "AVPlumber outputs" in wall and "/api/state" in wall and '<script src="/outputs.js">' in wall
    assert '<script src="/outputs.js">' in html
    assert "function programOptions" in get_page(url, "/outputs.js", "text/javascript")
    try:
        urllib.request.urlopen(url + "/nope", timeout=5)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_the_page_carries_where_the_player_is(client):
    # Without --preview-base the page keeps its own defaults: the player on port 8080 of its host.
    url, _ = client(FakeBridge())
    assert CONFIG_SCRIPT % "{}" in get_page(url) and CONFIG_SCRIPT % "{}" in get_page(url, "/wall")
    # Behind a reverse proxy every player loads from a path of the page's own origin; the page carries it
    # with the slash the player resolves its files against, whichever way the server was given it.
    for preview_base in ("/preview/", "/preview"):
        url, _ = client(FakeBridge(), preview_base=preview_base)
        assert CONFIG_SCRIPT % '{"preview_base": "/preview/"}' in get_page(url)
        assert CONFIG_SCRIPT % '{"preview_base": "/preview/"}' in get_page(url, "/wall")


def test_no_player_address_can_end_the_config_script():
    base = "/</script><script>alert(1)</script>/"
    html = page({"preview_base": base}).decode()
    assert html.count("</script>") == page({}).decode().count("</script>")
    opening = CONFIG_SCRIPT.partition("%s")[0]
    start = html.index(opening) + len(opening)
    script = html[start:html.index("</script>", start)]
    assert "<" not in script and json.loads(script) == {"preview_base": base}


@pytest.mark.parametrize("value, expected", [
    ("/preview", "/preview/"), ("/preview/", "/preview/"), ("https://other/player", "https://other/player/"), ("", ""),
])
def test_the_preview_base_ends_with_a_slash_so_the_player_finds_its_files(value, expected):
    assert normalize_preview_base(value) == expected


def test_shared_setup_widgets_and_application_page(client, http_server):
    url, _ = client(FakeBridge())
    assert "canvasControls" in get_page(url, "/setup.js", "text/javascript")
    assert "encodeRow" in get_page(url, "/setup.js", "text/javascript")
    with pytest.raises(urllib.error.HTTPError) as error:
        get_page(url, "/setup")
    assert error.value.code == 404
    server = http_server(FakeBridge(), setup=SimpleNamespace(page=lambda: b"AI mixer setup"))
    assert get_page(f"http://127.0.0.1:{server.server_port}", "/setup") == "AI mixer setup"
