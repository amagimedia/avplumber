"""The web UI bridge: routing, command building and reconnection."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

from pyplumber.mixer.control import mixer_command
from webui import GpuStats, MixerBridge, serve


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
def client(monkeypatch):
    monkeypatch.setattr(GpuStats, 'snapshot', lambda _: [])
    bridges = []

    def start(bridge):
        bridges.append(bridge)
        server = serve(bridge, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}", server

    yield start


def test_gpu_samples_are_cached_and_missing_metrics_stay_unknown(monkeypatch):
    now = [0]
    calls = []
    monkeypatch.setattr('webui.time.monotonic', lambda: now[0])
    def query(*args, **kwargs):
        calls.append(args)
        return '0, 93, [N/A], 12, 14336, 15360\n1, 0, 99, [N/A], 15104, 24576\n'
    monkeypatch.setattr('webui.subprocess.check_output', query)
    stats = GpuStats()
    first = stats.snapshot()
    assert first[0] == dict(index=0, gpu=93, decoder=None, encoder=12, memory_used_mib=14336, memory_total_mib=15360)
    assert first[1]['encoder'] is None
    assert first[1]['index'] == 1 and first[1]['gpu'] == 0
    now[0] = .5
    assert stats.snapshot() == first and len(calls) == 1
    now[0] = 1
    stats.lock.acquire()
    try:
        assert stats.snapshot() == first and len(calls) == 1
    finally:
        stats.lock.release()
    stats.snapshot()
    assert len(calls) == 2


@pytest.mark.parametrize('error', [FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 1)])
def test_gpu_query_failure_clears_old_sample_and_is_cached(monkeypatch, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise error
    monkeypatch.setattr('webui.subprocess.check_output', fail)
    stats = GpuStats()
    stats.values = [{'gpu': 90}]
    assert stats.snapshot() == []
    assert stats.snapshot() == [] and len(calls) == 1


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
    assert bridge.sent == ["mixer.status mixer", "mixer.scenes mixer", "mixer.settings mixer"]


STATE_REPLIES = {"mixer.status": '{"pgm_scene":"a"}', "mixer.scenes": '["a","b"]', "mixer.settings": "{}"}
ONE_POLL = ["mixer.status mixer", "mixer.scenes mixer", "mixer.settings mixer"]


def test_polls_share_one_state_until_it_expires_or_a_take_lands(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("webui.time.monotonic", lambda: now[0])
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
    assert post(url, {"command": "wipe", "scene": "two_up", "wipe_file": "/media/w.mov"})[0] == 200
    assert post(url, {"command": "interrupt"})[0] == 200

    assert bridge.sent == [
        'mixer.preview {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.cut {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"curve":"ease-in","mixer":"mixer"}',
        'mixer.wipe {"scene":"two_up","wipe_file":"/media/w.mov","mixer":"mixer"}',
        'mixer.interrupt {"mixer":"mixer"}',
    ]


def test_key_fades_reach_the_mixer_unchanged(client):
    bridge = FakeBridge({"mixer.dsk": "[]"})
    url, _ = client(bridge)
    assert post(url, {"command": "dsk", "key": "bug", "on": True, "fade_seconds": 0.5, "curve": "ease-out"})[0] == 200
    assert bridge.sent == ['mixer.dsk {"key": "bug", "on": true, "fade_seconds": 0.5, "curve": "ease-out"}']


def test_aux_page_requests_reach_the_mixer(client):
    bridge = FakeBridge({"mixer.aux_page": '{"page": 3}'})
    url, _ = client(bridge)
    assert post(url, {"command": "aux_page", "bus": "mv2", "step": 1})[0] == 200
    assert bridge.sent == ['mixer.aux_page {"bus": "mv2", "step": 1}']


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


def test_the_page_and_only_the_page_is_served(client):
    url, _ = client(FakeBridge({"mixer.status": "{}", "mixer.scenes": "[]"}))
    with urllib.request.urlopen(url + "/", timeout=5) as response:
        page = response.read().decode()
    assert response.headers["Content-Type"].startswith("text/html")
    assert "AVPlumber mixer" in page and "/api/state" in page
    try:
        urllib.request.urlopen(url + "/nope", timeout=5)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
