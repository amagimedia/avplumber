"""The web UI bridge: routing, command building and reconnection."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from webui import MixerBridge, serve


class FakeBridge(MixerBridge):
    """A bridge with the control connection replaced by a scripted stub."""

    def __init__(self, replies=None, fail_take=None):
        self.mixer = "mixer"
        self.timeout = 5.0
        self.sent: list[str] = []
        self.replies = replies or {}
        self.fail_take = fail_take

    def command(self, line: str):
        self.sent.append(line)
        if self.fail_take and line.startswith(self.fail_take):
            raise RuntimeError("mixer said no")
        for prefix, reply in self.replies.items():
            if line.startswith(prefix):
                return reply
        return None


@pytest.fixture
def client():
    bridges = []

    def start(bridge):
        bridges.append(bridge)
        server = serve(bridge, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}", server

    yield start


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


def test_state_survives_a_mixer_without_settings(client):
    bridge = FakeBridge({"mixer.status": "{}", "mixer.scenes": "[]"}, fail_take="mixer.settings")
    url, _ = client(bridge)
    assert get(url, "/api/state")[1]["settings"] == {}


def test_takes_reach_the_mixer_as_control_commands(client):
    bridge = FakeBridge()
    url, _ = client(bridge)

    assert post(url, {"command": "preview", "scene": "grid_4_page_0"})[0] == 200
    assert post(url, {"command": "cut", "scene": "grid_4_page_0"})[0] == 200
    assert post(url, {"command": "fade", "scene": "two_up", "duration_sec": 0.8})[0] == 200
    assert post(url, {"command": "wipe", "scene": "two_up", "wipe_file": "/media/w.mov"})[0] == 200
    assert post(url, {"command": "interrupt"})[0] == 200

    assert bridge.sent == [
        'mixer.preview {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.cut {"scene":"grid_4_page_0","mixer":"mixer"}',
        'mixer.fade {"scene":"two_up","duration_sec":0.8,"mixer":"mixer"}',
        'mixer.wipe {"scene":"two_up","wipe_file":"/media/w.mov","mixer":"mixer"}',
        'mixer.interrupt {"mixer":"mixer"}',
    ]


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
