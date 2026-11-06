"""The mixer's state through the preview server: last good reply while the mixer restarts."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import urllib.error
import urllib.request

import pytest


@pytest.fixture
def mixer():
    """A stand-in for the mixer's /api/state: answers 200 until told to answer something else."""
    state = {"status": 200, "body": {"status": {"pgm_scene": "det-6"}}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps(state["body"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{http.server_port}/api/state"
    yield state
    http.shutdown()
    http.server_close()


def get(http, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{http.server_port}{path}", timeout=5) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def test_a_restarting_mixer_serves_its_last_good_state_marked_stale(server, http, mixer, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(server, "MIXER_STATE_URL", mixer["url"])
    monkeypatch.setattr(server, "mixer_state", server.LastGood(clock=lambda: now[0]))
    status, headers, body = get(http, "/mixer-state")
    assert status == 200 and json.loads(body)["status"]["pgm_scene"] == "det-6" and "X-Mixer-State" not in headers
    mixer["status"], mixer["body"] = 503, {"error": "mixer starting"}
    now[0] += 30
    status, headers, body = get(http, "/mixer-state")
    assert status == 200 and json.loads(body)["status"]["pgm_scene"] == "det-6"
    assert headers["X-Mixer-State"] == "stale; age=30"
    now[0] += 100                                       # over 120 s: the restart is reported
    status, _, body = get(http, "/mixer-state")
    assert status == 503 and json.loads(body) == {"error": "mixer starting"}


def test_a_client_error_is_never_masked_and_a_dead_mixer_without_history_is_a_502(server, http, mixer, monkeypatch):
    monkeypatch.setattr(server, "MIXER_STATE_URL", mixer["url"])
    monkeypatch.setattr(server, "mixer_state", server.LastGood())
    assert get(http, "/mixer-state")[0] == 200
    mixer["status"] = 404
    assert get(http, "/mixer-state")[0] == 404
    monkeypatch.setattr(server, "MIXER_STATE_URL", "http://127.0.0.1:9/api/state")
    monkeypatch.setattr(server, "mixer_state", server.LastGood())
    assert get(http, "/mixer-state")[0] == 502


def test_metrics_json_is_no_content_when_embedded_in_the_mixer(server, http, monkeypatch):
    monkeypatch.setattr(server, "MIXER_STATE_URL", "http://127.0.0.1:9/api/state")
    status, _, body = get(http, "/metrics.json")
    assert status == 204 and body == b""
    monkeypatch.setattr(server, "MIXER_STATE_URL", "")
    assert get(http, "/metrics.json")[0] == 404           # a standalone player has no metrics file either


def test_the_player_binds_where_it_is_told(server):
    assert hasattr(server, "PREVIEW_BIND")                # loopback behind the login (1cca3a17)
