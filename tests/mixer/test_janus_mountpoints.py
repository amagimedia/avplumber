"""janus_mountpoints against a fake Janus HTTP API: sessions, the Streaming plugin's list, create
and destroy requests."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

import pyplumber.mixer.tools.janus_mountpoints as janus_mountpoints


class FakeJanus(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        state = self.server.state
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state["requests"].append((self.path, message))
        reply = {"janus": "success", "transaction": message["transaction"]}
        if message["janus"] in ("create", "attach"):
            reply["data"] = {"id": {"/janus": 11, "/janus/11": 22}[self.path]}
        elif message["janus"] == "message":
            assert self.path == "/janus/11/22"
            body, mountpoints = message["body"], state["mountpoints"]
            data = {"streaming": body["request"]}
            if body["request"] == "list":
                data["list"] = [{"id": i, "description": d} for i, d in mountpoints.items()]
            elif body["request"] == "info":
                data["info"] = json.loads(json.dumps(state["contracts"][body["id"]]))
                if data["info"].get("secret") and body.get("secret") != data["info"]["secret"]:
                    for media in data["info"]["media"]:
                        media.pop("port", None)
            elif body["request"] == "create" and (body["id"] in mountpoints or body["id"] in state["hidden"]):
                data = {"streaming": "event", "error_code": 456, "error": "A stream with the provided ID already exists"}
            elif body["request"] == "create" and body["id"] in state["refused"]:
                data = {"streaming": "event", "error_code": 456, "error": "Can't add 'rtp' stream"}
                state["refused"].remove(body["id"])
            elif body["request"] == "create":
                mountpoints[body["id"]] = body["description"]
                state["contracts"][body["id"]] = body
            else:
                del mountpoints[body["id"]]
            reply["plugindata"] = {"plugin": "janus.plugin.streaming", "data": data}
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def janus():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeJanus)
    server.state = {"requests": [], "hidden": set(), "refused": set(), "contracts": {}, "mountpoints": {
        1: "avplumber program WebRTC preview", 5008: "avplumber SDR multiview",
        5016: "avplumber extra aux aux1", 5024: "avplumber extra aux aux3"}}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(api=f"http://127.0.0.1:{server.server_port}/janus", state=server.state)
    server.shutdown()
    server.server_close()


def streaming(janus, request):
    return [m["body"] for _, m in janus.state["requests"] if m["janus"] == "message" and m["body"]["request"] == request]


def test_sync_creates_missing_mountpoints_and_prunes_only_its_own(janus):
    janus_mountpoints.sync(janus.api, {"aux1": 5016, "aux2": 5020})
    assert streaming(janus, "create") == [{
        "request": "create", "type": "rtp", "id": 5020, "description": "avplumber extra aux aux2",
        "audio": False, "video": True, "videoport": 5020, "videortcpport": 5021, "videoiface": "127.0.0.1",
        "videopt": 96, "videocodec": "h264", "videofmtp": "profile-level-id=640028;packetization-mode=1",
        "permanent": False}]
    assert 5024 in janus.state["mountpoints"], "the previous show's stay until it is replaced"
    janus_mountpoints.sync(janus.api, {"aux1": 5016, "aux2": 5020}, prune=True)
    assert streaming(janus, "destroy") == [{"request": "destroy", "id": 5024, "permanent": False}]
    assert janus.state["mountpoints"] == {1: "avplumber program WebRTC preview", 5008: "avplumber SDR multiview",
                                          5016: "avplumber extra aux aux1", 5020: "avplumber extra aux aux2"}
    sessions = [path for path, m in janus.state["requests"] if m["janus"] == "destroy"]
    assert sessions == ["/janus/11", "/janus/11"]


def test_a_mountpoint_created_meanwhile_counts_as_created(janus):
    janus.state["hidden"].add(5020)
    janus_mountpoints.sync(janus.api, {"aux2": 5020})


def test_a_refused_create_fails_and_closes_the_session(janus):
    janus.state["refused"].add(5020)
    with pytest.raises(RuntimeError, match="Janus streaming create: Can't add 'rtp' stream"):
        janus_mountpoints.sync(janus.api, {"aux2": 5020})
    assert janus.state["requests"][-1] == ("/janus/11", {"janus": "destroy", "transaction": "setup"})


def test_an_unreachable_api_fails():
    with pytest.raises(OSError):
        janus_mountpoints.sync("http://127.0.0.1:9/janus", {"aux1": 5016})


def contract(mid=1, port=5004, secret=""):
    return {"id": mid, "description": "avplumber program WebRTC preview", "secret": secret, "pin": "test-pin",
            "media": [{"type": "video", "mid": "v", "port": port, "rtcpport": port + 1, "pt": 96,
                       "codec": "h264", "fmtp": "profile-level-id=42e028;packetization-mode=1"},
                      {"type": "audio", "mid": "a", "port": 5002, "pt": 111, "codec": "opus"}]}


def test_codec_replacement_preserves_ports_audio_and_protection(janus, monkeypatch):
    previous = contract(secret="test-secret")
    janus.state["contracts"][1] = previous
    monkeypatch.setenv("JANUS_STREAMING_SECRET", "test-secret")
    desired = janus_mountpoints._create("sdr", 5004, "h265", "profile-id=1", extra=False)
    janus_mountpoints.sync(janus.api, {"sdr": desired}, replace=True)
    replacement = streaming(janus, "create")[0]
    assert (replacement["secret"], replacement["pin"]) == ("test-secret", "test-pin")
    assert replacement["media"][0] == {**previous["media"][0], "codec": "h265", "fmtp": "profile-id=1", "iface": "127.0.0.1"}
    assert replacement["media"][1] == {**previous["media"][1], "iface": "127.0.0.1"}
    assert streaming(janus, "destroy") == [{"request": "destroy", "id": 1, "permanent": False, "secret": "test-secret"}]


def test_replacement_preflights_secret_before_destroying(janus):
    janus.state["contracts"][1] = contract(secret="test-secret")
    desired = janus_mountpoints._create("sdr", 5004, "h265", "profile-id=1", extra=False)
    with pytest.raises(RuntimeError, match="JANUS_STREAMING_SECRET"):
        janus_mountpoints.sync(janus.api, {"sdr": desired}, replace=True)
    assert not streaming(janus, "destroy")


def test_replacement_failure_restores_old_codec(janus):
    janus.state["contracts"][1] = contract()
    janus.state["refused"].add(1)
    desired = janus_mountpoints._create("sdr", 5004, "h265", "profile-id=1", extra=False)
    with pytest.raises(RuntimeError, match="Can't add"):
        janus_mountpoints.sync(janus.api, {"sdr": desired}, replace=True)
    assert [body["media"][0]["codec"] for body in streaming(janus, "create")] == ["h265", "h264"]
    assert janus.state["contracts"][1]["media"][0]["codec"] == "h264"


def test_later_failure_restores_prior_replacements_and_removes_new_mounts(janus):
    janus.state["contracts"][1] = contract()
    janus.state["refused"].add(5030)
    specs = {"sdr": janus_mountpoints._create("sdr", 5004, "h265", "profile-id=1", extra=False),
             "aux4": 5028, "aux5": 5030}
    with pytest.raises(RuntimeError, match="Can't add"):
        janus_mountpoints.sync(janus.api, specs, replace=True)
    assert janus.state["contracts"][1]["media"][0]["codec"] == "h264"
    assert 5028 not in janus.state["mountpoints"]


def test_unchanged_codec_does_not_interrupt_viewers(janus):
    janus.state["contracts"][1] = contract()
    spec = janus_mountpoints._create("sdr", 5004, fmtp=contract()["media"][0]["fmtp"], extra=False)
    janus_mountpoints.sync(janus.api, {"sdr": spec}, replace=True)
    assert not streaming(janus, "destroy") and not streaming(janus, "create")


def test_output_contracts_distinguish_sdr_hevc_hdr_hevc_and_aux(janus):
    from test_graph import CONFIG
    show = {**CONFIG, "canvas": {**CONFIG["canvas"], "working_format": "p010le", "color": "hlg"},
            "renditions": [{"id": "sdr", "codec": "hevc_nvenc", "profile": "main", "color": "sdr", "port": 5004},
                           {"id": "hdr", "codec": "hevc_nvenc", "profile": "main10", "color": "hlg", "port": 5006}],
            "aux_buses": [{"id": "aux0", "layout": {"preset": "source_pages"},
                           "renditions": [{"id": "monitor", "codec": "hevc_nvenc", "port": 5016}]}]}
    specs = janus_mountpoints.outputs(show, extra_aux=1)
    assert [(s["id"], s["videocodec"], s["videofmtp"]) for s in specs.values()] == [
        (1, "h265", "profile-id=1"), (2, "h265", "profile-id=2"), (5016, "h265", "profile-id=1")]
    assert specs["aux0:monitor"]["description"].startswith(janus_mountpoints.DESCRIPTION)
    janus.state["mountpoints"].clear()
    janus_mountpoints.sync(janus.api, specs, replace=True)
    assert set(janus.state["mountpoints"]) == {1, 2, 5016}
