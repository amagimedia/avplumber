"""janus_mountpoints against a fake Janus HTTP API: sessions, the Streaming plugin's list, create
and destroy requests."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

import janus_mountpoints


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
            elif body["request"] == "create" and (body["id"] in mountpoints or body["id"] in state["hidden"]):
                data = {"streaming": "event", "error_code": 456, "error": "A stream with the provided ID already exists"}
            elif body["request"] == "create" and body["id"] in state["refused"]:
                data = {"streaming": "event", "error_code": 456, "error": "Can't add 'rtp' stream"}
            elif body["request"] == "create":
                mountpoints[body["id"]] = body["description"]
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
    server.state = {"requests": [], "hidden": set(), "refused": set(), "mountpoints": {
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
