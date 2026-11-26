"""The REST gateway over the fake engine: which control line each route sends, what it refuses
before the engine sees it, and how the engine's failures are answered."""
import importlib.util
import json
import socketserver
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from pyplumber.mixer.control import AvpProtocolError
from pyplumber.mixer.gui.bridge import MixerBridge
from pyplumber.mixer.rest import ClientBridge, mixer_router
from pyplumber.rest import ApiError, create_app, serve_in_thread
from pyplumber.rest.export import PATH as OPENAPI_PATH, document
from pyplumber.rest.fake import FakeEngine

ROOT = Path(__file__).resolve().parents[2]


def line(command, **payload):
    """The control line of a take: compact JSON, as `mixer_command` writes it."""
    return f"mixer.{command} {json.dumps(payload, separators=(',', ':'))}"


def spaced(command, **payload):
    """The control line of a key, AUX or audio command: `MixerBridge.take` writes those with json.dumps' default separators."""
    return f"mixer.{command} {json.dumps(payload)}"


# --- every route sends its command ---

@pytest.mark.parametrize("method, path, body, status, sent", [
    ("GET", "/engine/version", None, 200, "version"),
    ("GET", "/engine/nodes", None, 200, "nodes.json"),
    ("GET", "/engine/nodes/comp_a", None, 200, "node.param.get comp_a"),
    ("GET", "/engine/nodes/comp_a/objects/streams", None, 200, "node.object.get comp_a streams"),
    ("GET", "/engine/queues", None, 200, "queues.json"),
    ("GET", "/mixer/status", None, 200, "mixer.status mixer"),
    ("GET", "/mixer/scenes", None, 200, "mixer.scenes mixer"),
    ("GET", "/mixer/settings", None, 200, "mixer.settings mixer"),
    ("PUT", "/mixer/preview", {"scene": "cam2"}, 204, line("preview", scene="cam2", mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "cut", "scene": "cam2"}, 200, line("cut", scene="cam2", mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "cut", "scene": "cam2", "zoom": {"duration_sec": 0.5, "curve": "linear"}, "at_show_tick": 90},
     200, line("cut", scene="cam2", zoom={"duration_sec": 0.5, "curve": "linear"}, at_show_tick=90, mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "duration_sec": 2, "curve": "ease-out", "color": "#000000"},
     200, line("fade", scene="cam2", duration_sec=2.0, curve="ease-out", color="#000000", mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "color": None}, 200, line("fade", scene="cam2", mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "wipe", "scene": "cam2", "wipe_file": "/media/wipe one.mov"},
     200, line("wipe", scene="cam2", wipe_file="/media/wipe one.mov", mixer="mixer")),
    ("POST", "/mixer/takes", {"type": "move", "scene": "cam2", "duration_sec": 0.5, "unmatched": "cut"},
     200, line("move", scene="cam2", duration_sec=0.5, unmatched="cut", mixer="mixer")),
    ("POST", "/mixer/transition/interrupt", None, 204, line("interrupt", mixer="mixer")),
    ("PUT", "/mixer/lever", {"position": 0.25, "duration_sec": 1, "move": True},
     204, line("lever", position=0.25, duration_sec=1.0, move=True, mixer="mixer")),
    ("GET", "/mixer/keys", None, 200, "mixer.dsk_status"),
    ("PATCH", "/mixer/keys/logo", {"on": True, "fade_seconds": 0, "curve": "linear"},
     200, spaced("dsk", on=True, fade_seconds=0.0, curve="linear", key="logo")),
    ("PATCH", "/mixer/keys/logo", {"rate": 1.5}, 200, spaced("dsk", key="logo", rate=1.5)),
    ("PATCH", "/mixer/keys/logo", {"rate": None}, 200, spaced("dsk", key="logo", rate=None)),
    ("PATCH", "/mixer/keys", {"keys": ["logo", "lower_third"], "on": True}, 200, spaced("dsk", keys=["logo", "lower_third"], on=True)),
    ("PATCH", "/mixer/keys", {"keys": ["logo"], "toggle": True, "fade_seconds": 0.2},
     200, spaced("dsk", keys=["logo"], toggle=True, fade_seconds=0.2)),
    ("PATCH", "/mixer/keys", {"keys": [], "prepare": True}, 200, spaced("dsk", keys=[], prepare=True)),
    ("PUT", "/mixer/keys/rate", {"seconds": 0.8}, 200, spaced("dsk", all_rate=0.8)),
    ("GET", "/mixer/ftb", None, 200, "mixer.ftb_status"),
    ("PUT", "/mixer/ftb", {"on": True, "fade_seconds": 1}, 200, spaced("ftb", on=True, fade_seconds=1.0)),
    ("PUT", "/mixer/ftb", {"toggle": True}, 200, spaced("ftb", toggle=True)),
    ("GET", "/mixer/aux", None, 200, "mixer.aux_status"),
    ("PUT", "/mixer/aux/mv/layout", {"layout": {"preset": "source_pages"}}, 200, spaced("aux_layout", layout={"preset": "source_pages"}, bus="mv")),
    ("PUT", "/mixer/sources/cam1", {"url": "file:///clip.mp4", "loop": True},
     200, spaced("source.set", url="file:///clip.mp4", loop=True, id="cam1", mixer="mixer")),
    ("GET", "/mixer/audio/gains", None, 200, "mixer.audio_gain_status"),
    ("PATCH", "/mixer/audio/gains", {"gains": {"cam1": -6, "cam2": 3.5}}, 200, spaced("audio_gain", gains={"cam1": -6.0, "cam2": 3.5})),
    ("GET", "/mixer/audio/processors", None, 200, "mixer.audio_processor_status"),
    ("PATCH", "/mixer/audio/processors/program/duck", {"enabled": False, "depth_db": -12},
     200, spaced("audio_processor", enabled=False, depth_db=-12.0, output="program", id="duck")),
])
def test_route_sends_its_command(api, engine, method, path, body, status, sent):
    response = api.request(method, path, json=body)
    assert response.status_code == status, response.text
    assert sent in engine.sent   # the bridge reads the settings together with the show's other fixed parts


def test_take_reaches_program_and_swaps_preview(api):
    assert api.put("/mixer/preview", json={"scene": "cam3"}).status_code == 204
    assert api.post("/mixer/takes", json={"type": "cut", "scene": "cam3"}).json() == {"superseded": False}
    status = api.get("/mixer/status").json()
    assert (status["pgm_scene"], status["pvw_scene"]) == ("cam3", "cam1")
    assert status["lever"] == {"position": 0.0, "origin": 0.0, "engaged": False}   # fields the model does not name pass through


def test_source_set_repoints_a_wired_pad(api, engine):
    reply = api.put("/mixer/sources/cam2", json={"url": "file:///next.mp4", "loop": False}).json()
    assert reply == {"id": "cam2", "url": "file:///next.mp4", "loop": False}
    assert engine.sources["cam2"]["url"] == "file:///next.mp4"
    refused = api.put("/mixer/sources/nowhere", json={"url": "file:///x", "loop": True})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "engine_rejected"


def test_key_and_aux_replies_are_the_mixers(api):
    keys = api.patch("/mixer/keys/logo", json={"on": True}).json()["keys"]
    assert [(key["id"], key["on"]) for key in keys] == [("logo", True), ("lower_third", False)]
    assert api.get("/mixer/keys").json()["keys"] == keys   # the change expired the bridge's cached tally
    bus = api.get("/mixer/aux").json()["buses"][0]
    changed = api.put("/mixer/aux/mv/scenes", json={"scenes": ["cam3", None], "expected_revision": bus["revision"]}).json()
    assert changed["scenes"] == ["cam3", None] and changed["revision"] != bus["revision"]


# --- refused before the engine sees it ---

@pytest.mark.parametrize("method, path, body", [
    ("PUT", "/mixer/preview", {}),
    ("PUT", "/mixer/preview", {"scene": ""}),
    ("PUT", "/mixer/preview", {"scene": "cam2", "mixer": "other"}),            # a field the model does not name
    ("POST", "/mixer/takes", {"scene": "cam2"}),                               # no type
    ("POST", "/mixer/takes", {"type": "dissolve", "scene": "cam2"}),
    ("POST", "/mixer/takes", {"type": "cut", "scene": "cam2", "duration_sec": 1}),   # a fade's field
    ("POST", "/mixer/takes", {"type": "cut", "scene": "cam2", "zoom": {"duration_sec": 0.05}}),
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "duration_sec": 0}),
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "duration_sec": "1"}),   # strict: not a number
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "curve": "bounce"}),
    ("POST", "/mixer/takes", {"type": "fade", "scene": "cam2", "color": "black"}),
    ("POST", "/mixer/takes", {"type": "wipe", "scene": "cam2"}),               # no wipe_file
    ("POST", "/mixer/takes", {"type": "move", "scene": "cam2", "unmatched": "slide"}),
    ("PUT", "/mixer/lever", {"position": 1.5}),
    ("PUT", "/mixer/lever", {"position": True}),
    ("PATCH", "/mixer/keys/logo", {}),
    ("PATCH", "/mixer/keys/logo", {"on": True, "rate": 1}),
    ("PATCH", "/mixer/keys/logo", {"rate": 1, "fade_seconds": 1}),
    ("PATCH", "/mixer/keys/logo", {"on": True, "fade_seconds": 11}),
    ("PATCH", "/mixer/keys/logo", {"on": 1}),
    ("PATCH", "/mixer/keys", {"keys": ["logo"]}),
    ("PATCH", "/mixer/keys", {"keys": ["logo"], "on": True, "toggle": True}),
    ("PATCH", "/mixer/keys", {"keys": [], "on": True}),
    ("PUT", "/mixer/keys/rate", {"seconds": -1}),
    ("PUT", "/mixer/ftb", {}),
    ("PUT", "/mixer/ftb", {"on": True, "toggle": True}),
    ("PUT", "/mixer/aux/mv/scenes", {"scenes": ["cam1", None]}),               # no expected_revision
    ("PUT", "/mixer/aux/mv/layout", {"layout": "source_pages"}),
    ("PUT", "/mixer/aux/mv/page", {}),
    ("PUT", "/mixer/aux/mv/page", {"page": 1, "step": 1}),
    ("PUT", "/mixer/sources/cam1", {}),
    ("PUT", "/mixer/sources/cam1", {"url": "", "loop": True}),
    ("PUT", "/mixer/sources/cam1", {"url": "file:///a.mp4", "loop": 1}),
    ("PATCH", "/mixer/audio/gains", {"gains": {}}),
    ("PATCH", "/mixer/audio/gains", {"gains": {"cam1": 11}}),
    ("PATCH", "/mixer/audio/processors/program/duck", {}),
    ("PATCH", "/mixer/audio/processors/program/duck", {"depth_db": 1}),
    ("GET", "/engine/nodes/comp_a%20extra", None),                             # would be a second argument
    ("GET", "/engine/nodes/comp_a/objects/x%0Aversion", None),                 # would be a second command
    ("GET", "/engine/nodes/comp_a%0A", None),
])
def test_invalid_request_never_reaches_the_engine(api, engine, method, path, body):
    response = api.request(method, path, json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_request" and error["details"]
    assert engine.sent == []


def test_body_must_be_json(api, engine):
    """A page of another site can post text/plain without a preflight; JSON it cannot."""
    response = api.post("/mixer/takes", content='{"type": "cut", "scene": "cam2"}', headers={"Content-Type": "text/plain"})
    assert response.status_code == 422 and engine.sent == []


# --- the engine's failures ---

class Failing:
    def __init__(self, failure):
        self.failure = failure

    def command(self, line, timeout=None):
        if isinstance(self.failure, Exception):
            raise self.failure
        return self.failure


@pytest.mark.parametrize("failure, status, code, message", [
    (AvpProtocolError("400 Unknown command: mixer.status"), 404, "not_available", "Unknown command: mixer.status"),
    (AvpProtocolError("400 manager not ready: mixer.status"), 503, "engine_not_ready", "manager not ready: mixer.status"),
    (AvpProtocolError("500 ERROR: unknown mixer: mixer"), 409, "engine_rejected", "unknown mixer: mixer"),
    (ConnectionRefusedError("connection refused"), 503, "engine_unreachable", "connection refused"),
    (EOFError(), 503, "engine_unreachable", "EOFError"),
    (TimeoutError(), 504, "engine_timeout", None),
    ("not json", 502, "bad_engine_reply", None),
])
def test_engine_failure_is_answered_in_the_error_shape(failure, status, code, message):
    client = Failing(failure)
    api = TestClient(create_app(client, [mixer_router(ClientBridge(client))]), base_url="http://testserver/api/v1")
    for path in ("/mixer/status", "/engine/nodes"):
        response = api.get(path)
        assert response.status_code == status
        error = response.json()["error"]
        assert error["code"] == code and (message is None or error["message"] == message)


def test_reply_without_a_promised_field_is_the_engines_fault_not_the_gateways(api, engine):
    engine._commands["mixer.status"] = lambda _arg: {"pgm_scene": "cam1"}   # an engine without pvw_scene and transition
    response = api.get("/mixer/status")
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "bad_engine_reply" and "pvw_scene" in error["message"] and "transition" in error["message"]


def test_health_reports_an_unreachable_engine_without_failing():
    api = TestClient(create_app(Failing(ConnectionRefusedError("connection refused"))))
    response = api.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "engine": {"reachable": False, "version": None, "error": "connection refused"}}


def test_engine_refusal_of_a_take_is_a_conflict_with_its_reason(api):
    response = api.post("/mixer/takes", json={"type": "cut", "scene": "nowhere"})
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "engine_rejected", "message": "unknown scene: nowhere"}}


def test_stale_aux_revision_is_a_conflict(engine, bridge):
    remembered = []
    api = TestClient(create_app(engine, [mixer_router(bridge, on_aux_change=lambda bus, result: remembered.append((bus, result)))]),
                     base_url="http://testserver/api/v1")
    response = api.put("/mixer/aux/mv/scenes", json={"scenes": ["cam3", None], "expected_revision": "stale"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "conflict"
    assert remembered == []
    changed = api.put("/mixer/aux/mv/page", json={"step": 1})   # the fake's bus pages only in the source_pages layout
    assert changed.status_code == 409 and remembered == []
    changed = api.put("/mixer/aux/mv/layout", json={"layout": {"preset": "source_pages"}}).json()
    assert remembered == [("mv", changed)]


def test_failing_aux_persistence_does_not_fail_the_change(engine, bridge):
    def cannot_persist(bus, result):
        raise OSError("disk full")
    api = TestClient(create_app(engine, [mixer_router(bridge, on_aux_change=cannot_persist)]), base_url="http://testserver/api/v1")
    response = api.put("/mixer/aux/mv/layout", json={"layout": {"preset": "source_pages"}})
    assert response.status_code == 200 and response.json()["layout"]["preset"] == "source_pages"


def test_superseded_take_is_reported_and_the_bridge_take_is_looked_up_per_request(api, bridge, engine):
    """An application wraps `bridge.take` after the router was built (avplumber-mixer's director)."""
    seen = []
    bridge.take = lambda request: seen.append(request) or False
    assert api.post("/mixer/takes", json={"type": "cut", "scene": "cam2"}).json() == {"superseded": True}
    assert seen == [{"command": "cut", "scene": "cam2"}] and engine.sent == []


class ControlPort(socketserver.ThreadingTCPServer):
    """The engine's line protocol in front of a FakeEngine: a greeting, then per command line
    `200 OK`, `201 OK` with the content and an empty line, or the engine's error line."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, engine):
        self.engine = engine
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write(b"100 VTR READY\n")
            for raw in self.rfile:
                try:
                    content = self.server.engine.command(raw.decode().rstrip("\n"))
                    reply = "200 OK\n" if content is None else f"201 OK\n{content}\n"
                except AvpProtocolError as exc:
                    reply = f"{exc}\n"
                self.wfile.write(reply.encode())


def test_gateway_process_path_over_a_control_port(engine):
    """What `python -m pyplumber.rest --mixer mixer` builds: bridges with control connections of their own."""
    port = ControlPort(engine)
    threading.Thread(target=port.serve_forever, daemon=True).start()
    address = port.server_address
    try:
        app = create_app(MixerBridge(*address, ""), [mixer_router(MixerBridge(*address, "mixer", timeout=5.0))])
        api = TestClient(app, base_url="http://testserver/api/v1")
        assert api.get("/health").json()["status"] == "ok"
        assert [node["name"] for node in api.get("/engine/nodes").json()["nodes"]][-1] == "comp_a"
        assert api.post("/mixer/takes", json={"type": "cut", "scene": "cam2"}).json() == {"superseded": False}
        assert api.get("/mixer/status").json()["pgm_scene"] == "cam2"
        assert api.put("/mixer/lever", json={"position": 0.5}).status_code == 204
        assert api.post("/mixer/takes", json={"type": "cut", "scene": "nowhere"}).json()["error"]["code"] == "engine_rejected"
        del engine._commands["mixer.dsk_status"]   # a show without the Python keyer's commands
        assert api.get("/mixer/keys").json()["error"] == {"code": "not_available", "message": "Unknown command: mixer.dsk_status"}
    finally:
        port.shutdown()
        port.server_close()
    # Connections that were open still answer (the handler threads live on); a new one is refused.
    app = create_app(MixerBridge(*address, ""))
    response = TestClient(app, base_url="http://testserver/api/v1").get("/engine/version")
    assert response.status_code == 503 and response.json()["error"]["code"] == "engine_unreachable"


def test_engine_an_application_is_starting_is_not_asked(engine, bridge):
    """It would not answer before its graph is built: each request would wait for its timeout."""
    reason = ["Restarting mixer…"]
    starting = lambda: reason[0]
    app = create_app(engine, [mixer_router(bridge, starting=starting), application_router()], token="s3cret", starting=starting)
    api = TestClient(app, base_url="http://testserver/api/v1", headers={"Authorization": "Bearer s3cret"})
    for method, path, body in (("GET", "/engine/nodes", None), ("GET", "/mixer/status", None),
                               ("POST", "/mixer/takes", {"type": "cut", "scene": "cam2"})):
        response = api.request(method, path, json=body)
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "engine_not_ready", "message": "Restarting mixer…"}}
    assert api.get("/health").json() == {"status": "degraded", "engine": {"reachable": False, "version": None, "error": "Restarting mixer…"}}
    assert engine.sent == []
    assert api.get("/mixer/status", headers={"Authorization": "Bearer wrong"}).status_code == 401   # who asks comes first
    assert api.get("/director").json() == {"enabled": True}   # the application's own state needs no engine
    reason[0] = None
    assert api.get("/mixer/status").status_code == 200 and api.get("/engine/nodes").status_code == 200
    assert api.get("/health").json()["status"] == "ok"


# --- who may call ---

def test_token_is_required_on_routes_but_not_on_health_and_the_document(engine, bridge):
    api = TestClient(create_app(engine, [mixer_router(bridge)], token="s3cret"), base_url="http://testserver/api/v1")
    for path in ("/mixer/status", "/engine/nodes"):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic s3cret"}):
            response = api.get(path, headers=headers)
            assert response.status_code == 401 and response.json()["error"]["code"] == "unauthorized"
            assert response.headers["WWW-Authenticate"] == "Bearer"
        assert api.get(path, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert api.post("/mixer/takes", json={"type": "cut", "scene": "cam2"}).status_code == 401
    assert [sent for sent in engine.sent if sent.startswith("mixer.cut")] == []
    assert api.get("/health").status_code == 200
    assert api.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("headers, status", [
    ({"Origin": "http://evil.example"}, 403),
    ({"Sec-Fetch-Site": "cross-site"}, 403),
    ({"Origin": "http://testserver"}, 204),
    ({}, 204),
])
def test_another_sites_page_cannot_take(api, engine, headers, status):
    """`POST /mixer/transition/interrupt` has no body, so no preflight would stop a cross-site form."""
    response = api.post("/mixer/transition/interrupt", headers=headers)
    assert response.status_code == status
    assert bool(engine.sent) == (status == 204)


# --- routers ---

def test_gateway_without_a_mixer_serves_the_engine_only(engine):
    app = create_app(engine)
    api = TestClient(app, base_url="http://testserver/api/v1")
    assert api.get("/engine/version").status_code == 200
    response = api.get("/mixer/status")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert not [path for path in app.openapi()["paths"] if "/mixer" in path]


def application_router(prefix="/director", tags=("director",)):
    router = APIRouter(prefix=prefix, tags=list(tags))

    @router.get("")
    def describe():
        return {"enabled": True}

    @router.delete("")
    def fail():
        raise KeyError("/secret/path")

    @router.patch("")
    def control():
        raise ApiError(409, "director_busy", "a rundown is running")
    return router


def test_application_router_is_served_beside_the_core_under_its_rules(engine, bridge):
    app = create_app(engine, [mixer_router(bridge), application_router()], token="s3cret")
    api = TestClient(app, base_url="http://testserver/api/v1", raise_server_exceptions=False)
    assert api.get("/director").status_code == 401
    authorized = {"Authorization": "Bearer s3cret"}
    failed = api.delete("/director", headers=authorized)   # an exception the router did not expect
    assert failed.status_code == 500 and failed.json()["error"]["code"] == "internal_error"
    assert "secret" not in failed.text
    assert api.get("/director", headers=authorized).json() == {"enabled": True}
    assert api.patch("/director", headers=authorized).json() == {"error": {"code": "director_busy", "message": "a rundown is running"}}
    assert api.get("/mixer/status", headers=authorized).status_code == 200
    operations = app.openapi()["paths"]["/api/v1/director"]
    assert operations["get"]["operationId"] == "director_describe"


@pytest.mark.parametrize("router", [
    application_router("/engine/nodes", ("app",)),   # GET /engine/nodes is the core's
    application_router("/mixer/keys", ("app",)),     # GET and PATCH /mixer/keys are the mixer router's
])
def test_router_cannot_replace_an_operation(engine, bridge, router):
    with pytest.raises(ValueError, match="already served"):
        create_app(engine, [mixer_router(bridge), router])


def test_parameter_name_does_not_make_a_path_new(engine):
    router = APIRouter(prefix="/engine", tags=["app"])
    router.add_api_route("/nodes/{node_id}", lambda node_id: {}, methods=["GET"])
    with pytest.raises(ValueError, match="already served"):
        create_app(engine, [router])


# --- the document ---

def test_checked_in_openapi_document_matches_the_code():
    assert OPENAPI_PATH.read_text() == document(), \
        "regenerate with: python -m pyplumber.rest.export (FastAPI and pydantic as pinned in pyplumber/rest/requirements.txt)"


def test_document_names_operations_and_the_error_shape(api):
    schema = api.get("/openapi.json").json()
    take = schema["paths"]["/api/v1/mixer/takes"]["post"]
    assert take["operationId"] == "mixer_take"
    assert take["responses"]["409"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert take["responses"]["422"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert take["security"] == [{"HTTPBearer": []}]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]
    ids = [operation["operationId"] for path in schema["paths"].values() for operation in path.values()]
    assert len(ids) == len(set(ids))
    assert api.get("/docs").status_code == 200


# --- serving ---

def test_serve_in_thread_and_the_smoke_tool(engine, bridge, capsys):
    server = serve_in_thread(create_app(engine, [mixer_router(bridge)]), port=0)
    try:
        url = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(f"{url}/api/v1/health", timeout=5) as response:
            assert json.load(response)["status"] == "ok"
        spec = importlib.util.spec_from_file_location("rest_smoke", ROOT / "tools" / "rest_smoke.py")
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        assert smoke.main(["--url", url]) == 0, capsys.readouterr()
        takes = [sent.split()[0] for sent in engine.sent if sent.startswith(("mixer.cut", "mixer.preview"))]
        assert takes == ["mixer.preview", "mixer.cut", "mixer.cut"]   # no Preview before: only Program is put back
        assert engine.pgm == "cam1"
        engine.sent.clear()
        assert smoke.main(["--url", url, "--read-only"]) == 0
        assert not [sent for sent in engine.sent if sent.startswith(("mixer.cut", "mixer.preview"))]
    finally:
        server.close()


def test_control_page_and_fake_engine_load_without_fastapi():
    """The gateway's dependencies are optional: the mixer's control page shares only `pyplumber.rest.origin`."""
    code = ("import sys\n"
            "for name in ('fastapi', 'pydantic', 'uvicorn', 'starlette'): sys.modules[name] = None\n"
            "import pyplumber.mixer.gui.web, pyplumber.rest, pyplumber.rest.fake, pyplumber.rest.origin\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
