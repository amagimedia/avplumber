"""mixer.source.set / prepare / media / resume_at without a live engine."""
from __future__ import annotations

import json
import socketserver
import threading
import time
from types import SimpleNamespace

import pytest

from pyplumber.mixer.config import ConfigError
from pyplumber.mixer.source_control import (
    SourceController, SourcePad, SourceUrls, continuous_loop_for, control_command, group_shape,
    register_source_commands, source_pads, stop_group, url_protocol,
)


def test_group_shape_names_the_last_nodes_and_the_groups_own_queues():
    nodes = [{"name": "input_0", "dst": ["v_pkt", "a_pkt"]},
             {"name": "decode_0", "src": "v_pkt", "dst": "v_raw"},
             {"name": "otm", "src": "v_raw", "dst": ["to_slot_a", "to_slot_b"]},
             {"name": "decode_audio_0", "src": "a_pkt", "dst": "to_audio_mixer"}]
    assert group_shape(nodes) == (["otm", "decode_audio_0"], ["a_pkt", "v_pkt", "v_raw"])
    assert group_shape([]) == ([], [])


class _Node:
    def __init__(self, name, *, src=None, dst=None, working=True):
        self.name = name
        self.parameters = {"name": name, **({} if src is None else {"src": src}),
                           **({} if dst is None else {"dst": dst})}
        self.isWorking = working
        self.stopped = False

    def stopAndWait(self):
        self.stopped = True
        self.isWorking = False


class _Edge:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _Group:
    def __init__(self, nodes):
        self.sortedNodes = list(nodes)
        self.stop_asked = False

    def stopNodes(self):
        self.stop_asked = True
        for node in self.sortedNodes:
            node.isWorking = False


class _Avp:
    """`refuse` lists the first words of lines this engine logs and drops, as the binding does."""

    def __init__(self, nodes, edges=None, refuse=()):
        self._group = _Group(nodes)
        self._edges = {name: _Edge() for name in (edges or [])}
        self.commands = []
        self.handlers = {}
        self.refuse = refuse

    def group(self, _name):
        return self._group

    def node(self, name):
        return next(node for node in self._group.sortedNodes if node.name == name)

    def getEdge(self, name, data_type=None):
        return self._edges[name]

    def executeCommandsFromString(self, command):
        self.commands.append(command)
        word, _, rest = command.partition(" ")
        if word == "node.param.set" and word not in self.refuse:
            name, key, value = rest.split(" ", 2)
            self.node(name).parameters[key] = json.loads(value)

    def registerControlCommand(self, name, handler, _payload, _quiet=False):
        self.handlers[name] = handler


def test_stop_group_stops_terminals_first_then_clears_internal_edges():
    terminals = [_Node("otm", src="v_raw", dst=["a", "b"]),
                 _Node("decode_audio", src="a_pkt", dst="audio_out")]
    middle = [_Node("input", dst=["v_pkt", "a_pkt"]),
              _Node("decode", src="v_pkt", dst="v_raw")]
    nodes = middle + terminals
    avp = _Avp(nodes, edges=["a_pkt", "v_pkt", "v_raw"])
    stop_group(avp, "input_0")
    assert all(n.stopped for n in terminals)
    assert not any(n.stopped for n in middle)   # stopNodes asked; stopAndWait only on lasts
    assert avp._group.stop_asked
    assert all(edge.cleared == 1 for edge in avp._edges.values())


@pytest.mark.parametrize("stdout, expect", [("12.5\n", 12_500), ("N/A\n", None), ("", None)])
def test_probe_gives_ffprobe_the_url_as_its_input_never_as_an_option(monkeypatch, stdout, expect):
    from pyplumber.mixer import source_control
    runs = []

    def run(argv, **_kwargs):
        runs.append(argv)
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(source_control.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(source_control.subprocess, "run", run)
    assert source_control.probe_duration_ms("-protocol_whitelist") == expect
    assert runs[0][0] == "/usr/bin/ffprobe" and runs[0][-2:] == ["-i", "-protocol_whitelist"]


def _pad(**over):
    base = dict(id="cam1", index=0, kind="video", group="input_0", input_node="input_0",
                sync_team="input_0_av_sync", pause_team="input_0_pause")
    base.update(over)
    return SourcePad(**base)


def test_set_repoints_url_and_restarts_the_group():
    avp = _Avp([_Node("input_0", dst="v_pkt"), _Node("otm", src="v_pkt", dst="out")],
               edges=["v_pkt"])
    controller = SourceController(avp, {"cam1": _pad()})
    assert controller.set({"mixer": "mixer", "id": "cam1", "url": "file:///clip.mp4", "loop": True}) == {
        "id": "cam1", "url": "file:///clip.mp4", "loop": True}
    assert avp._group.stop_asked
    assert avp.commands == [
        'node.param.set input_0 url "file:///clip.mp4"',
        "node.param.set input_0 loop true",
        "node.param.set input_0 loop_continuous_ts true",
        "realtime.team.reset input_0_av_sync",
        "resume input_0_pause",
        "group.start input_0",
    ]


def test_set_without_sync_or_pause_skips_those_commands():
    avp = _Avp([_Node("input_0", dst="out")])
    controller = SourceController(avp, {"cam1": _pad(sync_team=None, pause_team=None)})
    controller.set({"id": "cam1", "url": "srt://live", "loop": False})
    assert avp.commands == [
        'node.param.set input_0 url "srt://live"',
        "node.param.set input_0 loop false",
        "node.param.set input_0 loop_continuous_ts false",
        "group.start input_0",
    ]


@pytest.mark.parametrize("request_, match", [
    ({"id": "cam1", "url": "x", "loop": True, "mixer": "other"}, "unknown mixer"),
    ({"url": "x", "loop": True}, "non-empty id"),
    ({"id": "missing", "url": "x", "loop": True}, "unknown source id"),
    ({"id": "cam1", "loop": True}, "non-empty url"),
    ({"id": "cam1", "url": "x"}, "boolean loop"),
    ({"id": "cam1", "url": "x", "loop": 1}, "loop must be a boolean"),
    ({"id": "cam1", "url": "x", "loop": True, "audio_track": 1}, "unsupported fields"),
])
def test_set_checks_the_request_before_stopping(request_, match):
    avp = _Avp([_Node("input_0", dst="out")])
    controller = SourceController(avp, {"cam1": _pad()})
    with pytest.raises(ConfigError, match=match):
        controller.set(request_)
    assert avp.commands == []
    assert not avp._group.stop_asked


@pytest.mark.parametrize("url, protocol", [
    ("srt://ingest:9000?mode=caller", "srt"),
    ("/media/clip.mp4", "file"), ("clip.mp4", "file"), ("file:/media/clip.mp4", "file"),
    ("/media/a:b.mp4", "file"),                           # a colon after a character no scheme has
    ("concat:/media/a.ts|/etc/passwd", "concat"),
    ("crypto+https://cdn/clip.ts", "crypto+https"),       # FFmpeg opens it with crypto
    ("subfile,,start,0,end,100,,:/etc/passwd", "subfile"),
    ("subfile,name.mp4", "file"),                         # no colon behind: FFmpeg reads a path
    (":x", ""),
])
def test_url_protocol_is_the_one_ffmpeg_opens_the_url_with(url, protocol):
    assert url_protocol(url) == protocol


def test_source_urls_allow_protocols_and_the_files_below_directories(tmp_path):
    media, other = tmp_path / "media", tmp_path / "other"
    media.mkdir(), other.mkdir()
    (media / "out").symlink_to(other)
    urls = SourceUrls(("srt", "https", str(media)))
    for allowed in ("srt://ingest:9000", "https://cdn/clip.mp4", f"{media}/clip.mp4", f"file:{media}/sub/clip.mp4",
                    f"{media}/sub/../clip.mp4"):
        assert urls.refusal(allowed) is None, allowed
    for refused, why in (("udp://239.0.0.1:5000", "protocol 'udp'"),
                         (f"concat:{media}/a.ts|/etc/passwd", "protocol 'concat'"),
                         ("crypto+https://cdn/clip.ts", "protocol 'crypto+https'"),
                         ("/etc/passwd", "outside"), (f"{media}/../other/clip.mp4", "outside"),
                         (f"{media}/out/clip.mp4", "outside"),          # a link that leads out of the directory
                         (f"{media}-2/clip.mp4", "outside"), (str(media), "outside"),
                         ("clip.mp4", "outside")):                      # relative: resolved in the engine's directory
        assert why in urls.refusal(refused), refused
    assert SourceUrls(("file",)).refusal("/etc/passwd") is None       # "file" is every file
    assert SourceUrls(()).refusal("srt://ingest") == "protocol 'srt'"   # an empty list allows nothing
    assert SourceUrls().refusal("concat:a|b") is None                 # no list: every URL


@pytest.mark.parametrize("allowed", ["srt", ("srt://",), ("media/clips",), ("",), (7,)])
def test_a_malformed_list_of_source_urls_is_refused_when_the_mixer_is_built(allowed):
    with pytest.raises(ValueError, match="source_urls"):
        SourceController(_Avp([]), {}, allowed_urls=allowed)


@pytest.mark.parametrize("command, extra", [("set", {}), ("prepare", {"cue_in_ms": 0})])
def test_a_url_that_is_not_allowed_is_refused_before_anything_is_stopped_or_probed(command, extra):
    probed, avp = [], _Avp([_Node("input_0", dst="out")])
    controller = SourceController(avp, {"cam1": _pad()}, run_command=lambda _text: None, probe=probed.append,
                                  allowed_urls=("srt",))
    with pytest.raises(ConfigError, match=f"source.{command}: url is not allowed here: a file outside"):
        getattr(controller, command)({"id": "cam1", "url": "/etc/passwd", "loop": False, **extra})
    assert avp.commands == [] and not avp._group.stop_asked and probed == []
    assert getattr(controller, command)({"id": "cam1", "url": "srt://ingest:9000", "loop": False, **extra})["url"] \
        == "srt://ingest:9000"


def test_a_parameter_the_engine_did_not_take_is_an_error_and_the_source_plays_again():
    avp = _Avp([_Node("input_0", dst="v_pkt"), _Node("otm", src="v_pkt", dst="out")],
               edges=["v_pkt"], refuse=("node.param.set",))
    controller = SourceController(avp, {"cam1": _pad()})
    with pytest.raises(RuntimeError, match="input_0: url was not set"):
        controller.set({"id": "cam1", "url": "file:///clip.mp4", "loop": True})
    # Stopped and emptied, so it is safe to start: with the URL it had, not dark.
    assert avp._group.stop_asked and avp.commands[-1] == "group.start input_0"


def test_a_group_that_does_not_stop_is_an_error_that_says_the_source_stays_stopped(monkeypatch):
    class Stuck(_Group):
        def stopNodes(self):
            self.stop_asked = True   # and input_0 keeps running

    avp = _Avp([_Node("input_0", dst="v_pkt"), _Node("otm", src="v_pkt", dst="out")], edges=["v_pkt"])
    avp._group = Stuck(avp._group.sortedNodes)
    controller = SourceController(avp, {"cam1": _pad()})
    monkeypatch.setattr(controller, "STOP_TIMEOUT_S", 0.02)
    with pytest.raises(RuntimeError, match="input_0 did not stop within 0.02 s; source 'cam1' stays stopped"):
        controller.set({"id": "cam1", "url": "file:///clip.mp4", "loop": True})
    # Its queues were not cleared: nothing is changed or started over them.
    assert avp.commands == [] and avp._edges["v_pkt"].cleared == 0


def test_unsupported_kind_is_refused():
    avp = _Avp([])
    controller = SourceController(avp, {"page": SourcePad(
        id="page", index=1, kind="browser", group="input_1", input_node="input_1")})
    with pytest.raises(ConfigError, match="unsupported"):
        controller.set({"id": "page", "url": "https://example", "loop": False})
    for kind in ("mxl", "vmm"):
        controller = SourceController(avp, {"x": SourcePad(
            id="x", index=0, kind=kind, group="input_0", input_node="input_0")})
        with pytest.raises(ConfigError, match="unsupported"):
            controller.set({"id": "x", "url": "file:///a", "loop": False})


def test_raw_pads_accept_source_set_without_playback():
    avp = _Avp([_Node("input_0", dst="out")])
    for kind in ("v210", "nv12", "p010"):
        pad = SourcePad(id="bars", index=0, kind=kind, group="input_0", input_node="input_0",
                        continuous_loop=kind == "v210")
        controller = SourceController(avp, {"bars": pad})
        assert controller.set({"id": "bars", "url": "file:///next.v210", "loop": True}) == {
            "id": "bars", "url": "file:///next.v210", "loop": True}
        with pytest.raises(ConfigError, match="no pause or seek"):
            controller.prepare({"id": "bars", "url": "file:///next.v210", "loop": False})


def test_continuous_loop_follows_the_raw_upload_path():
    assert continuous_loop_for("video") and continuous_loop_for("v210")
    assert continuous_loop_for("nv12", raw_upload="direct")
    assert not continuous_loop_for("nv12", raw_upload="hwupload")
    assert not continuous_loop_for("p010")   # default hwupload


def test_source_pads_follow_build_application_naming():
    sources = [SimpleNamespace(id="a", kind="video", has_audio=True),
               SimpleNamespace(id="b", kind="ndi", has_audio=False),
               SimpleNamespace(id="c", kind="browser", has_audio=False),
               SimpleNamespace(id="d", kind="video", has_audio=False),
               SimpleNamespace(id="e", kind="nv12", has_audio=False),
               SimpleNamespace(id="f", kind="v210", has_audio=False)]
    pads = source_pads(sources, external={"d"})
    assert pads["a"].group == "input_0" and pads["a"].sync_team == "input_0_av_sync"
    assert pads["b"].sync_team == "input_1_av_sync" and pads["b"].kind == "ndi"
    assert pads["c"].kind == "browser" and pads["c"].sync_team is None and pads["c"].pause_team is None
    assert pads["d"].kind == "vmm"
    assert pads["d"].sync_team is None and pads["a"].pause_team is None
    assert pads["e"].kind == "nv12" and pads["e"].sync_team is None and not pads["e"].continuous_loop
    assert pads["f"].continuous_loop and pads["f"].sync_team is None
    assert source_pads(sources, raw_upload="direct")["e"].continuous_loop
    play = source_pads(sources, playback_control=True)
    assert play["b"].pause_team is None and play["b"].sync_team == "input_1_av_sync"   # NDI: live, no playback
    assert play["d"].sync_team == "input_3_av_sync" and play["d"].pause_team == "input_3_pause"
    assert play["c"].pause_team is None and play["c"].sync_team is None   # a browser pad is no build_input chain
    assert play["e"].pause_team is None and play["f"].pause_team is None


def test_register_source_commands_wires_the_control_handler():
    avp = _Avp([_Node("input_0", dst="out")])
    register_source_commands(avp, SourceController(avp, {"cam1": _pad(sync_team=None, pause_team=None)}))
    reply = json.loads(avp.handlers["mixer.source.set"](json.dumps(
        {"id": "cam1", "url": "file:///a.mp4", "loop": False})))
    assert reply == {"id": "cam1", "url": "file:///a.mp4", "loop": False}


def _controller(avp=None, *, run=None, probe=None, **pad_over):
    avp = avp or _Avp([_Node("input_0", dst="v_pkt"), _Node("otm", src="v_pkt", dst="out")],
                      edges=["v_pkt"])
    return SourceController(avp, {"cam1": _pad(**pad_over)},
                            run_command=run, probe=probe or (lambda _url: 12_345)), avp


def test_prepare_loads_pauses_and_seeks_to_cue():
    commands = []

    def run(text):
        commands.append(text)

    controller, avp = _controller(run=run)
    assert controller.prepare({"id": "cam1", "url": "file:///clip.mp4", "loop": False,
                               "cue_in_ms": 1500}) == {
        "id": "cam1", "url": "file:///clip.mp4", "loop": False, "cue_in_ms": 1500,
        "duration_ms": 12_345}
    assert avp._group.stop_asked
    assert "node.param.set input_0 url \"file:///clip.mp4\"" in avp.commands
    assert commands == ["pause input_0_pause now", "seek input_0_av_sync now 1500"]


@pytest.mark.parametrize("command, request_", [
    ("prepare", {"url": "ndi://feed", "loop": False}),
    ("media", {"action": "pause"}),
    ("resume_at", {"at_ms": 1000}),
])
def test_playback_is_refused_on_a_live_ndi_pad_that_set_still_takes(command, request_):
    commands = []
    controller, avp = _controller(run=commands.append, kind="ndi", pause_team=None)
    with pytest.raises(ConfigError, match="'ndi' has no pause or seek"):
        getattr(controller, command)({"id": "cam1", **request_})
    assert commands == [] and avp.commands == [] and not avp._group.stop_asked
    assert controller.set({"id": "cam1", "url": "ndi://feed", "loop": False})["url"] == "ndi://feed"


def test_prepare_probes_beside_the_load_and_not_under_the_pads_lock():
    """ffprobe opens the URL itself, for up to PROBE_TIMEOUT_S."""
    commands, probing, probed = [], threading.Event(), threading.Event()

    def probe(_url):
        probing.set()
        probed.wait(5)
        return 9_000

    controller, _ = _controller(run=commands.append, probe=probe)
    replies = []
    prepare = threading.Thread(daemon=True, target=lambda: replies.append(controller.prepare(
        {"id": "cam1", "url": "file:///clip.mp4", "loop": False})))
    prepare.start()
    assert probing.wait(5)
    deadline = time.monotonic() + 5
    while len(commands) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    # Loaded, paused and cued while the probe still runs, and the pad takes its next command.
    assert commands == ["pause input_0_pause now", "seek input_0_av_sync now 0"]
    controller.media({"id": "cam1", "action": "play"})
    assert prepare.is_alive() and replies == []
    probed.set()
    prepare.join(5)
    assert replies[0]["duration_ms"] == 9_000


def test_a_command_waits_for_its_own_pad_only():
    commands = []
    avp = _Avp([_Node("input_0", dst="out")])
    controller = SourceController(avp, {"cam1": _pad(), "cam2": _pad(id="cam2", index=1)}, run_command=commands.append)
    with controller._locks["cam2"]:   # a prepare at work on cam2
        controller.media({"id": "cam1", "action": "play"})
    assert commands == ["resume input_0_pause"]


def test_prepare_refuses_without_playback_teams():
    controller, _ = _controller(pause_team=None, sync_team=None)
    with pytest.raises(ConfigError, match="playback_control"):
        controller.prepare({"id": "cam1", "url": "file:///a.mp4", "loop": True})


@pytest.mark.parametrize("action, expect", [
    ("play", ["resume input_0_pause"]),
    ("pause", ["pause input_0_pause now"]),
    ("restart", ["seek input_0_av_sync now 0", "resume input_0_pause"]),
    ("stop", ["seek input_0_av_sync now 0", "pause input_0_pause now"]),
])
def test_media_actions(action, expect):
    commands = []
    controller, _ = _controller(run=commands.append)
    assert controller.media({"id": "cam1", "action": action}) == {"id": "cam1", "action": action}
    assert commands == expect


def test_media_seek_needs_position():
    commands = []
    controller, _ = _controller(run=commands.append)
    assert controller.media({"id": "cam1", "action": "seek", "position_ms": 2500}) == {
        "id": "cam1", "action": "seek", "position_ms": 2500}
    assert commands == ["seek input_0_av_sync now 2500"]
    with pytest.raises(ConfigError, match="position_ms"):
        controller.media({"id": "cam1", "action": "seek"})


def test_playback_command_retries_until_a_frame_exists():
    attempts = {"n": 0}
    commands = []

    def run(text):
        commands.append(text)
        if text.startswith("pause ") and attempts["n"] < 2:
            attempts["n"] += 1
            raise RuntimeError("Relative seek not possible. Can not determine current time.")

    controller, _ = _controller(run=run)
    controller.media({"id": "cam1", "action": "pause"})
    # Nothing between the tries: a failed pause leaves the team as it was, and a resume here
    # would start a team the operator had paused.
    assert commands == ["pause input_0_pause now"] * 3


def test_another_refusal_is_not_tried_again():
    commands = []

    def run(text):
        commands.append(text)
        raise RuntimeError("seek: 500 ERROR: Seek target is before the first frame")

    controller, _ = _controller(run=run)
    with pytest.raises(RuntimeError, match="before the first frame"):
        controller.media({"id": "cam1", "action": "seek", "position_ms": 10})
    assert commands == ["seek input_0_av_sync now 10"]


@pytest.mark.parametrize("command, request_", [
    ("prepare", {"url": "file:///clip.mp4", "loop": False}),
    ("media", {"action": "pause"}),
    ("resume_at", {"at_ms": 1000}),
])
def test_playback_without_a_control_port_is_refused_before_anything_changes(command, request_):
    """The retry needs the engine's refusal, and the binding only logs one."""
    avp = _Avp([_Node("input_0", dst="out")])
    controller = SourceController(avp, {"cam1": _pad()})
    with pytest.raises(ConfigError, match="remote_control_port"):
        getattr(controller, command)({"id": "cam1", **request_})
    assert avp.commands == [] and not avp._group.stop_asked


class _ControlPort(socketserver.ThreadingTCPServer):
    """AVPlumber's line protocol: a greeting, then per line the reply of `answers` or 200 OK."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, answers):
        self.lines, self.answers = [], answers
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write(b"100 VTR Ready\n")
            for raw in self.rfile:
                line = raw.decode().rstrip("\n")
                self.server.lines.append(line)
                self.wfile.write(self.server.answers.get(line.split(" ", 1)[0], "200 OK\n").encode())


def test_control_command_returns_the_reply_and_raises_the_engines_refusal():
    server = _ControlPort({"version": "201 OK\navplumber 1\n\n",
                           "pause": "500 ERROR: Relative seek not possible. Can not determine current time.\n"})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        assert control_command(port, "resume input_0_pause") is None
        assert control_command(port, "version") == "avplumber 1\n"
        with pytest.raises(RuntimeError, match="^pause: 500 ERROR: .*Can not determine current time"):
            control_command(port, "pause input_0_pause now")
        assert server.lines == ["resume input_0_pause", "version", "pause input_0_pause now"]
    finally:
        server.shutdown()
        server.server_close()


def test_resume_at_schedules_the_pause_team():
    commands = []
    controller, _ = _controller(run=commands.append)
    assert controller.resume_at({"id": "cam1", "at_ms": 90_000}) == {"id": "cam1", "at_ms": 90_000}
    assert commands == ["resume input_0_pause at 90000"]
    with pytest.raises(ConfigError, match="unsupported fields"):   # the engine tracks no playhead to set
        controller.resume_at({"id": "cam1", "at_ms": 90_000, "position_ms": 1500})
    assert commands == ["resume input_0_pause at 90000"]


def test_register_wires_prepare_media_and_resume_at():
    commands = []
    controller, _ = _controller(run=commands.append)
    register_source_commands(controller.avp, controller)
    assert set(controller.avp.handlers) >= {
        "mixer.source.set", "mixer.source.prepare", "mixer.source.media", "mixer.source.resume_at"}
    prepared = json.loads(controller.avp.handlers["mixer.source.prepare"](json.dumps(
        {"id": "cam1", "url": "file:///a.mp4", "loop": True, "cue_in_ms": 0})))
    assert prepared["duration_ms"] == 12_345 and prepared["id"] == "cam1"
