"""mixer.source.set / prepare / media / resume_at without a live engine."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pyplumber.mixer.config import ConfigError
from pyplumber.mixer.source_control import (
    SourceController, SourcePad, group_shape, register_source_commands, source_pads, stop_group,
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
    def __init__(self, nodes, edges=None):
        self._group = _Group(nodes)
        self._edges = {name: _Edge() for name in (edges or [])}
        self.commands = []
        self.handlers = {}

    def group(self, _name):
        return self._group

    def getEdge(self, name, data_type=None):
        return self._edges[name]

    def executeCommandsFromString(self, command):
        self.commands.append(command)

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


def test_unsupported_kind_is_refused():
    avp = _Avp([])
    controller = SourceController(avp, {"page": SourcePad(
        id="page", index=1, kind="browser", group="input_1", input_node="input_1")})
    with pytest.raises(ConfigError, match="unsupported"):
        controller.set({"id": "page", "url": "https://example", "loop": False})


def test_source_pads_follow_build_application_naming():
    sources = [SimpleNamespace(id="a", kind="video", has_audio=True),
               SimpleNamespace(id="b", kind="ndi", has_audio=False),
               SimpleNamespace(id="c", kind="browser", has_audio=False),
               SimpleNamespace(id="d", kind="video", has_audio=False)]
    pads = source_pads(sources, external={"d"})
    assert pads["a"].group == "input_0" and pads["a"].sync_team == "input_0_av_sync"
    assert pads["b"].sync_team == "input_1_av_sync" and pads["b"].kind == "ndi"
    assert pads["c"].kind == "browser" and pads["c"].sync_team is None and pads["c"].pause_team is None
    assert pads["d"].kind == "vmm"
    assert pads["d"].sync_team is None and pads["a"].pause_team is None
    play = source_pads(sources, playback_control=True)
    assert play["b"].pause_team is None and play["b"].sync_team == "input_1_av_sync"   # NDI: live, no playback
    assert play["d"].sync_team == "input_3_av_sync" and play["d"].pause_team == "input_3_pause"
    assert play["c"].pause_team is None and play["c"].sync_team is None   # a browser pad is no build_input chain


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
    assert commands == [
        "pause input_0_pause now", "resume input_0_pause",
        "pause input_0_pause now", "resume input_0_pause",
        "pause input_0_pause now",
    ]


def test_resume_at_schedules_the_pause_team():
    commands = []
    controller, _ = _controller(run=commands.append)
    assert controller.resume_at({"id": "cam1", "at_ms": 90_000, "position_ms": 1500}) == {
        "id": "cam1", "at_ms": 90_000, "position_ms": 1500}
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
