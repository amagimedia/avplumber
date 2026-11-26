"""mixer.source.set: safe group stop + URL/loop repoint without a live engine."""
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
    assert pads["c"].kind == "browser" and pads["c"].sync_team is None
    assert pads["d"].kind == "vmm"


def test_register_source_commands_wires_the_control_handler():
    avp = _Avp([_Node("input_0", dst="out")])
    register_source_commands(avp, SourceController(avp, {"cam1": _pad(sync_team=None, pause_team=None)}))
    reply = json.loads(avp.handlers["mixer.source.set"](json.dumps(
        {"id": "cam1", "url": "file:///a.mp4", "loop": False})))
    assert reply == {"id": "cam1", "url": "file:///a.mp4", "loop": False}
