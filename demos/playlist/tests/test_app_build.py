"""Live builder and lifecycle against a host-independent AVPlumber fake."""

import json
import time
import types

from helpers import clips
from playlist import SWITCHER_TYPE, item_node_names


class FakeWrapper:
    def __init__(self):
        self.isWorking = False

    def stop(self, _wait):
        self.isWorking = False


class FakeNodeObject:
    def __init__(self, parameters):
        self.parameters = parameters
        self._wrapper = FakeWrapper()


class FakePythonNode(FakeNodeObject):
    pass


class FakeEdge:
    def wait_peek(self, _timeout_ms):
        return object()


class FakeEdges:
    def __init__(self):
        self.plans = []

    def planCapacity(self, pattern, capacity):
        self.plans.append((pattern, capacity))


class FakeGroup:
    def __init__(self, name, avp):
        self.name = name
        self.avp = avp

    def stopNodes(self):
        self.avp.commands.append(f"direct-stop:{self.name}")


class FakeWorkingNode:
    isWorking = True


class FakeAvp:
    def __init__(self):
        self.commands = []
        self.added = []
        self.edges = FakeEdges()
        self.backend = None
        self.on_exception = None
        self.ready = False
        self.shutdown_called = False
        self.log_files = []

    def setLogFile(self, value):
        self.log_files.append(value)

    def executeCommandsFromString(self, command):
        self.commands.append(command)
        if command.startswith("node.add "):
            self.added.append(json.loads(command[len("node.add "):]))
        if command == "group.start switch":
            for node in self.added:
                if node.get("group") == "switch" and "object" in node:
                    node["object"]._wrapper.isWorking = True
        if command.startswith("node.object.set pl_switcher active"):
            self.backend.observe_frame({})
            self.backend.observe_frame({})

    def addNode(self, node):
        parameters = dict(node.parameters)
        parameters["object"] = node
        self.added.append(parameters)

    def getEdge(self, _name):
        return FakeEdge()

    def node(self, _name):
        return FakeWorkingNode()

    def group(self, name):
        return FakeGroup(name, self)

    def setReady(self):
        self.ready = True

    def setExceptionCallback(self, _callback):
        pass

    def shutdown(self):
        self.shutdown_called = True


class FakeListener:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def fake_api():
    def factory(type_name):
        class Node(FakeNodeObject):
            TYPE = type_name

            def __init__(self, parameters):
                super().__init__({"type": type_name, **parameters})
        return Node

    return types.SimpleNamespace(
        AVPlumber=FakeAvp,
        Bsf=factory("bsf"),
        EncVideo=factory("enc_video"),
        ForceKeyFrame=factory("force_keyframe"),
        Mux=factory("mux"),
        Output=factory("output"),
        PythonNode=FakePythonNode,
        RtcpFeedbackListener=FakeListener,
        by_type={name: factory(name) for name in (
            "input_rec", "demux", "dec_video", "speed_video", "force_fps",
            "pause", SWITCHER_TYPE, "realtime<av::VideoFrame>")},
    )


def build(tmp_path):
    import playlist_app
    config = playlist_app.PlaylistConfig(
        clips=clips("a", "b", "c", "d", "e"),
        log_file=str(tmp_path / "playlist.log"),
        control_timeout=1,
    )
    app = playlist_app.build_playlist_application(config, api=fake_api())
    app.avp.backend = app.backend
    return app


def added_by_name(app):
    return {node["name"]: node for node in app.avp.added}


def test_builder_adds_one_initial_item_stable_switcher_and_replay_output(tmp_path):
    app = build(tmp_path)
    nodes = added_by_name(app)
    for name in item_node_names(0):
        assert name in nodes
    for name in ("pl_switcher", "pl_realtime", "pl_position",
                 "janus_force_keyframe", "janus_encoder", "janus_headers",
                 "janus_mux", "janus_rtp_output"):
        assert name in nodes
    assert nodes["pl_switcher"]["type"] == SWITCHER_TYPE
    assert "repeat_on_stall" not in repr(app.avp.added)
    assert "sentinel" not in repr(app.avp.added).lower()
    assert app.avp.edges.plans == [("*", 1)]
    assert app.avp.log_files


def test_start_reaches_source_ready_before_janus_ready(tmp_path):
    app = build(tmp_path)
    app.start()
    status = app.controller.status()
    assert status.playing and status.output_alive
    assert app.avp.ready is True
    assert app.rtcp_feedback_listener.started is True
    commands = app.avp.commands
    assert commands.index("group.start pl_item_0") < commands.index("group.start switch")
    assert commands.index("group.start switch") < commands.index("group.start output")
    app.stop()


def test_stop_is_bounded_and_tears_down_permanent_groups_only_at_shutdown(tmp_path):
    app = build(tmp_path)
    app.start()
    app.controller.stop()
    deadline = time.monotonic() + 1
    while "group.stop pl_item_0" not in app.avp.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "group.stop output" not in app.avp.commands
    app.stop()
    assert "direct-stop:output" in app.avp.commands
    assert "direct-stop:switch" in app.avp.commands
    assert app.avp.shutdown_called
