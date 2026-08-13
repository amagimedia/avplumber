"""Live builder and lifecycle against a host-independent AVPlumber fake."""

import json
import time
import types

from helpers import clips
from playlist import ElementMode, SWITCHER_TYPE, item_node_names


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
    def __init__(self):
        self.callbacks = []

    def addWiretapCallback(self, callback):
        self.callbacks.append(callback)

    def emit(self, value=None):
        value = object() if value is None else value
        for callback in self.callbacks:
            callback(value)

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
    def __init__(self, avp, name):
        self.avp = avp
        self.name = name

    @property
    def isWorking(self):
        return self.avp.node_states.get(
            self.name, self.name == "janus_rtp_output")


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
        self.edge_objects = {}
        self.node_states = {}

    def setLogFile(self, value):
        self.log_files.append(value)

    def executeCommandsFromString(self, command):
        self.commands.append(command)
        if command.startswith("node.add "):
            node = json.loads(command[len("node.add "):])
            self.added.append(node)
            self.node_states[node["name"]] = False
        if command.startswith("group.start "):
            group = command[len("group.start "):]
            for node in self.added:
                if node.get("group") == group:
                    self.node_states[node["name"]] = True
        if command.startswith("group.stop "):
            group = command[len("group.stop "):]
            for node in self.added:
                if node.get("group") == group:
                    self.node_states[node["name"]] = False
        if command == "group.start switch":
            for node in self.added:
                if node.get("group") == "switch" and "object" in node:
                    node["object"]._wrapper.isWorking = True
        if command.startswith("resume pl_item_"):
            slot = command.split("_", 3)[2]
            self.getEdge(f"pl_item_{slot}_normalized").emit()
        if command.startswith("node.object.set pl_switcher active"):
            self.backend.observe_frame({})
            self.backend.observe_frame({})

    def addNode(self, node):
        parameters = dict(node.parameters)
        parameters["object"] = node
        self.added.append(parameters)
        self.node_states[parameters["name"]] = False

    def getEdge(self, name):
        return self.edge_objects.setdefault(name, FakeEdge())

    def node(self, name):
        return FakeWorkingNode(self, name)

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
            SWITCHER_TYPE, "realtime<av::VideoFrame>")},
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


def test_selected_switch_edge_reports_eof_before_realtime_consumes_it(tmp_path):
    app = build(tmp_path)
    app.backend._probe_item = app.controller.clips[0].item_id
    app.avp.getEdge("pl_switched").emit(types.SimpleNamespace(width=0))
    events = app.backend.poll_events()
    assert [(event.kind, event.item_id) for event in events] == [
        ("eof", app.controller.clips[0].item_id)]


def test_start_preheats_first_source_between_switch_and_permanent_output(tmp_path):
    app = build(tmp_path)
    app.start()
    status = app.controller.status()
    assert status.playing and status.output_alive
    assert app.avp.ready is True
    assert app.rtcp_feedback_listener.started is True
    commands = app.avp.commands
    switch = commands.index("group.start switch")
    source = commands.index("group.start pl_item_0")
    output = commands.index("group.start output")
    pauses = [index for index, command in enumerate(commands)
              if command == "pause pl_item_0_pause_team now"]
    resumes = [index for index, command in enumerate(commands)
               if command == "resume pl_item_0_pause_team"]
    select = commands.index("node.object.set pl_switcher active 0")
    assert (switch < pauses[0] < source < resumes[0] < pauses[1]
            < output < resumes[1] < select)
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


def test_repeated_active_mode_edits_do_not_exhaust_switcher_slots(tmp_path):
    app = build(tmp_path)
    app.start()
    try:
        for edit in range(24):
            mode = (ElementMode.LOOP_SELF if edit % 2 == 0
                    else ElementMode.PLAY_TO_END)
            app.controller.set_element_mode(0, mode)
            deadline = time.monotonic() + 1
            while app.controller.status().pending_index is not None:
                app.controller.poll(int(time.monotonic() * 1000))
                if time.monotonic() >= deadline:
                    raise AssertionError(f"mode edit {edit} did not finish")
                time.sleep(0.005)
            status = app.controller.status()
            assert status.playing
            assert status.active_index == 0
            assert status.error == ""
        assert not [command for command in app.avp.commands
                    if command.startswith("speed.set ")]
    finally:
        app.stop()
