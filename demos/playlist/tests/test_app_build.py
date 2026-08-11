"""Exercise the live builder (playlist_app.build_playlist_application) against a
fake AVPlumber, so the graph wiring/lifecycle is covered without a GPU."""
import types

import pytest

from helpers import clips


class FakeGroup:
    def __init__(self, name, backend):
        self.name, self.backend = name, backend
    def startNodes(self):
        self.backend.started.append(self.name)
    def stopNodes(self):
        self.backend.stopped.append(self.name)


class FakeNode:
    isWorking = True


class FakeAvp:
    def __init__(self):
        self.commands, self.added = [], []
        self.started, self.stopped = [], []
        self.groups = {}
        self.ready = False
        self.edges = types.SimpleNamespace(planCapacity=lambda *a: None)
        self.on_exception = None
    def executeCommandsFromString(self, cmd):
        self.commands.append(cmd)
        if cmd.startswith("node.add "):
            import json
            self.added.append(json.loads(cmd[len("node.add "):])["name"])
    def addNode(self, node):
        self.added.append(getattr(node, "_name", "python_node"))
    def group(self, name):
        return self.groups.setdefault(name, FakeGroup(name, self))
    def node(self, name):
        return FakeNode()
    def setReady(self):
        self.ready = True
    def setExceptionCallback(self, cb):
        pass
    def shutdown(self):
        pass


class FakeProbe:
    worker_running = False
    def __init__(self, params, controller):
        self._name = params["name"]
        self.controller = controller
    def request_stop(self): pass
    def detach(self): pass


class FakeListener:
    def __init__(self, **kw): self.kw = kw
    def start(self): pass
    def stop(self): pass


def fake_api():
    class _NT:
        def __init__(self, type_):
            self.type_ = type_
        def __call__(self, params):
            n = types.SimpleNamespace(**params)
            n._name = params["name"]
            return n
    by_type = {t: _NT(t) for t in (
        "input_rec", "demux", "dec_video", "speed_video", "rescale_video",
        "force_fps", "pause", "source_switcher", "realtime<av::VideoFrame>")}
    return types.SimpleNamespace(
        AVPlumber=FakeAvp, Bsf=_NT("bsf"), EncVideo=_NT("enc_video"),
        ForceKeyFrame=_NT("force_keyframe"), Mux=_NT("mux"), Output=_NT("output"),
        PositionProbe=FakeProbe, RtcpFeedbackListener=FakeListener, by_type=by_type)


def _build(**kw):
    import playlist_app as pa
    cfg = pa.PlaylistConfig(clips=clips("a", "b", "c"), **kw)
    return pa, pa.build_playlist_application(cfg, api=fake_api())


def test_build_adds_worker0_switcher_and_output():
    _, app = _build()
    added = set(app.avp.added)
    # worker 0 chain
    for n in ("worker0_input", "worker0_pause"):
        assert n in added
    # switcher + shared realtime + probe + janus output
    for n in ("pl_switcher", "pl_realtime", "pl_position", "janus_encoder",
              "janus_rtp_output"):
        assert n in added


def test_start_releases_worker0_and_reaches_ready():
    _, app = _build()
    # make readiness happen as soon as worker0 is released
    orig_play = app.controller.play
    def play_then_ready():
        orig_play()
        app.controller.ready = True
    app.controller.play = play_then_ready
    app.start()
    assert app.avp.ready is True
    assert "worker0" in app.avp.started and "output" in app.avp.started
    assert f"resume worker0_pauseteam" in app.avp.commands


def test_stop_tears_down_groups():
    _, app = _build()
    app.controller.ready = True
    app.controller.play = lambda: setattr(app.controller, "ready", True)
    app.start()
    app.stop()
    assert "output" in app.avp.stopped
    assert "worker0" in app.avp.stopped
