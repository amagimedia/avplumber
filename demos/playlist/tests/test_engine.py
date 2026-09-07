"""Native command sequences the engine issues, against a fake AVPlumber."""

import pytest

import engine as eng
from engine import PlaylistConfig, PlaylistEngine, chain_edge, hms, slot_pause_team
from helpers import clips
from playlist import BackendEvent, ElementMode, Transition


class Edge:
    def __init__(self, owner):
        self.owner = owner

    @property
    def enqueued_total(self):
        return self.owner.frames          # grows when a group starts


class Group:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def startNodes(self):
        self.owner.log.append(f"group.start {self.name}")
        self.owner.working = True
        self.owner.frames += 1


class FakeAvp:
    def __init__(self):
        self.nodes, self.log, self.ready, self.working, self.frames = [], [], False, True, 1
        self.edges = type("E", (), {"planCapacity": lambda *_: None})()

    def addNode(self, node):
        self.nodes.append(node.p)

    def executeCommandsFromString(self, s):
        self.log.extend(s.split("\n"))
        if "group.stop" in s:
            self.working = False

    def enableControlServer(self, port):
        self.log.append(f"control {port}")

    def setLogFile(self, _):
        pass

    def setReady(self):
        self.ready = True

    def node(self, _):
        return type("Node", (), {"isWorking": self.working})()

    def getEdge(self, _):
        return Edge(self)

    def group(self, name):
        return Group(self, name)

    def shutdown(self):
        self.log.append("shutdown")


class FakeMixer:
    def __init__(self, avp, **kw):
        self.avp, self.kw, self.sources, self.scenes, self.log = avp, kw, [], {}, avp.log

    def add_source(self, name, **kw):
        self.sources.append((name, kw))

    def add_scene(self, name, sources, **_):
        self.scenes[name] = sources

    def set_initial_scene(self, scene, slot="A"):
        self.initial = (scene, slot)

    def build(self):
        return "mixer_final_out"

    def cut(self, scene, start_pts_ms=-1):
        self.log.append(f"mixer.cut {scene} {start_pts_ms}")

    def fade(self, scene, duration_sec, start_pts_ms=-1):
        self.log.append(f"mixer.fade {scene} {duration_sec} {start_pts_ms}")

    def wipe(self, scene, wipe_file, duration_sec, start_pts_ms=-1):
        self.log.append(f"mixer.wipe {scene} {wipe_file} {duration_sec} {start_pts_ms}")

    def initialize_routes(self): self.log.append("init_routes")
    def start_groups(self): self.log.append("start_groups")
    def begin_transition_preheat(self): self.log.append("preheat")
    def finish_transition_preheat(self): self.log.append("preheat done")
    def start_output(self): self.log.append("start_output")


def node_type(kind):
    return type(kind, (), {"__init__": lambda self, p: setattr(self, "p", {"type": kind, **p})})


class Listener:
    def __init__(self, **_): pass
    def start(self): pass
    def stop(self): pass


def api():
    from types import SimpleNamespace
    names = {"InputRec": "input_rec", "Demux": "demux", "DecVideo": "dec_video", "Split": "split",
             "SpeedVideo": "speed_video", "Pause": "pause", "Realtime": "realtime", "ForceFPS": "force_fps",
             "ForceKeyFrame": "force_key_frame", "AssumeVideoFormat": "assume_video_format",
             "EncVideo": "enc_video", "Bsf": "bsf", "Mux": "mux", "Output": "output"}
    return SimpleNamespace(MixerGraphBuilder=FakeMixer, RtcpFeedbackListener=Listener,
                           **{k: node_type(v) for k, v in names.items()})


class Clock:
    now = 100_000


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setattr(eng, "probe_length_ms", lambda url: 10_000)
    monkeypatch.setattr(eng, "now_ms", lambda: Clock.now)
    Clock.now = 100_000
    avp = FakeAvp()
    e = PlaylistEngine(avp, api(), PlaylistConfig(control_port=0, log_file="", preroll_ms=40))
    e.build(clips("a")[0])
    return e, avp


def cue(e, clip, request_id, at_ms, transition=Transition.CUT, transition_ms=0):
    e._handle(eng._Task("cue", clip.item_id, clip, request_id, at_ms, transition, transition_ms))


def events(e):
    return [ev for ev in e.poll_events() if ev.kind != "health"]


def test_build_registers_sixteen_fullscreen_scenes_and_the_replay_style_chain(built):
    e, avp = built
    assert len(e.mixer.sources) == 16 and e.mixer.sources[3][1]["pre_otm_edge"] == chain_edge(3)
    assert e.mixer.scenes["item_7"] == {"source_7": {"dst_x": 0, "dst_y": 0, "dst_w": 1920, "dst_h": 1080, "fit": "contain"}}
    assert e.mixer.initial == ("item_0", "A")
    by_name = {n["name"]: n for n in avp.nodes}
    chain = [n["name"] for n in avp.nodes if n["name"].endswith("_pl0")]
    assert chain == ["input_pl0", "demux_pl0", "decode_pl0", "speed_pl0", "pause_pl0", "realtime_pl0", "fps_pl0"]
    assert by_name["input_pl0"]["pause_team"] == slot_pause_team(0) and by_name["input_pl0"]["loop"] is True
    assert by_name["input_pl0"]["start_ts"] == "00:00:00.000" and "auto_restart" not in by_name["decode_pl0"]
    assert by_name["pause_pl0"] == {"type": "pause", "name": "pause_pl0", "src": "input_pl0_speeded",
                                    "dst": "input_pl0_paused", "team": "pl_item_0_pause",
                                    "sync_team": "pl_item_0_sync", "group": "pl_item_0"}
    assert by_name["realtime_pl0"]["team"] == "pl_item_0_sync" and by_name["realtime_pl0"]["set_pts"] is True
    assert by_name["realtime_pl0"]["tick_period"] == "1/30" and by_name["speed_pl0"]["sync_node"] == "realtime_pl0"
    assert by_name["janus_encoder"]["codec"] == "h264_nvenc" and by_name["janus_rtp_output"]["format"] == "rtp"


def test_start_follows_the_mixer_preheat_order(built):
    e, avp = built
    e.start()
    e.close()
    wanted = ("group.start pl_item_0", "init_routes", "start_groups", "preheat", "preheat done",
              "start_output", "group.start output", "shutdown")
    assert [l for l in avp.log if l in wanted] == list(wanted)


def test_scheduled_cue_primes_and_parks_then_arms_shortly_before_the_cut(built):
    e, avp = built
    b = clips(("b", ElementMode.PLAY_TO_END, {"play_from_ms": 2000}))[0]
    avp.log.clear()
    cue(e, b, 7, 130_000)
    # prime: run to first frame, then pause + seek through the sync team
    assert avp.log == ["group.start pl_item_1", "pause pl_item_1_pause now",
                       f"seek pl_item_1_sync now {hms(2000)}"]
    assert e._armed.issued is False                      # 30 s ahead: not armed yet
    Clock.now = 129_000
    e._tick()
    assert e._armed.issued is False
    Clock.now = 129_500                                   # inside arm_lead (600 ms)
    e._tick()
    assert avp.log[-2:] == ["mixer.cut item_1 130000", "resume pl_item_1_pause at 129960"]
    Clock.now = 130_010
    e._tick()                                             # timer confirmation without a control port
    assert events(e) == []
    Clock.now = 130_040
    e._tick()
    assert events(e) == [BackendEvent("on_air", b.item_id, 7, at_ms=130_000)]
    assert e._pgm_slot == 1 and e._armed is None


def test_immediate_cue_lets_the_mixer_pick_the_start_and_resumes_now(built):
    e, avp = built
    b = clips("b")[0]
    cue(e, b, 1, None)
    assert avp.log[-2:] == ["mixer.cut item_1 -1", "resume pl_item_1_pause at 100060"]
    assert (e._armed.start, e._armed.end) == (100_100, 100_100)   # mixer: now + switch margin


def test_fade_starts_early_enough_to_end_on_time_and_wipe_needs_a_file(built):
    e, avp = built
    b, c = clips("b", "c")
    cue(e, b, 1, 100_500, Transition.FADE, 800)          # would have to start in the past ...
    assert avp.log[-2:] == ["mixer.fade item_1 0.8 -1", "resume pl_item_1_pause at 100060"]
    assert (e._armed.start, e._armed.end) == (100_100, 100_900)   # ... so it starts at now + margin
    Clock.now = 101_000
    e._tick()
    events(e)
    cue(e, c, 2, 105_000, Transition.FADE, 800)
    Clock.now = 103_700
    e._tick()
    assert avp.log[-2:] == ["mixer.fade item_2 0.8 104200", "resume pl_item_2_pause at 104160"]
    with pytest.raises(RuntimeError, match="wipe-file"):
        cue(e, clips("d")[0], 3, None, Transition.WIPE, 500)


def test_superseding_an_armed_element_interrupts_and_parks_it(built):
    e, avp = built
    b, c = clips("b", "c")
    cue(e, b, 1, None)                                     # issued
    avp.log.clear()
    cue(e, c, 2, None)
    assert avp.log[0] == 'mixer.interrupt {"mixer":"mixer"}'
    assert avp.log[1:3] == ["pause pl_item_1_pause now", f"seek pl_item_1_sync now {hms(0)}"]
    assert e._armed.item_id == c.item_id


def test_park_of_an_armed_element_disarms_and_re_cues_it(built):
    e, avp = built
    b = clips(("b", ElementMode.PLAY_TO_END, {"play_from_ms": 1500}))[0]
    cue(e, b, 1, 130_000)
    assert e._armed is not None
    avp.log.clear()
    e._handle(eng._Task("park", b.item_id))
    assert e._armed is None and "mixer.interrupt" not in " ".join(avp.log)   # never issued
    assert avp.log == ["pause pl_item_1_pause now", f"seek pl_item_1_sync now {hms(1500)}"]
    e._handle(eng._Task("pause", b.item_id)); e._handle(eng._Task("resume", b.item_id))
    assert avp.log[-2:] == ["pause pl_item_1_pause now", "resume pl_item_1_pause"]
    e._handle(eng._Task("remove", b.item_id))
    assert "node.delete input_pl1" in avp.log and 1 not in e._bound


def test_failed_arm_interrupts_and_parks_the_incoming_chain(built):
    e, avp = built
    b = clips("b")[0]
    cue(e, b, 1, None)
    avp.log.clear()
    e._handle(eng._Task("graph_error", "decode_pl1", message="decode_pl1 (dec_video): boom"))
    assert avp.log[0] == 'mixer.interrupt {"mixer":"mixer"}' and avp.log[1] == "pause pl_item_1_pause now"
    assert events(e) == [BackendEvent("failed", b.item_id, 1, message="decode_pl1 (dec_video): boom")]
    assert e._armed is None


def test_cue_to_the_program_slot_restarts_it_from_cue_in(built):
    e, avp = built
    a = clips("a")[0]
    avp.log.clear()
    cue(e, a, 5, None)
    assert avp.log == ["pause pl_item_0_pause now", f"seek pl_item_0_sync now {hms(0)}", "resume pl_item_0_pause"]
    assert events(e) == [BackendEvent("on_air", a.item_id, 5, at_ms=100_000)]


def test_removing_the_program_element_frees_its_slot_after_the_next_switch(built):
    e, avp = built
    a, b = clips("a", "b")
    e._handle(eng._Task("remove", a.item_id))
    assert 0 in e._orphaned and 0 in e._bound
    cue(e, b, 1, None)
    Clock.now = 100_200
    e._tick()
    assert e._pgm_slot == 1 and 0 not in e._bound and "node.delete input_pl0" in avp.log


def test_edited_element_rebuilds_its_chain_in_place(built):
    e, avp = built
    b = clips("b")[0]
    cue(e, b, 1, None)
    edited = clips(("b", ElementMode.PLAY_TO_END, {"speed": 2.0}))[0]
    avp.log.clear()
    cue(e, edited, 2, None)
    assert avp.log[0] == 'mixer.interrupt {"mixer":"mixer"}' and avp.log[1] == "group.stop pl_item_1"
    assert "node.delete pause_pl1" in avp.log
    assert [n["speed"] for n in avp.nodes if n["name"] == "speed_pl1"][-1] == 2.0


def test_record_option_splits_the_program_before_janus(monkeypatch):
    monkeypatch.setattr(eng, "probe_length_ms", lambda url: 10_000)
    avp = FakeAvp()
    e = PlaylistEngine(avp, api(), PlaylistConfig(control_port=0, log_file="", record="/tmp/p.mp4"))
    e.build(clips("a")[0])
    by_name = {n["name"]: n for n in avp.nodes}
    assert by_name["program_split"]["dst"] == ["program_janus", "program_record"]
    assert by_name["janus_fps"]["src"] == "program_janus"
    assert by_name["record_output"]["url"] == "/tmp/p.mp4" and by_name["record_output"]["format"] == "mp4"


def test_hms():
    assert hms(3_723_004) == "01:02:03.004"
