"""Graph plans use only existing nodes and stable per-element switcher slots."""

from playlist import (DEFAULT_SLOT_CAPACITY, SWITCHER_TYPE, Clip,
                      ElementMode as E, item_edge, item_group,
                      item_node_names, item_pause_team, plan_item_nodes,
                      plan_switch_nodes)


def test_item_chain_matches_existing_replay_video_path():
    specs = plan_item_nodes(3, Clip(url="/media/a.mp4"), fps=30)
    assert [spec.type for spec in specs] == [
        "input_rec", "demux", "dec_video", "speed_video", "force_fps"]
    assert [spec.name for spec in specs] == item_node_names(3)
    assert all(spec.group == item_group(3) for spec in specs)
    assert specs[0].params["pause_team"] == item_pause_team(3)
    assert specs[-1].params["dst"] == item_edge(3, "normalized")


def test_item_chain_applies_cues_speed_and_only_native_force_fps_parameters():
    clip = Clip(url="/media/a.mp4", play_from_ms=1250, play_to_ms=8750,
                speed=1.25)
    specs = plan_item_nodes(0, clip, fps=30)
    source = specs[0]
    speed = next(spec for spec in specs if spec.type == "speed_video")
    force_fps = next(spec for spec in specs if spec.type == "force_fps")
    assert source.params["start_ts"] == "00:00:01.250"
    assert source.params["stop_ts"] == "00:00:08.750"
    assert speed.params["speed"] == 1.25
    assert force_fps.params == {
        "src": item_edge(0, "speeded"),
        "dst": item_edge(0, "normalized"),
        "fps": "30/1",
    }
    assert "repeat_on_stall" not in force_fps.params


def test_only_loop_self_enables_input_looping():
    for mode, expected in (
        (E.PLAY_TO_END, False), (E.TIMED, False), (E.LOOP_SELF, True)):
        kwargs = {"duration_ms": 1000} if mode is E.TIMED else {}
        source = plan_item_nodes(
            0, Clip(url="/media/a", element_mode=mode, **kwargs))[0]
        assert source.params["loop"] is expected


def test_permanent_switch_path_uses_existing_cpp_switcher_and_realtime_only():
    specs = plan_switch_nodes(DEFAULT_SLOT_CAPACITY, fps=30)
    assert [spec.type for spec in specs] == [
        SWITCHER_TYPE, "realtime<av::VideoFrame>"]
    assert all(spec.group == "switch" for spec in specs)
    switcher, realtime = specs
    assert switcher.params == {
        "src": [item_edge(slot, "normalized")
                for slot in range(DEFAULT_SLOT_CAPACITY)],
        "dst": "pl_switched",
        "active": 0,
    }
    assert realtime.params["src"] == "pl_switched"
    assert realtime.params["dst"] == "pl_realtime_out"


def test_plan_contains_no_sentinel_black_or_framework_extension():
    specs = plan_item_nodes(0, Clip(url="/media/a")) + plan_switch_nodes()
    serialized = repr(specs).lower()
    assert "sentinel" not in serialized
    assert "fallback_active" not in serialized
    assert "repeat_on_stall" not in serialized
