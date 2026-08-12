"""Stable-slot asynchronous backend behavior and responsiveness."""

import time

import pytest

from helpers import clips
from playlist import Clip, item_group, item_pause_team
from playlist_app import AsyncPlaylistBackend, PlaylistConfig
from test_app_build import FakeAvp, FakeListener, fake_api


def backend_for(clip_list=None, avp=None):
    playlist = clip_list or clips("a", "b", "c", "d", "e")
    avp = avp or FakeAvp()
    config = PlaylistConfig(
        clips=playlist, control_timeout=1, log_file="", slot_capacity=8)
    backend = AsyncPlaylistBackend(avp, fake_api(), config)
    backend.register_built_item(playlist[0].item_id, 0, playlist[0])
    backend.bind_listener(FakeListener())
    avp.backend = backend
    backend.start()
    return backend, avp, playlist


def wait_events(backend, kind, timeout=1):
    deadline = time.monotonic() + timeout
    found = []
    while time.monotonic() < deadline:
        found.extend(backend.poll_events())
        matches = [event for event in found if event.kind == kind]
        if matches:
            return matches, found
        time.sleep(0.005)
    raise AssertionError(f"no {kind} event; got {found}")


def activate(backend, clip, request=1):
    backend.play_item(request, clip.item_id, clip)
    events, _ = wait_events(backend, "ready")
    assert events[-1].request_id == request


def test_public_activation_returns_immediately_while_worker_is_slow():
    class SlowAvp(FakeAvp):
        def executeCommandsFromString(self, command):
            if command.startswith("group.start pl_item_"):
                time.sleep(0.15)
            super().executeCommandsFromString(command)

    backend, _, playlist = backend_for(avp=SlowAvp())
    started = time.monotonic()
    backend.play_item(1, playlist[0].item_id, playlist[0])
    elapsed = time.monotonic() - started
    assert elapsed < 0.03
    wait_events(backend, "ready")
    backend.close()


def test_destination_is_readied_then_resumed_and_selected_before_old_stops():
    backend, avp, playlist = backend_for()
    activate(backend, playlist[0], 1)
    avp.commands.clear()
    activate(backend, playlist[1], 2)
    destination = backend._item_slots[playlist[1].item_id]
    commands = avp.commands
    pauses = [index for index, command in enumerate(commands)
              if command == f"pause {item_pause_team(destination)} now"]
    resumes = [index for index, command in enumerate(commands)
               if command == f"resume {item_pause_team(destination)}"]
    start = commands.index(f"group.start {item_group(destination)}")
    flip = commands.index(f"node.object.set pl_switcher active {destination}")
    stop_old = commands.index("group.stop pl_item_0")
    assert pauses[0] < start < resumes[0] < pauses[1] < resumes[1] < flip < stop_old
    assert "group.stop output" not in commands
    backend.close()


def test_stop_pause_and_resume_are_item_addressed_and_never_stop_output():
    backend, avp, playlist = backend_for()
    activate(backend, playlist[0])
    avp.commands.clear()
    item_id = playlist[0].item_id
    backend.pause_item(item_id)
    backend.resume_item(item_id)
    backend.stop_item(item_id)
    deadline = time.monotonic() + 1
    while "group.stop pl_item_0" not in avp.commands and time.monotonic() < deadline:
        time.sleep(0.005)
    assert avp.commands == [
        "pause pl_item_0_pause_team now",
        "resume pl_item_0_pause_team",
        "group.stop pl_item_0",
    ]
    assert backend.output_alive() is True
    backend.close()


def test_superseded_activation_never_reports_stale_item_ready():
    class SlowAvp(FakeAvp):
        def executeCommandsFromString(self, command):
            if command.startswith("group.start pl_item_0"):
                time.sleep(0.05)
            super().executeCommandsFromString(command)

    backend, _, playlist = backend_for(avp=SlowAvp())
    backend.play_item(1, playlist[0].item_id, playlist[0])
    backend.play_item(2, playlist[1].item_id, playlist[1])
    ready, all_events = wait_events(backend, "ready")
    assert [(event.item_id, event.request_id) for event in ready] == [
        (playlist[1].item_id, 2)]
    assert not [event for event in all_events
                if event.kind == "ready" and event.request_id == 1]
    backend.close()


def test_remove_reuses_fixed_edge_with_a_unique_stopped_source_generation():
    backend, avp, playlist = backend_for()
    activate(backend, playlist[0], 1)
    activate(backend, playlist[1], 2)
    old_slot = backend._item_slots[playlist[1].item_id]
    backend.remove_item(playlist[1].item_id)
    deadline = time.monotonic() + 1
    while playlist[1].item_id in backend._item_slots and time.monotonic() < deadline:
        time.sleep(0.005)
    added = Clip(url="/media/new", name="new", item_id="item-new")
    activate(backend, added, 3)
    assert backend._item_slots[added.item_id] == old_slot
    stop = avp.commands.index(f"group.stop {item_group(old_slot)}")
    start = avp.commands.index(f"group.start {item_group(old_slot, 1)}")
    assert stop < start
    assert not [command for command in avp.commands
                if command.startswith("node.delete ")]
    backend.close()


def test_no_sentinel_black_or_stall_extension_appears_in_dynamic_commands():
    backend, avp, playlist = backend_for()
    activate(backend, playlist[1], 1)
    serialized = "\n".join(avp.commands).lower()
    assert "sentinel" not in serialized
    assert "fallback_active" not in serialized
    assert "repeat_on_stall" not in serialized
    backend.close()


def test_inactive_target_graph_error_fails_readiness_without_global_error():
    backend, _, playlist = backend_for()
    backend.report_graph_exception(
        "pl_item_0", "NodeGroup", "No such file or directory")
    with pytest.raises(RuntimeError, match="No such file or directory"):
        backend._wait_for_item_frame(0, 0)
    assert not [event for event in backend.poll_events()
                if event.kind == "error"]
    backend.close()
