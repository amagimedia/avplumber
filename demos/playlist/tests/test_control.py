"""JSON control protocol shared by the backend process and the TUI."""

import asyncio

from control import LocalClient, apply_verb, clip_from_json, clip_to_json, status_json
from helpers import advance, clips, controller
from playlist import ElementMode, Transition


def test_clip_json_round_trip():
    clip = clips(("a", ElementMode.TIMED, {"duration_ms": 1500, "play_from_ms": 20, "speed": 1.5}))[0]
    assert clip_from_json(clip_to_json(clip)) == clip


def test_status_json_lists_every_element_with_state():
    ctl, _, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    data = status_json(ctl, 1000)
    assert data["transport"] == "Playing" and data["active"] == 0 and data["next"] == 1
    assert [c["state"] for c in data["clips"]] == ["Playing", "Stopped", "Stopped"]
    assert data["transition"] == "Cut" and data["transition_ms"] == 500
    assert data["position_ms"] == 1000 and data["next_at_ms"] == 10_000


def test_every_verb_dispatches():
    ctl, backend, clock = controller()
    apply_verb(ctl, "play", {}); advance(ctl, clock, 0)
    apply_verb(ctl, "select", {"index": 2})
    apply_verb(ctl, "transition", {"transition": "Fade", "duration_ms": 700})
    apply_verb(ctl, "end_mode", {"index": 2, "end": "Timed", "duration": 2000})
    apply_verb(ctl, "add", {"clip": {"url": "/media/new"}})
    apply_verb(ctl, "edit", {"index": 3, "clip": {"url": "/media/new", "cue_in": 100}})
    apply_verb(ctl, "move", {"index": 3, "to": 1})
    apply_verb(ctl, "enable", {"index": 1, "enabled": False})
    apply_verb(ctl, "remove", {"index": 1})
    apply_verb(ctl, "take", {"index": 2}); advance(ctl, clock, 0)
    apply_verb(ctl, "hold", {"index": 0})
    apply_verb(ctl, "park", {"index": 0})
    apply_verb(ctl, "mode", {"mode": "PlayAll"})
    apply_verb(ctl, "pause", {}); apply_verb(ctl, "toggle", {})
    apply_verb(ctl, "next", {}); apply_verb(ctl, "prev", {}); apply_verb(ctl, "stop", {})
    s = status_json(ctl, clock.now)
    assert s["transition"] == "Fade" and s["mode"] == "PlayAll" and len(s["clips"]) == 3
    assert ctl.transition is Transition.FADE


def test_local_client_reports_bad_requests_as_status_errors():
    ctl, *_ = controller()
    client = LocalClient(ctl)

    async def scenario():
        await client.send("take", index=9)
        await client.send("mode", mode="Nope")
        return await client.status()
    snap = asyncio.run(scenario())
    assert snap.error and snap.transport == "Stopped"
