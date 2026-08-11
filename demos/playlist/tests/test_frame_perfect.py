"""Frame-perfect *pre-conditions* (structural, host-independent).

True zero-gap/zero-dup is a bench measurement; here we assert the conditions the
design says are necessary: a scheduled (cued) cut carries an explicit boundary
pts, and a sequential next reuses the already-preloaded worker rather than
rebuilding it in the critical window."""
from playlist import worker_group
from helpers import controller
from playlist import PlaylistMode as M
import json


def test_cued_goto_carries_boundary_pts():
    ctl, rec = controller()
    ctl.goto(2, at_pts_ms=15000)
    ts = rec.matching("timeline.set ")
    assert ts
    payload = json.loads(ts[-1][len("timeline.set "):])
    assert payload["at"] == 15000


def test_sequential_next_uses_preloaded_worker_without_rebuild():
    ctl, rec = controller(mode=M.LOOP_ALL)     # a,b,c ; worker1 preloaded with b
    # first next: b is on the idle worker already? preload happens at construction
    # only for worker0; so worker1 gets built here. Do it, then measure the 2nd.
    ctl.next()                                  # -> b
    rec.clear()
    ctl.next()                                  # -> c ; c was preloaded on worker0
    # the critical-window cut must NOT stop+rebuild the target worker before cut
    target = ctl.status().active_worker
    stops_before_cut = []
    for c in rec.cmds:
        if c.startswith("node.object.set") and "active" in c:
            break
        if c == f"group.stop {worker_group(target)}":
            stops_before_cut.append(c)
    assert not stops_before_cut, "target worker rebuilt in the cut window"
