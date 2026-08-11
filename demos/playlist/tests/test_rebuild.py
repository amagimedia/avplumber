"""Worker rebuild honours the backend node namespace: node.add rejects a busy
name, so every rebuild must node.delete the old worker nodes first.  Uses
FakeBackend, which enforces that rule, so these tests fail loudly if teardown
regresses."""
from playlist import (Clip, ElementMode, PlaylistController, PlaylistMode,
                      worker_node_names, worker_group)
from helpers import FakeBackend, clips


def _seed_worker0(fb):
    """The live builder creates worker 0's nodes; mirror that so the FakeBackend
    namespace matches a real session."""
    fb.live.update(worker_node_names(0))


def _controller_with_backend(cl=None, mode=PlaylistMode.LOOP_ALL, **kw):
    fb = FakeBackend()
    ctl = PlaylistController(fb, cl or clips("a", "b", "c"), mode=mode, **kw)
    _seed_worker0(fb)
    return ctl, fb


def test_many_cuts_never_hit_name_busy():
    ctl, fb = _controller_with_backend()
    # a long walk that repeatedly rebuilds both workers
    for _ in range(6):
        ctl.next()
    # if any rebuild forgot to delete, FakeBackend would already have raised
    assert fb.cmds


def test_rebuild_deletes_before_readding():
    ctl, fb = _controller_with_backend()
    fb.clear()
    ctl.next()                     # cut to worker1, then rebuild worker0 for next
    ctl.next()                     # cut to worker0 rebuild target -> rebuild worker1
    # every re-add of a worker name is preceded by a delete of that same name
    live_before = set()
    for c in fb.cmds:
        if c.startswith("node.delete "):
            live_before.discard(c[len("node.delete "):])
        elif c.startswith("node.add "):
            import json
            name = json.loads(c[len("node.add "):])["name"]
            assert name not in live_before, f"re-added busy name {name}"
            live_before.add(name)


def test_goto_rebuild_is_namespace_safe():
    ctl, fb = _controller_with_backend(clips("a", "b", "c", "d"))
    ctl.goto(3)
    ctl.goto(1)
    ctl.goto(2)
    assert fb.cmds  # no AssertionError raised == namespace stayed consistent
