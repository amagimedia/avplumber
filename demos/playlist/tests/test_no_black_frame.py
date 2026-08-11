"""Never-black + gapless-ordering regression at the controller/command level.
The pixel-level never-black check is a bench test (needs GPU/Janus); here we
assert the *mechanism* that guarantees it: no black source is ever added, and
every cut unpauses the incoming worker before flipping active."""
from playlist import contains_black_source, NodeSpec, pause_team
from helpers import clips, controller


def _added_specs(rec):
    specs = []
    for node in rec.added_nodes():
        specs.append(NodeSpec(node["type"], node["name"], node["group"],
                              {k: v for k, v in node.items()
                               if k not in ("type", "name", "group")}))
    return specs


def test_no_black_source_added_during_full_session():
    ctl, rec = controller()
    ctl.next(); ctl.prev(); ctl.goto(2)
    assert contains_black_source(_added_specs(rec)) is False


def test_every_cut_unpauses_before_flip():
    ctl, rec = controller()
    for op in (ctl.next, ctl.next, ctl.prev):
        rec.clear()
        op()
        # find the unpause of the target worker and the active flip; unpause first
        target = ctl.status().active_worker
        seq = rec.cmds
        unpause = f"resume {pause_team(target)}"
        flips = [i for i, c in enumerate(seq)
                 if c.endswith(f"active {target}") or
                 (c.startswith("timeline.set") and f'"val": {target}' in c)]
        assert unpause in seq, seq
        assert flips, seq
        assert seq.index(unpause) < flips[0]


def test_goto_unpauses_before_flip():
    ctl, rec = controller()
    ctl.goto(2)
    target = ctl.status().active_worker
    unpause = f"resume {pause_team(target)}"
    flip_idx = next(i for i, c in enumerate(rec.cmds) if "active" in c)
    assert rec.cmds.index(unpause) < flip_idx


def test_paused_workers_start_frozen():
    """Any worker node.add for a pause node must carry paused=true."""
    ctl, rec = controller()
    ctl.next()
    for node in rec.added_nodes():
        if node["type"] == "pause":
            assert node["paused"] is True
