"""The lever's wipe of a clip the clip cache does not hold, through the native module's control
commands on a mixer with no nodes: one error, and the Preview stays as the operator chose it."""
import importlib.util
import json
from pathlib import Path

import pytest


def test_uncached_lever_wipe_keeps_the_preview(run_script):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native extension")
    result = run_script(Path(__file__).resolve(), timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    status = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(status) == 1, result.stdout
    assert "the lever wipes only cached clips, not /clips/not-cached.mov" in result.stdout + result.stderr
    # The take never began: nothing touched the (missing) slot nodes, so nothing cleared the Preview.
    assert status[0]["pvw_scene"] == "b" and status[0]["transition"] == "idle"
    assert status[0]["lever"]["engaged"] is False   # inactive until brought to an end


def run():
    from pyplumber import AVPlumber
    avp = AVPlumber()
    try:
        wipe = {"wipe_otm": "wipe_otm", "wipe_selector": "wipe_selector", "wipe_group": "wipe",
                "wipe_input_node": "wipe_clip", "wipe_cache_store": "lever_test_clips"}
        avp.executeCommandsFromString(
            "mixer.init m " + json.dumps({"initial_pgm_scene": "a", "initial_pvw_scene": "b", **wipe}) + "\n"
            'mixer.scene m a {"sources":{}}\n'
            'mixer.scene m b {"sources":{}}\n'
            'mixer.lever {"mixer":"m","position":0.3,"wipe_file":"/clips/not-cached.mov"}\n'
            "mixer.status m\n")
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run()
