"""The move transition at the Python boundary: graph.move sends mixer.move, the show's control
section may make it the pick transition with its unmatched sources' fade or cut, and the page's
bridge passes it on as a program take (layer_move.hpp, orchestrator/move.cpp are the engine side)."""
import copy
import json

import pytest

from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.application import MixerOptions, build_application
from test_audio_routing import _api
from test_graph import CONFIG
from test_layer_zoom_commands import _show


def _sent(app, prefix):
    return [json.loads(e.split(" ", 1)[1]) for e in app.avp.events if e.startswith(prefix)]


def test_move_sends_its_rate_curve_and_unmatched(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_show(None), MixerOptions(), api=_api(nodes, builder))
    scene = "full"
    app.mixer.move(scene, duration_sec=0.8, curve="ease-in-out", unmatched="cut")
    assert _sent(app, "mixer.move ")[-1] == {"mixer": app.mixer.name, "scene": scene, "duration_sec": 0.8,
                                              "curve": "ease-in-out", "unmatched": "cut"}
    app.mixer.move(scene)
    assert _sent(app, "mixer.move ")[-1] == {"mixer": app.mixer.name, "scene": scene, "duration_sec": 1.0}
    with pytest.raises(ValueError, match="fade or cut"):
        app.mixer.move(scene, unmatched="slide")


def test_control_move_and_its_unmatched_sources():
    doc = copy.deepcopy(CONFIG)
    cfg = mixer_config.parse(doc)
    assert cfg.move_unmatched == cfg.settings()["move_unmatched"] == "fade"
    chosen = mixer_config.parse({**doc, "control": {"transition": "move", "move_unmatched": "cut"}})
    assert chosen.transition == chosen.settings()["transition"] == "move"
    assert chosen.settings()["move_unmatched"] == "cut"
    with pytest.raises(mixer_config.ConfigError, match="control.move_unmatched must be fade or cut"):
        mixer_config.parse({**doc, "control": {"move_unmatched": "slide"}})


def test_page_has_the_move_button_and_its_unmatched_choice():
    from pathlib import Path
    import pyplumber.mixer.gui as gui
    page = (Path(gui.__file__).parent / "assets" / "index.html").read_text()
    assert '<button id="move" type="button"' in page and 'aria-keyshortcuts="M"' in page
    assert 'id="move-unmatched"' in page and 'data-unmatched="fade"' in page and 'data-unmatched="cut"' in page
    # Auto and the lever send the move with the fade rate and curve and the unmatched choice.
    assert 'kind === "move"' in page and "unmatched: moveUnmatched" in page and "move: true" in page
