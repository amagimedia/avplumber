"""Layer zoom at the Python boundary: a source's declared picture reaches the engine as
mixer.source_picture after its mixer.source, a cut carries its zoom, and a bad picture is a
ConfigError (layer_zoom.hpp is the engine side)."""
import json

import pytest

from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.application import MixerOptions, build_application
from test_audio_routing import _api
from test_graph import CONFIG

PICTURE = {"reference": {"x": 1920, "y": 0, "w": 1920, "h": 1080}, "metadata_key": "face_normalize_layers_v1",
           "mapped": [0, 1, 2]}


def _show(picture=PICTURE):
    camera = {**CONFIG["sources"][0], "width": 5760, "height": 3240, **({"picture": picture} if picture else {})}
    return mixer_config.parse({"canvas": CONFIG["canvas"], "sources": [camera],
                               "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
                               "renditions": [{"id": "record", "target": "program.ts"}]})


def test_source_picture_is_declared_after_its_source(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_show(), MixerOptions(), api=_api(nodes, builder))
    events = app.avp.events
    source = next(i for i, e in enumerate(events) if e.startswith("mixer.source "))
    picture = next(i for i, e in enumerate(events) if e.startswith("mixer.source_picture "))
    assert picture > source
    sent = json.loads(events[picture].split(" ", 1)[1])
    assert sent["name"] == "cam" and sent["reference"] == PICTURE["reference"]
    assert sent["metadata_key"] == "face_normalize_layers_v1" and sent["mapped"] == [0, 1, 2]


def test_no_picture_no_command(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_show(None), MixerOptions(), api=_api(nodes, builder))
    assert not [e for e in app.avp.events if e.startswith("mixer.source_picture ")]


@pytest.mark.parametrize("picture", [
    {**PICTURE, "reference": {"x": 5000, "y": 0, "w": 1920, "h": 1080}},   # outside the 5760x3240 source
    {**PICTURE, "metadata_key": 3}, {**PICTURE, "mapped": ["a"]}, {"reference": PICTURE["reference"]}])
def test_bad_picture_is_rejected(picture):
    with pytest.raises(mixer_config.ConfigError, match="picture"):
        _show(picture)


def test_cut_passes_zoom(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_show(), MixerOptions(), api=_api(nodes, builder))
    app.mixer.cut("full", zoom={"duration_sec": 1.5, "curve": "ease-in-out"})
    cut = [e for e in app.avp.events if e.startswith("mixer.cut ")][-1]
    assert json.loads(cut.split(" ", 1)[1])["zoom"] == {"duration_sec": 1.5, "curve": "ease-in-out"}
    app.mixer.cut("full")
    assert "zoom" not in json.loads([e for e in app.avp.events if e.startswith("mixer.cut ")][-1].split(" ", 1)[1])
