"""Source thumbnail banks: a fixed 8x8 atlas of the sources, 128x72 each, in a 1024x576 15 fps
H.264 baseline output that clients crop thumbnails from (pyplumber/mixer/thumbnail.py)."""

import pytest

from pyplumber.mixer.aux_layout import layout_cells
from pyplumber.mixer.config import ConfigError, Item, MixerConfig, Rect, Scene, Source, parse_aux_buses
from pyplumber.mixer import thumbnail


def config(n):
    sources = tuple(Source(f"s{i}", "video", f"clip{i}.mp4", 1920, 1080) for i in range(n))
    return MixerConfig(1080, 1920, 30, sources, (Scene("full", (Item("s0", Rect(0, 0, 1920, 1080)),)),))


def bank(page=0, **rendition):
    return {"id": f"thumbnails{page + 1}", "layout": {"preset": "source_thumbnails", "page": page},
            "renditions": [{"id": "monitor", "port": 5120 + 4 * page, "width": 1024, "height": 576, "fps": 15,
                            "codec": "h264_nvenc", "profile": "baseline", **rendition}]}


def test_a_bank_maps_up_to_64_sources_in_an_8x8_atlas():
    cfg = config(63)
    (bus,) = parse_aux_buses([bank()], cfg)
    cells = layout_cells(cfg, bus.layout, 0)
    assert len(cells) == 63 and all(c["role"] == "source" and c["w"] == 128 and c["h"] == 72 for c in cells)
    assert (cells[9]["source"], cells[9]["x"], cells[9]["y"]) == (9, 128, 72)
    assert bus.layouts == (bus.layout,) and (bus.renditions[0].width, bus.renditions[0].fps) == (1024, 15)
    assert [c["source"] for c in thumbnail.cells(config(130), 2)] == [128, 129]   # the third bank's tail


@pytest.mark.parametrize("bad", [{"width": 1920, "height": 1080}, {"profile": "high"}, {"fps": 30}])
def test_a_bank_is_silent_baseline_at_its_own_size(bad):
    with pytest.raises(ConfigError):
        parse_aux_buses([bank(**bad)], config(8))


def test_other_buses_cannot_switch_to_a_bank_layout():
    multiview = {"id": "mv", "scenes": ["full"], "layouts": [{"preset": "source_thumbnails"}],
                 "renditions": [{"id": "monitor", "port": 5010}]}
    with pytest.raises(ConfigError):
        parse_aux_buses([multiview], config(8))


def test_a_face_sheet_shows_its_camera_picture_shrunk_without_aliasing():
    from pyplumber.mixer.aux_layout import base_composition
    picture = {"reference": {"x": 1920, "y": 0, "w": 1920, "h": 1080}, "metadata_key": "face_normalize_layers_v1",
               "mapped": [0, 1]}
    sources = (Source("leo", "video", "leo.mp4", 5760, 3240, picture=picture), Source("gfx", "browser", "http://x", 1920, 1080))
    cfg = MixerConfig(1080, 1920, 30, sources, (Scene("full", (Item("leo", Rect(0, 0, 1920, 1080)),)),))
    (bus,) = parse_aux_buses([bank()], cfg)
    layers = base_composition(cfg, layout_cells(cfg, bus.layout, 0), [])["layers"]
    leo, gfx = layers[0], layers[1]
    assert leo["crop"] == {"x": 1920, "y": 0, "w": 1920, "h": 1080}         # the camera, not the whole sheet
    assert "crop" not in gfx                                                 # a plain source: its whole frame
    assert leo["filter"] == gfx["filter"] == "auto" and leo["samples"] == 8  # multisample for a 15x shrink
