import pytest

from extra_aux import layouts
from pyplumber.mixer.aux_layout import parse_layout
from pyplumber.mixer.config import MixerConfig, Source


def config(width, height, sources):
    return MixerConfig(width, height, 30, tuple(Source(f"s{i}", "video", f"clip{i}.mp4") for i in range(sources)), ())


@pytest.mark.parametrize("width, height", [(1080, 1920), (1920, 1080)])
@pytest.mark.parametrize("sources", [1, 3, 5, 16, 110])
def test_random_layouts_tile_the_canvas_with_distinct_sources(width, height, sources):
    cfg = config(width, height, sources)
    specs = layouts("aux1", cfg)
    sizes = [len(spec["cells"]) for spec in specs]
    # Different cell counts, fewest first: 4 to 16, never more cells than sources.
    assert sizes == sorted(set(sizes)) and len(sizes) == min(5, min(16, sources) - min(4, sources) + 1)
    assert all(min(4, sources) <= n <= min(16, sources) for n in sizes)
    for spec in specs:
        assert parse_layout(cfg, spec) == spec   # even cells inside the canvas, source cells only
        cells = spec["cells"]
        assert {c["role"] for c in cells} == {"source"}
        assert len({c["source"] for c in cells}) == len(cells)
        assert sum(c["w"] * c["h"] for c in cells) == width * height   # inside it and disjoint: it tiles it


def test_random_layouts_follow_the_bus_id_and_source_list():
    cfg = config(1080, 1920, 40)
    assert layouts("aux1", cfg) == layouts("aux1", config(1080, 1920, 40))
    assert layouts("aux2", cfg) != layouts("aux1", cfg)
    assert layouts("aux1", config(1080, 1920, 41)) != layouts("aux1", cfg)
