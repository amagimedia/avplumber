import pytest

from layouts import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    GRID_SHAPES,
    all_scenes,
    cell_size,
    fitted_even_size,
    grid_page_count,
    grid_scene,
    layout_filter_graph,
)


def test_approved_grid_dimensions_cover_canvas():
    expected = {
        2: (1, 2, 1080, 960),
        4: (1, 4, 1080, 480),
        8: (2, 4, 540, 480),
        16: (2, 8, 540, 240),
    }
    for capacity, (columns, rows, width, height) in expected.items():
        assert GRID_SHAPES[capacity] == (columns, rows)
        assert cell_size(capacity) == (width, height)
        assert columns * width == CANVAS_WIDTH
        assert rows * height == CANVAS_HEIGHT


def test_pages_assign_consecutive_sources_without_reflow():
    assert grid_page_count(17, 16) == 2
    first = grid_scene(17, 16, 0)
    second = grid_scene(17, 16, 1)

    assert [placement.source_index for placement in first.placements] == list(range(16))
    assert [placement.source_index for placement in second.placements] == [16]
    assert second.placements[0].slot_index == 0
    assert second.placements[0].x == 0
    assert second.placements[0].y == 0


def test_incomplete_pages_leave_unused_cells_empty():
    scene = grid_scene(3, 8, 0)
    assert len(scene.placements) == 3
    assert {placement.slot_index for placement in scene.placements} == {0, 1, 2}


def test_every_capacity_exists_even_for_one_input():
    names = {scene.name for scene in all_scenes(1)}
    assert names == {
        "fullscreen_0",
        "grid_2_page_0",
        "grid_4_page_0",
        "grid_8_page_0",
        "grid_16_page_0",
    }


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((1080, 1920), (1080, 606)),
        ((1080, 960), (1080, 606)),
        ((1080, 480), (852, 480)),
        ((540, 480), (540, 302)),
        ((540, 240), (426, 240)),
    ],
)
def test_landscape_content_is_even_and_never_cropped(box, expected):
    fitted = fitted_even_size(1920, 1080, *box)
    assert fitted == expected
    assert fitted[0] <= box[0]
    assert fitted[1] <= box[1]
    assert fitted[0] % 2 == fitted[1] % 2 == 0


def test_filter_graph_scales_to_fit_then_pads_black():
    graph = layout_filter_graph(8)
    assert "scale_cuda=w=540:h=480" in graph
    assert "force_original_aspect_ratio=decrease" in graph
    assert "force_divisible_by=2" in graph
    assert "pad_cuda=w=540:h=480" in graph
    assert "color=black" in graph


def test_invalid_capacity_and_page_are_rejected():
    with pytest.raises(ValueError, match="unsupported grid capacity"):
        grid_page_count(2, 3)
    with pytest.raises(ValueError, match="outside"):
        grid_scene(2, 2, 1)

