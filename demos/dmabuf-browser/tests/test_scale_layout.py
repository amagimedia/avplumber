import sys
from pathlib import Path

GRAPH_DIR = Path(__file__).resolve().parents[1] / "graph"
sys.path.insert(0, str(GRAPH_DIR))

from dmabuf_scale_layout import fit_filter_graph, fit_rect, grid_cells, grid_shape


def test_eight_sources_use_three_by_three_grid():
    assert grid_shape(8) == (3, 3)
    cells = grid_cells(8, 1920, 1080)
    assert len(cells) == 8
    assert {(cell.width, cell.height) for cell in cells} == {(640, 360)}
    assert (cells[-1].x, cells[-1].y) == (640, 720)


def test_source_count_is_not_fixed_to_eight():
    assert grid_shape(1) == (1, 1)
    assert grid_shape(4) == (2, 2)
    assert grid_shape(12) == (4, 3)
    assert len(grid_cells(12, 1920, 1080)) == 12


def test_filter_preserves_aspect_ratio_and_centers_without_cuda_padding():
    fitted = fit_rect(grid_cells(2, 1920, 1080)[0], 1920, 1080)
    assert fitted == type(fitted)(x=0, y=270, width=960, height=540)
    assert fit_filter_graph(fitted) == "scale_cuda=w=960:h=540"
