"""Pure grid geometry used by the DMA-BUF browser scaling test."""

from dataclasses import dataclass
from math import ceil, sqrt


MAX_COMPOSITOR_INPUTS = 32


@dataclass(frozen=True)
class GridCell:
    x: int
    y: int
    width: int
    height: int


def grid_shape(source_count):
    if source_count < 1:
        raise ValueError("source_count must be positive")
    columns = ceil(sqrt(source_count))
    rows = ceil(source_count / columns)
    return columns, rows


def grid_cells(source_count, canvas_width, canvas_height):
    if min(canvas_width, canvas_height) < 2:
        raise ValueError("canvas dimensions must be at least 2")
    columns, rows = grid_shape(source_count)
    cell_width = (canvas_width // columns) & ~1
    cell_height = (canvas_height // rows) & ~1
    if min(cell_width, cell_height) < 2:
        raise ValueError("source_count produces cells smaller than 2 pixels")
    return tuple(
        GridCell(
            x=(index % columns) * cell_width,
            y=(index // columns) * cell_height,
            width=cell_width,
            height=cell_height,
        )
        for index in range(source_count)
    )


def fit_rect(cell, source_width, source_height):
    if min(source_width, source_height) < 2:
        raise ValueError("source dimensions must be at least 2")
    scale = min(cell.width / source_width, cell.height / source_height)
    width = max(2, int(source_width * scale) & ~1)
    height = max(2, int(source_height * scale) & ~1)
    return GridCell(
        x=cell.x + (cell.width - width) // 2,
        y=cell.y + (cell.height - height) // 2,
        width=width,
        height=height,
    )


def fit_filter_graph(rect):
    return f"scale_cuda=w={rect.width}:h={rect.height}"
