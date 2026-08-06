"""Pure scene and geometry generation for the generic portrait mixer demo."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
CANONICAL_SOURCE_WIDTH = 1920
CANONICAL_SOURCE_HEIGHT = 1080
GRID_SHAPES = {
    2: (1, 2),
    4: (1, 4),
    8: (2, 4),
    16: (2, 8),
}
PREHEAT_CAPACITIES = (1, *GRID_SHAPES)


@dataclass(frozen=True)
class Placement:
    source_index: int
    slot_index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SceneSpec:
    name: str
    capacity: int
    page: int
    placements: tuple[Placement, ...]


def _require_input_count(input_count: int) -> None:
    if input_count < 1:
        raise ValueError("input_count must be at least 1")


def grid_page_count(input_count: int, capacity: int) -> int:
    _require_input_count(input_count)
    if capacity not in GRID_SHAPES:
        raise ValueError(f"unsupported grid capacity: {capacity}")
    return ceil(input_count / capacity)


def cell_size(capacity: int) -> tuple[int, int]:
    if capacity == 1:
        return CANVAS_WIDTH, CANVAS_HEIGHT
    try:
        columns, rows = GRID_SHAPES[capacity]
    except KeyError as exc:
        raise ValueError(f"unsupported layout capacity: {capacity}") from exc
    return CANVAS_WIDTH // columns, CANVAS_HEIGHT // rows


def layout_source_name(capacity: int, slot_index: int) -> str:
    if capacity not in PREHEAT_CAPACITIES:
        raise ValueError(f"unsupported layout capacity: {capacity}")
    if not 0 <= slot_index < capacity:
        raise ValueError(f"slot {slot_index} is outside capacity {capacity}")
    return f"layout_{capacity}_slot_{slot_index}"


def layout_filter_graph(capacity: int) -> str:
    width, height = cell_size(capacity)
    return (
        f"scale_cuda=w={width}:h={height}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad_cuda=w={width}:h={height}:"
        "x=(ow-iw)/2:y=(oh-ih)/2:color=black"
    )


def fitted_even_size(
    source_width: int,
    source_height: int,
    box_width: int,
    box_height: int,
) -> tuple[int, int]:
    if min(source_width, source_height, box_width, box_height) <= 0:
        raise ValueError("source and box dimensions must be positive")
    if source_width * box_height >= source_height * box_width:
        width = box_width
        height = source_height * box_width // source_width
    else:
        height = box_height
        width = source_width * box_height // source_height
    return max(2, width & ~1), max(2, height & ~1)


def fullscreen_scene(source_index: int) -> SceneSpec:
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    return SceneSpec(
        name=f"fullscreen_{source_index}",
        capacity=1,
        page=source_index,
        placements=(
            Placement(
                source_index=source_index,
                slot_index=0,
                x=0,
                y=0,
                width=CANVAS_WIDTH,
                height=CANVAS_HEIGHT,
            ),
        ),
    )


def grid_scene(input_count: int, capacity: int, page: int) -> SceneSpec:
    pages = grid_page_count(input_count, capacity)
    if not 0 <= page < pages:
        raise ValueError(f"page {page} is outside grid capacity {capacity} page range 0..{pages - 1}")

    columns, _ = GRID_SHAPES[capacity]
    width, height = cell_size(capacity)
    first_source = page * capacity
    placements = []
    for slot_index, source_index in enumerate(
        range(first_source, min(first_source + capacity, input_count))
    ):
        row, column = divmod(slot_index, columns)
        placements.append(
            Placement(
                source_index=source_index,
                slot_index=slot_index,
                x=column * width,
                y=row * height,
                width=width,
                height=height,
            )
        )
    return SceneSpec(
        name=f"grid_{capacity}_page_{page}",
        capacity=capacity,
        page=page,
        placements=tuple(placements),
    )


def all_scenes(input_count: int) -> tuple[SceneSpec, ...]:
    _require_input_count(input_count)
    scenes = [fullscreen_scene(index) for index in range(input_count)]
    for capacity in GRID_SHAPES:
        scenes.extend(
            grid_scene(input_count, capacity, page)
            for page in range(grid_page_count(input_count, capacity))
        )
    return tuple(scenes)

