#!/usr/bin/env python3
"""Convert raw results to PNG and assemble deterministic contact sheets."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from overlay_common import HEIGHT, MAX_OVERLAYS, MODES, WIDTH


def raw_to_png(
    ffmpeg: str,
    source: Path,
    pixel_format: str,
    destination: Path,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def _sheet(images: list[tuple[str, Path]], destination: Path, columns: int = 5) -> None:
    if not images:
        return
    thumb_width = 321
    thumb_height = 181
    label_height = 24
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        (24, 27, 35),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(images):
        row, column = divmod(index, columns)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        with Image.open(path) as source:
            thumbnail = source.convert("RGB").resize((thumb_width, thumb_height))
        canvas.paste(thumbnail, (x, y))
        draw.text((x + 7, y + thumb_height + 6), label, fill=(245, 245, 245))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def make_result_sheets(result_png_dir: Path, sheet_dir: Path) -> None:
    for mode_name in MODES:
        images = [
            (f"{count} overlay{'s' if count != 1 else ''} / inputs={count + 1}",
             result_png_dir / f"{mode_name}-overlays_{count:02d}.png")
            for count in range(1, MAX_OVERLAYS + 1)
        ]
        existing = [(label, path) for label, path in images if path.is_file()]
        _sheet(existing, sheet_dir / f"results-{mode_name}.png")


def make_source_sheet(asset_png_dir: Path, sheet_dir: Path) -> None:
    images = [("base", asset_png_dir / "base.png")]
    images.extend(
        (f"overlay {index}", asset_png_dir / f"overlay_{index:02d}.png")
        for index in range(1, MAX_OVERLAYS + 1)
    )
    _sheet(images, sheet_dir / "source-layers.png", columns=4)
