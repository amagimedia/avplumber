#!/usr/bin/env python3
"""Generate deterministic PNG samples and planar fixtures for the demo."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from overlay_common import (
    HEIGHT,
    MAX_OVERLAYS,
    WIDTH,
    base_fixture_path,
    frame_size,
    overlay_fixture_path,
    read_frame,
    write_frame,
)


PALETTE = (
    (239, 71, 111),
    (255, 145, 77),
    (255, 209, 102),
    (131, 197, 95),
    (6, 214, 160),
    (17, 138, 178),
    (77, 150, 255),
    (113, 90, 255),
    (171, 71, 188),
    (244, 114, 182),
    (230, 57, 70),
    (42, 157, 143),
    (87, 117, 144),
    (144, 190, 109),
    (249, 132, 74),
)

ALPHA_SAMPLES = (0, 1, 127, 128, 254, 255)
INPUT_STREAM_FRAMES = 2


def make_base(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    image = Image.new("RGB", (width, height), (18, 22, 31))
    pixels = np.asarray(image).copy()
    colors = np.array(
        [
            (235, 235, 235),
            (235, 220, 30),
            (30, 220, 220),
            (30, 205, 55),
            (220, 35, 210),
            (220, 40, 40),
            (35, 60, 220),
        ],
        dtype=np.uint8,
    )
    band = max(1, width // len(colors))
    for index, color in enumerate(colors):
        start = index * band
        end = width if index == len(colors) - 1 else (index + 1) * band
        pixels[: height * 2 // 3, start:end] = color

    yy, xx = np.indices((height - height * 2 // 3, width))
    checker = ((xx // 16 + yy // 16) & 1).astype(bool)
    lower = pixels[height * 2 // 3 :]
    lower[checker] = (35, 40, 52)
    lower[~checker] = (205, 210, 220)
    image = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 228, 25), fill=(0, 0, 0))
    draw.text((8, 7), f"CUDA OVERLAY BASE {width}x{height}", fill=(255, 255, 255))
    label_x = max(0, width - 50)
    draw.rectangle((label_x, 4, width - 8, 24), fill=(0, 0, 0))
    draw.text((label_x + 6, 10), "L0", fill=(255, 255, 255))
    return image


def make_overlay(index: int, width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    if not 1 <= index <= MAX_OVERLAYS:
        raise ValueError(f"overlay index must be in 1..{MAX_OVERLAYS}")
    color = PALETTE[index - 1]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    bar_width = 38 + index * 2
    x0 = 12 + (index - 1) * ((width - 90) // MAX_OVERLAYS)
    x1 = min(width, x0 + bar_width)
    ramp = np.linspace(1, 254, max(1, x1 - x0), dtype=np.uint8)
    rgba[:, x0:x1, :3] = color
    rgba[:, x0:x1, 3] = ramp[np.newaxis, :]

    center_x0 = width // 4
    center_x1 = width * 3 // 4
    center_y0 = height // 3
    center_y1 = height * 2 // 3
    rgba[center_y0:center_y1, center_x0:center_x1, :3] = color
    rgba[center_y0:center_y1, center_x0:center_x1, 3] = 48 + index * 12

    sample_width = max(1, width // len(ALPHA_SAMPLES))
    for sample_index, alpha in enumerate(ALPHA_SAMPLES):
        start = sample_index * sample_width
        end = width if sample_index == len(ALPHA_SAMPLES) - 1 else (sample_index + 1) * sample_width
        rgba[height - 13 : height - 1, start:end, :3] = color
        rgba[height - 13 : height - 1, start:end, 3] = alpha

    rgba[-1, :, :3] = color
    rgba[-1, :, 3] = 128
    rgba[:, -1, :3] = color
    rgba[:, -1, 3] = 127

    image = Image.fromarray(rgba, "RGBA")
    draw = ImageDraw.Draw(image)
    label_x = min(width - 48, x0 + 2)
    draw.rectangle((label_x, 28, label_x + 42, 48), fill=(*color, 230))
    draw.text((label_x + 5, 34), f"L{index}", fill=(0, 0, 0, 255))
    return image


def _convert_png(
    ffmpeg: str,
    source: Path,
    destination: Path,
    pixel_format: str,
    width: int,
    height: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"format={pixel_format}",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        str(destination),
    ]
    subprocess.run(command, check=True)
    if pixel_format.startswith("yuva"):
        with Image.open(source) as image:
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        planes = read_frame(destination, pixel_format, width, height)
        planes[3] = alpha
        write_frame(destination, planes)
    expected = frame_size(pixel_format, width, height)
    actual = destination.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{destination} has {actual} bytes after conversion; expected {expected}")


def generate_all(
    artifact_dir: Path,
    ffmpeg: str = "ffmpeg",
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict[str, object]:
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    png_dir = artifact_dir / "assets" / "png"
    raw_dir = artifact_dir / "assets" / "raw"
    stream_dir = artifact_dir / "assets" / "input-streams"
    png_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)

    base_png = png_dir / "base.png"
    make_base(width, height).save(base_png)
    overlay_pngs: list[Path] = []
    for index in range(1, MAX_OVERLAYS + 1):
        path = png_dir / f"overlay_{index:02d}.png"
        make_overlay(index, width, height).save(path)
        overlay_pngs.append(path)

    for pixel_format in ("yuv420p", "yuv444p"):
        _convert_png(
            ffmpeg,
            base_png,
            base_fixture_path(raw_dir, pixel_format),
            pixel_format,
            width,
            height,
        )
    for index, source in enumerate(overlay_pngs, start=1):
        for pixel_format in ("yuva420p", "yuva444p"):
            _convert_png(
                ffmpeg,
                source,
                overlay_fixture_path(raw_dir, index, pixel_format),
                pixel_format,
                width,
                height,
            )

    for source in raw_dir.glob("*.yuv"):
        frame = source.read_bytes()
        (stream_dir / source.name).write_bytes(frame * INPUT_STREAM_FRAMES)

    manifest = {
        "width": width,
        "height": height,
        "overlay_count": MAX_OVERLAYS,
        "input_stream_frames": INPUT_STREAM_FRAMES,
        "alpha_samples": list(ALPHA_SAMPLES),
        "base_png": str(base_png.relative_to(artifact_dir)),
        "overlay_pngs": [str(path.relative_to(artifact_dir)) for path in overlay_pngs],
    }
    (artifact_dir / "assets" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    generate_all(args.artifact_dir, args.ffmpeg, args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
