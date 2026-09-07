"""Numbered video fixtures for checking actual pixels across mixer transitions.

Generate files outside the repository. The code includes source identity, frame
number, a checksum, and an inverted second row to reject blended/corrupt reads.
"""
import argparse
import pathlib
import subprocess

import numpy as np


WIDTH, HEIGHT = 640, 360
# Interleave warm and cool hues so neighboring cells remain distinct in grids.
SOURCE_COLOURS = (
    (255, 72, 72), (32, 210, 255), (255, 200, 32), (166, 100, 255),
    (48, 230, 120), (255, 88, 192), (48, 116, 255), (255, 144, 40),
    (32, 222, 194), (230, 100, 255), (194, 236, 40), (255, 120, 148),
    (100, 168, 255), (255, 224, 104), (100, 248, 184), (198, 148, 255),
)


def frame_image(source: int, frame: int) -> np.ndarray:
    if not 0 <= source < 16 or not 0 <= frame < 65536:
        raise ValueError("source/frame outside fixture range")
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = SOURCE_COLOURS[source]
    x = frame * 7 % 580
    image[40:100, x:x+60] = 16
    image[44:96, x+4:x+56] = 240
    checksum = (source * 37 + (frame >> 8) + (frame & 255)) & 255
    code = (0xA << 28) | (source << 24) | (frame << 8) | checksum
    for bit in range(32):
        value = 240 if code & (1 << (31-bit)) else 16
        image[140:160, 64+bit*16:80+bit*16] = value
        image[160:180, 64+bit*16:80+bit*16] = 256-value
    return image


def read_code(tile: np.ndarray) -> tuple[int, int] | None:
    """Read a whole 16:9 source tile at any scale, without OCR."""
    height, width = tile.shape[:2]
    code = 0
    for bit in range(32):
        x = round((72+bit*16) * width / WIDTH)
        values = [float(np.mean(tile[round(y*height/HEIGHT), x])) for y in (150, 170)]
        if abs(values[0] - values[1]) < 120:
            return None
        code = (code << 1) | int(values[0] > values[1])
    if code >> 28 != 0xA:
        return None
    source = (code >> 24) & 15
    frame = (code >> 8) & 65535
    checksum = (source * 37 + (frame >> 8) + (frame & 255)) & 255
    return (source, frame) if checksum == (code & 255) else None


def generate(path: pathlib.Path, source: int, fps: int, seconds: int) -> None:
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-threads", "2", str(path),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(fps * seconds):
            encoder.stdin.write(frame_image(source, frame).tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("fixture encoding failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for source in range(args.sources):
        generate(args.directory / f"source-{source}.mp4", source, args.fps, args.seconds)
