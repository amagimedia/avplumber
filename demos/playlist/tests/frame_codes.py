"""Read the per-frame code that ``test-media/generate.sh`` burns into every fixture.

Strip at (64,0), 32 cells of 16x12 px, second row inverted:
``0xA<<28 | clip<<24 | frame<<8 | checksum``.  Frames are 1920x1080 luma.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

X0, CELL, ROW = 64, 16, 12
HEADER = 0xA


def checksum(clip: int, frame: int) -> int:
    return (clip * 37 + (frame >> 8) + (frame & 255)) & 255


def encode(clip: int, frame: int) -> int:
    return (HEADER << 28) | (clip << 24) | (frame << 8) | checksum(clip, frame)


def read_code(luma: np.ndarray, x0: int = X0, cell: int = CELL, row: int = ROW) -> Optional[Tuple[int, int]]:
    """Return (clip, frame) or None when the strip is blended, missing or corrupt."""
    xs = x0 + cell // 2 + cell * np.arange(32)
    top = luma[row // 2, xs].astype(int)
    bottom = luma[row + row // 2, xs].astype(int)
    if np.any(np.abs(top - bottom) < 120):
        return None
    bits = (top > bottom).astype(np.uint32)
    code = int(np.packbits(bits).view(">u4")[0])
    clip, frame, check = (code >> 24) & 0xF, (code >> 8) & 0xFFFF, code & 0xFF
    if code >> 28 != HEADER or check != checksum(clip, frame):
        return None
    return clip, frame
