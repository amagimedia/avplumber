"""Static code transforms for the shared GPU sampler; no frame pixels read on CPU."""

import math


def white_wash_lut(amount: float = 0.3) -> list[list[int]]:
    """The legacy limited-range NV12 lutyuv wash, including truncation BEFORE upscale."""
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("wash amount must be in 0..1")
    gain = 1 - amount
    y = [int(code * gain + amount * 235) for code in range(256)]
    uv = [int((code - 128) * gain + 128) for code in range(256)]
    return [y, uv, uv.copy()]
