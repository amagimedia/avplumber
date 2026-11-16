"""Source-paced composition using the mixer's renderer, move geometry and easing.

The caller owns decode, inference, pacing and encode. A move is attached to each
source frame, so buffered inference and file playback cannot change its timing.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

FADE_CURVES = ("linear", "ease-in", "ease-out", "ease-in-out")
DEFAULT_FADE_CURVE = "linear"
MOVE_UNMATCHED = ("fade", "cut")


def validate_curve(value, where="curve"):
    """Validate the curve names implemented by the native move/fade primitives."""
    if not isinstance(value, str) or value not in FADE_CURVES:
        raise ValueError(f"{where} must be one of {', '.join(FADE_CURVES)}")
    return value


@dataclass(frozen=True)
class SourceMove:
    """A frozen pair of compositions, sampled on the source's timeline in seconds.

    Keep the same instance for a take; detections during that take must not replace
    its endpoints. Geometry and the selected easing curve are evaluated natively.
    """

    _from: list
    _to: list
    start: float
    duration: float
    curve: str = DEFAULT_FADE_CURVE
    unmatched: str = "fade"

    def __post_init__(self):
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.duration)
            or self.duration <= 0
        ):
            raise ValueError(
                "source move: start must be finite and duration positive and finite"
            )
        validate_curve(self.curve, "source move curve")
        if self.unmatched not in MOVE_UNMATCHED:
            raise ValueError("source move: unmatched sources fade or cut")
        object.__setattr__(self, "_from", deepcopy(self._from))
        object.__setattr__(self, "_to", deepcopy(self._to))

    def at(self, pts: float) -> dict:
        if not math.isfinite(pts):
            raise ValueError("source move: frame PTS must be finite")
        return {
            "from": deepcopy(self._from),
            "to": deepcopy(self._to),
            "progress": min(1.0, max(0.0, (pts - self.start) / self.duration)),
            "curve": self.curve,
            "unmatched": self.unmatched,
        }
