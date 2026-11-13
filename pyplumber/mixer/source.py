"""Source-paced composition using the mixer's renderer, move geometry and easing.

The caller owns decode, inference, pacing and encode. A move is attached to each
source frame, so buffered inference and file playback cannot change its timing.
"""

from copy import deepcopy
from dataclasses import dataclass
import math

from .backends.cuda import CudaMixerBackend
from .config import DEFAULT_FADE_CURVE, MOVE_UNMATCHED, fade_curve


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
        fade_curve(self.curve, "source move curve")
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


def source_compositor(params, *, api=None):
    """Build the source-clocked variant of the shared CUDA mixer compositor.

    ``move_metadata_key`` optionally selects a JSON ``SourceMove.at(pts)`` request
    on ``move_metadata_input`` (default 0). Without one, ordinary per-frame layer
    metadata is rendered unchanged. Upstream retains ownership of timestamps/EOF.
    """
    return CudaMixerBackend().source_compositor(params, api=api)
