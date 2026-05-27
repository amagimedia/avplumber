"""Default talk-show input-role profile.

The profile preserves the original auto-switch semantics without binding them
to client-specific filenames. Explicit CLI indices can override every value.
"""

from __future__ import annotations

DEFAULT_PROGRAM_AUDIO_INPUT_INDEX = 0
DEFAULT_SPECIAL_SPEAKER_INDEX = 0
DEFAULT_VAD_ONLY_PRIORITY_SPEAKER_INDEX = 1
DEFAULT_STATIC_FACE_CROP_INPUTS = (2,)

DEFAULT_SPECIAL_SPEAKER_MARGIN_DB = 3.0
