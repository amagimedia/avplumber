"""Output format policy for auto mixer recording outputs."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlparse

SUPPORTED_RECORD_OUTPUT_FORMATS = ("mpegts", "flv")
_MPEGTS_SUFFIXES = {".ts", ".m2ts", ".mts", ".mpegts"}


def infer_record_output_format(
    output_url: str,
    explicit_format: str | None = None,
) -> str:
    """Infer or validate the muxer for the supported recording outputs."""
    parsed = urlparse(output_url)
    scheme = parsed.scheme.lower()
    fmt = explicit_format.lower() if explicit_format else None

    if fmt is not None and fmt not in SUPPORTED_RECORD_OUTPUT_FORMATS:
        raise ValueError("--output-format must be 'mpegts' or 'flv'")

    if scheme == "rtmp":
        if fmt not in (None, "flv"):
            raise ValueError("RTMP outputs require --output-format flv")
        return "flv"

    if scheme == "srt":
        if fmt not in (None, "mpegts"):
            raise ValueError("SRT outputs require --output-format mpegts")
        return "mpegts"

    if fmt is not None:
        return fmt

    suffix = PurePosixPath(parsed.path if scheme else output_url).suffix.lower()
    if suffix == ".flv":
        return "flv"
    if suffix in _MPEGTS_SUFFIXES:
        return "mpegts"
    raise ValueError(
        "Recording output must be RTMP, SRT, .flv, or MPEG-TS "
        "(.ts, .m2ts, .mts, .mpegts); otherwise pass --output-format."
    )
