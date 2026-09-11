#!/usr/bin/env python3
"""MXL round-trip demo.

Builds two avplumber graphs in one process:

* writer: file -> demux -> decode -> rawvideo encode -> mxl mux -> mxl:// URL
* reader: mxl:// URL -> demux -> decode -> re-encode -> file

The two graphs share a `/dev/shm/mxl` domain and a set of flow UUIDs so
the reader picks up what the writer publishes. The point is to exercise
the FFmpeg MXL demuxer and muxer end-to-end through avplumber's generic
`input` / `output` nodes; MXL is passed via `format="mxl"` and per-stream
UUIDs in `options`, with no new node types.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pyplumber
from pyplumber.node import (
    Demux,
    DecVideo,
    EncVideo,
    Input,
    InputRec,
    Mux,
    Output,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=os.environ.get("AVP_INPUT", "lavfi:testsrc=size=320x240:rate=25"),
        help=(
            "Source for the writer. A filesystem path is played back through "
            "InputRec (looped, realtime-paced). A value prefixed with "
            "'lavfi:' opens an FFmpeg libavfilter source (infinite, "
            "monotonic timestamps) — the recommended default because MXL is "
            "a live shared-memory transport."
        ),
    )
    p.add_argument(
        "--output",
        default=os.environ.get("AVP_OUTPUT", "output.mp4"),
        help="Local file the reader side writes.",
    )
    p.add_argument(
        "--domain",
        default=os.environ.get("AVP_MXL_DOMAIN", "/dev/shm/mxl"),
        help="MXL shared-memory domain directory.",
    )
    p.add_argument(
        "--video-flow-id",
        default=os.environ.get("AVP_MXL_VIDEO_ID"),
        help="Video flow UUID. Random if unset.",
    )
    p.add_argument(
        "--audio-flow-id",
        default=os.environ.get("AVP_MXL_AUDIO_ID"),
        help="Audio flow UUID. Unused if input has no audio.",
    )
    p.add_argument(
        "--reader-only",
        action="store_true",
        help="Skip the writer side (attach to a domain another process is publishing to).",
    )
    p.add_argument(
        "--writer-only",
        action="store_true",
        help="Skip the reader side (publish only).",
    )
    return p.parse_args()


def _writer_url(domain: str) -> str:
    """MXL muxer wants a plain filesystem path (AVFMT_NOFILE)."""
    return domain


def _reader_url(domain: str, video_id: str) -> str:
    """MXL demuxer takes an mxl:// URI with the flow id in a query param."""
    return f"mxl://{domain}?id={video_id}"


def _build_writer(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> None:
    """source -> demux -> decode -> rescale -> v210 encode -> mxl output."""
    from pyplumber.node import RescaleVideo, AssumeVideoFormat

    if args.input.startswith("lavfi:"):
        # Live libavfilter source: infinite, monotonic PTS, always
        # ready. No loop-wraparound DTS glitches.
        avp.addNode(Input({
            "url": args.input.removeprefix("lavfi:"),
            "format": "lavfi",
            "dst": "w_in",
            "group": "w_in",
            "name": "w_input",
            "auto_restart": "off",
            "timeout": -1,
        }))
    else:
        # File input: loop and pace to source frame rate so the
        # downstream MXL writer sees a steady live stream.
        avp.addNode(InputRec({
            "url": args.input,
            "dst": "w_in",
            "group": "w_in",
            "name": "w_input",
            "auto_restart": "off",
            "timeout": -1,
            "loop": True,
            "realtime": True,
        }))
    avp.addNode(Demux({
        "src": "w_in",
        "routing": {"v:0": "w_vpkt"},
        "group": "w_in",
        "auto_restart": "off",
    }))
    avp.addNode(DecVideo({
        "src": "w_vpkt",
        "dst": "w_vframe",
        "group": "w_in",
        "name": "w_dec",
        "auto_restart": "off",
    }))
    # v210 requires 10-bit 4:2:2 planar. Convert explicitly so the
    # encoder's metadata chain is unambiguous.
    avp.addNode(RescaleVideo({
        "src": "w_vframe",
        "dst": "w_vscaled",
        "group": "w_in",
        "name": "w_scale",
        "dst_pixel_format": "yuv422p10le",
        "auto_restart": "off",
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "w_vscaled",
        "dst": "w_vscaled_assumed",
        "group": "w_in",
        "width": 320,
        "height": 240,
        "pixel_format": "yuv422p10le",
        "real_pixel_format": "yuv422p10le",
        "auto_restart": "off",
    }))
    avp.addNode(EncVideo({
        "src": "w_vscaled_assumed",
        "dst": "w_venc",
        "group": "w_in",
        "name": "w_enc",
        "codec": "v210",
        "auto_restart": "off",
    }))
    avp.addNode(Mux({
        "src": ["w_venc"],
        "dst": "w_mux",
        "group": "w_in",
    }))
    mxl_options: dict[str, str] = {}
    if args.video_flow_id:
        mxl_options["video_flow_id"] = args.video_flow_id
    if args.audio_flow_id:
        mxl_options["audio_flow_id"] = args.audio_flow_id
    avp.addNode(Output({
        "src": "w_mux",
        "url": _writer_url(args.domain),
        "format": "mxl",
        "options": mxl_options,
        "group": "w_in",
        "auto_restart": "off",
    }))


def _build_reader(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> None:
    """mxl:// -> demux -> decode -> rescale -> mpeg4 encode -> file."""
    from pyplumber.node import RescaleVideo, AssumeVideoFormat

    # The MXL demuxer returns EAGAIN when the writer hasn't published
    # a new grain yet, which avplumber's Input node treats as fatal.
    # `auto_restart:"group"` restarts the reader chain when that
    # happens; the group's own auto-restart handling reopens the edge
    # cleanly. `blocking=1` should make av_read_frame block instead,
    # but passing it via the `options` dict does not yet reach the
    # demuxer's private AVOptions (see TODO in README).
    avp.addNode(Input({
        "url": _reader_url(args.domain, args.video_flow_id),
        "format": "mxl",
        "dst": "r_in",
        "group": "r_in",
        "name": "r_input",
        "auto_restart": "group",
        "timeout": -1,
        "options": {
            "blocking": "1",
            "grain_index_init": "head",
        },
    }))
    avp.addNode(Demux({
        "src": "r_in",
        "routing": {"v:0": "r_vpkt"},
        "group": "r_in",
        "auto_restart": "off",
    }))
    avp.addNode(DecVideo({
        "src": "r_vpkt",
        "dst": "r_vframe",
        "group": "r_in",
        "name": "r_dec",
        "auto_restart": "off",
    }))
    # mpeg4 needs yuv420p — rescale from the 10-bit 4:2:2 flow.
    avp.addNode(RescaleVideo({
        "src": "r_vframe",
        "dst": "r_vscaled",
        "group": "r_in",
        "name": "r_scale",
        "dst_pixel_format": "yuv420p",
        "auto_restart": "off",
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "r_vscaled",
        "dst": "r_vscaled_assumed",
        "group": "r_in",
        "width": 320,
        "height": 240,
        "pixel_format": "yuv420p",
        "real_pixel_format": "yuv420p",
        "auto_restart": "off",
    }))
    avp.addNode(EncVideo({
        "src": "r_vscaled_assumed",
        "dst": "r_venc",
        "group": "r_in",
        "name": "r_enc",
        "codec": "mpeg4",
        "auto_restart": "off",
    }))
    avp.addNode(Mux({
        "src": ["r_venc"],
        "dst": "r_mux",
        "group": "r_in",
    }))
    avp.addNode(Output({
        "src": "r_mux",
        "url": args.output,
        "format": "mp4",
        # Fragmented MP4 so the file stays playable even if the demo
        # is interrupted before the mp4 trailer is written.
        "options": {"movflags": "frag_keyframe+empty_moov+default_base_moof"},
        "group": "r_in",
        "auto_restart": "off",
    }))


def main() -> int:
    args = _parse_args()
    if args.reader_only and args.writer_only:
        print("--reader-only and --writer-only are mutually exclusive", file=sys.stderr)
        return 2
    if not args.video_flow_id:
        args.video_flow_id = str(uuid.uuid4())
        print(f"video flow id: {args.video_flow_id}")

    avp = pyplumber.AVPlumber()

    if not args.reader_only:
        _build_writer(avp, args)
    if not args.writer_only:
        _build_reader(avp, args)

    if not args.reader_only:
        avp.group("w_in").startNodes()
    if not args.writer_only:
        # Give the writer a moment to open the flow before the reader
        # tries to attach.
        if not args.reader_only:
            time.sleep(1.0)
        avp.group("r_in").startNodes()

    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
