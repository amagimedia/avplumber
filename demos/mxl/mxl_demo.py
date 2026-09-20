#!/usr/bin/env python3
"""MXL round-trip demo.

Builds two avplumber graphs in one process:

* writer: file -> demux -> decode -> v210 encode -> mxl mux -> mxl:// URL
* reader: mxl:// URL -> demux -> v210 unpack -> re-encode -> file

The reader unpacks on the GPU (`v210_to_cuda` + NVENC) when a CUDA
device is present and falls back to the CPU v210 decoder otherwise, and
takes grains zero-copy out of the shared-memory ring by default.

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
    ForceFPS,
    Input,
    InputRec,
    Mux,
    Output,
    Realtime,
)
from pyplumber.mixer.inputs import v210_row_stride

# Name of the CUDA device the GPU reader path initializes and shares.
_HWACCEL = "@gpu"


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
    p.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("AVP_WIDTH", 320)),
        help="Flow width. The writer rescales to it; v210 requires it to be even.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("AVP_HEIGHT", 240)),
        help="Flow height. The writer rescales to it.",
    )
    p.add_argument(
        "--fps",
        default=os.environ.get("AVP_FPS", "25"),
        help="Flow frame rate, '25' or '30000/1001'. Must match the source.",
    )
    p.add_argument(
        "--gpu-unpack",
        choices=("auto", "on", "off"),
        default=os.environ.get("AVP_GPU_UNPACK", "auto"),
        help=(
            "Read the flow straight onto the GPU: v210_to_cuda unpacks each "
            "grain into CUDA P210 frames and NVENC encodes the output. "
            "'auto' (default) uses it when a CUDA device is present, else "
            "falls back to the CPU v210 decoder and mpeg4."
        ),
    )
    p.add_argument(
        "--writer-pace",
        choices=("on", "off"),
        default=os.environ.get("AVP_WRITER_PACE", "on"),
        help=(
            "Pace the writer to --fps with realtime + force_fps (default). "
            "'off' publishes grains as fast as the source produces them, "
            "which is only useful for throughput measurements: it runs the "
            "flow into the future and starves readers of history."
        ),
    )
    p.add_argument(
        "--bench-seconds",
        type=int,
        default=int(os.environ.get("AVP_BENCH_SECONDS", 0)),
        help=(
            "Instead of running until killed, sample throughput on the "
            "writer and reader edges once a second for N seconds, print a "
            "summary and exit. Skips teardown, which hangs (see README)."
        ),
    )
    p.add_argument(
        "--bench-warmup",
        type=int,
        default=int(os.environ.get("AVP_BENCH_WARMUP", 5)),
        help="Seconds of --bench-seconds to exclude from the summary.",
    )
    p.add_argument(
        "--no-zero-copy",
        dest="zero_copy",
        action="store_false",
        default=os.environ.get("AVP_MXL_ZERO_COPY", "1") != "0",
        help=(
            "Copy each grain out of shared memory instead of pointing the "
            "AVPacket at it. Zero-copy is the default; disable it if the "
            "reader can fall behind the writer's ring buffer."
        ),
    )
    args = p.parse_args()
    if args.gpu_unpack == "auto":
        args.gpu_unpack = _cuda_present()
        print(f"gpu unpack: {'on' if args.gpu_unpack else 'off'} (auto-detected)")
    else:
        args.gpu_unpack = args.gpu_unpack == "on"
    return args


def _cuda_present() -> bool:
    """True if this container/host has an NVIDIA device node.

    Cheaper and more reliable than probing the driver: the NVIDIA
    Container Toolkit maps /dev/nvidiactl in, and its absence is exactly
    the case where v210_to_cuda cannot run.
    """
    return os.path.exists("/dev/nvidiactl")


def _fps_ratio(fps: str) -> str:
    """Accept '25' as well as '30000/1001'."""
    return fps if "/" in fps else f"{fps}/1"


def _fps_rounded(fps: str) -> int:
    """Nearest whole frame rate, for options that want one (GOP length)."""
    num, _, den = _fps_ratio(fps).partition("/")
    return max(1, round(float(num) / float(den)))


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
    # MXL is a realtime transport: grain N is expected in shared memory
    # at wall-clock N/fps. Nothing upstream paces a lavfi source, and an
    # unpaced writer publishes thousands of grains per second, which
    # shrinks the ring's history to milliseconds and makes every reader
    # "too late". realtime(set_pts) rebases onto the host clock and
    # force_fps pins the cadence the flow advertises.
    if args.writer_pace == "on":
        paced_src = "w_vpaced"
        avp.addNode(Realtime({
            "src": "w_vframe",
            "dst": "w_vrt",
            "group": "w_in",
            "name": "w_realtime",
            "set_pts": True,
            "auto_restart": "off",
        }))
        avp.addNode(ForceFPS({
            "src": "w_vrt",
            "dst": "w_vpaced",
            "group": "w_in",
            "name": "w_fps",
            "fps": _fps_ratio(args.fps),
            "auto_restart": "off",
        }))
    else:
        paced_src = "w_vframe"
    # v210 requires 10-bit 4:2:2 planar. Convert explicitly so the
    # encoder's metadata chain is unambiguous.
    avp.addNode(RescaleVideo({
        "src": paced_src,
        "dst": "w_vscaled",
        "group": "w_in",
        "name": "w_scale",
        "dst_pixel_format": "yuv422p10le",
        # Force the declared geometry: the reader derives the v210 row
        # stride from --width, and packed v210 carries no dimensions, so
        # a source of another size would silently break the contract.
        "dst_width": args.width,
        "dst_height": args.height,
        "auto_restart": "off",
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "w_vscaled",
        "dst": "w_vscaled_assumed",
        "group": "w_in",
        "width": args.width,
        "height": args.height,
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
    """mxl:// -> demux -> v210 unpack -> encode -> file.

    With `--gpu-unpack` the packed grains go straight to the GPU
    (`v210_to_cuda`) and NVENC writes the file; otherwise the CPU v210
    decoder plus swscale feed the mpeg4 encoder.
    """
    from pyplumber.node import RescaleVideo, AssumeVideoFormat

    if args.zero_copy:
        # Zero-copy hands out AVPackets pointing straight into the MXL
        # ring buffer in /dev/shm, with no refcount held on the grain: the
        # bytes stay valid only until the writer laps that slot. Keep the
        # packet queue at one frame so nothing ages while it waits. See
        # also the history_duration note in README.md.
        avp.executeCommandsFromString("queue.plan_capacity r_vpkt 1")

    # `blocking=1` makes the demuxer wait up to one frame period for
    # the next grain instead of returning EAGAIN immediately, so the
    # reader stays open once the writer has produced its first grain.
    # `auto_restart:"group"` still covers the brief startup race
    # before grain 0 is available.
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
            # avformat_open_input's probe already reads a grain, so the
            # index is picked when the node is *created* — seconds before
            # the group starts, since CUDA and NVENC init sit in between.
            # By then it is behind the ring tail; "reset" re-derives it
            # from grain_index_init on the next read instead of failing.
            "on_too_late": "reset",
            # ...which leaves a hole in the timestamps where the skipped
            # grains would have been. Rebase PTS to zero after it so the
            # output file does not start with a multi-second gap.
            "reset_on_drop": "1",
            "zero_copy": "1" if args.zero_copy else "0",
        },
    }))
    avp.addNode(Demux({
        "src": "r_in",
        "routing": {"v:0": "r_vpkt"},
        "group": "r_in",
        "auto_restart": "off",
    }))
    if args.gpu_unpack:
        _build_gpu_reader_tail(avp, args)
    else:
        _build_cpu_reader_tail(avp, args, RescaleVideo, AssumeVideoFormat)
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
        # flush_packets is what actually makes that true: without it the
        # muxer's 256 KiB avio buffer is only written out when it fills,
        # so a low-bitrate run killed after 25 s leaves a 28-byte file
        # holding nothing but the ftyp box.
        "options": {
            "movflags": "frag_keyframe+empty_moov+default_base_moof",
            "flush_packets": "1",
        },
        "group": "r_in",
        "auto_restart": "off",
    }))


def _build_gpu_reader_tail(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> None:
    """r_vpkt -> v210_to_cuda -> scale_cuda -> NVENC -> r_venc.

    One packed frame per grain is exactly `v210_to_cuda`'s input
    contract, so the whole CPU v210 decoder and swscale drop out: the
    grain is copied once into pinned memory, unpacked to CUDA P210 by
    the PTX kernel, and never touches host memory again. NVENC also
    sidesteps the CPU path's mpeg4 fallback (the image has no libx264).
    """
    from pyplumber.node import AssumeVideoFormat, FilterVideo, V210ToCuda

    avp.addNode(V210ToCuda({
        "src": "r_vpkt",
        "dst": "r_vcuda",
        "group": "r_in",
        "name": "r_unpack",
        "hwaccel": _HWACCEL,
        "width": args.width,
        "height": args.height,
        "stride": v210_row_stride(args.width),
        "fps": _fps_ratio(args.fps),
        "timebase": "1/90000",
        "format": "p210le",
        # Packed v210 carries no metadata at all, so the node stamps it.
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "color_range": "tv",
        "auto_restart": "off",
    }))
    # NVENC only takes 4:2:2 10-bit on Blackwell-class hardware, so
    # convert on the GPU to 8-bit 4:2:0 for the demo's output file.
    avp.addNode(FilterVideo({
        "src": "r_vcuda",
        "dst": "r_vnv12",
        "group": "r_in",
        "name": "r_filter",
        "graph": "scale_cuda=format=nv12",
        "hwaccel": _HWACCEL,
        "auto_restart": "off",
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "r_vnv12",
        "dst": "r_vassumed",
        "group": "r_in",
        "width": args.width,
        "height": args.height,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "auto_restart": "off",
    }))
    avp.addNode(EncVideo({
        "src": "r_vassumed",
        "dst": "r_venc",
        "group": "r_in",
        "name": "r_enc",
        "codec": "h264_nvenc",
        "hwaccel": _HWACCEL,
        # One keyframe per second: movflags=frag_keyframe cuts a fragment
        # per keyframe, so the output file becomes playable a second in
        # instead of after NVENC's default 250-frame GOP.
        "options": {"preset": "p4", "profile": "high", "bf": 0,
                    "g": _fps_rounded(args.fps)},
        "auto_restart": "off",
    }))


def _build_cpu_reader_tail(avp: pyplumber.AVPlumber, args: argparse.Namespace,
                           RescaleVideo, AssumeVideoFormat) -> None:
    """r_vpkt -> CPU v210 decode -> swscale -> mpeg4 -> r_venc."""
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
        "width": args.width,
        "height": args.height,
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


def _edge_counter(avp: pyplumber.AVPlumber, name: str):
    """Total-enqueued reader for an edge, or None if it does not exist.

    Edges materialize with their producing node, so this is called after
    the groups have started.
    """
    try:
        edge = avp.getEdge(name, "packet")
    except Exception as exc:                    # edge missing: node failed
        print(f"bench: no edge {name}: {exc}")
        return None
    return lambda: edge.enqueued_total


def _bench(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> int:
    """Sample per-second throughput on the writer and reader edges.

    `enqueued_total` on an edge is a plain counter, so polling it once a
    second costs nothing on the media path — unlike a wiretap callback,
    which would take the GIL per packet.
    """
    target = _fps_rounded(args.fps)
    counters = [(n, c) for n, c in (
        ("writer", None if args.reader_only else _edge_counter(avp, "w_venc")),
        ("reader", None if args.writer_only else _edge_counter(avp, "r_venc")),
    ) if c is not None]
    if not counters:
        print("bench: nothing to measure", file=sys.stderr)
        return 1

    print("bench: elapsed_s," + ",".join(f"{n}_fps" for n, _ in counters), flush=True)
    samples: dict[str, list[float]] = {n: [] for n, _ in counters}
    start = time.monotonic()
    prev_t = start
    prev = {n: c() for n, c in counters}
    while (elapsed := time.monotonic() - start) < args.bench_seconds:
        time.sleep(1.0)
        avp.heartbeat()
        now = time.monotonic()
        dt = now - prev_t
        row = []
        for name, counter in counters:
            total = counter()
            fps = (total - prev[name]) / dt
            prev[name] = total
            row.append(fps)
            # Skip the warmup and, beyond it, the leading zeros: the
            # reader only starts producing once CUDA, NVENC and the MXL
            # attach are done, and that startup is not a stall.
            if now - start > args.bench_warmup and (fps > 0 or samples[name]):
                samples[name].append(fps)
        prev_t = now
        print(f"bench: {now - start:6.1f}," +
              ",".join(f"{fps:8.2f}" for fps in row), flush=True)

    print(f"bench: target {target} fps, "
          f"{args.width}x{args.height}, "
          f"pace {args.writer_pace}, "
          f"gpu_unpack {'on' if args.gpu_unpack else 'off'}, "
          f"zero_copy {'on' if args.zero_copy else 'off'}")
    for name, counter in counters:
        got = samples[name]
        if not got:
            continue
        mean = sum(got) / len(got)
        print(f"bench: {name}: mean {mean:.2f} fps "
              f"(min {min(got):.2f}, max {max(got):.2f}) "
              f"over {len(got)} s, "
              f"{mean / target * 100:.1f}% of target, "
              f"{counter()} frames total")
    # Teardown hangs (see README); the mp4 is flushed per packet, so the
    # output is complete without it.
    sys.stdout.flush()
    os._exit(0)


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
        if args.gpu_unpack:
            avp.executeCommandsFromString(
                'hwaccel.init { "name": "%s", "type": "cuda" }' % _HWACCEL
            )
        _build_reader(avp, args)

    if not args.reader_only:
        avp.group("w_in").startNodes()
    if not args.writer_only:
        # Give the writer a moment to open the flow before the reader
        # tries to attach.
        if not args.reader_only:
            time.sleep(1.0)
        avp.group("r_in").startNodes()

    if args.bench_seconds:
        return _bench(avp, args)

    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
