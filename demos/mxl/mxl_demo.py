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
        "--reader-tail",
        choices=("encode", "scale", "unpack", "demux"),
        default=os.environ.get("AVP_READER_TAIL", "encode"),
        help=(
            "What the reader does with each grain. 'encode' (default) "
            "writes --output. The three measurement tails drop the data "
            "instead: 'scale' after the conversion to the encoder's format, "
            "'unpack' after the v210 unpack, 'demux' straight out of the "
            "demuxer. They isolate the MXL read, the unpack and the "
            "conversion from the output encoder, which otherwise dominates "
            "the CPU path."
        ),
    )
    p.add_argument(
        "--sws-flags",
        default=os.environ.get("AVP_SWS_FLAGS"),
        help=(
            "swscale flags for the writer's and the CPU reader's rescale "
            "nodes, comma-separated: 'fast_bilinear', 'bilinear', "
            "'neighbor', 'area', ... Unset lets rescale_video choose, which "
            "for these same-size conversions means area. The choice only "
            "moves the chroma resampling cost; see README."
        ),
    )
    p.add_argument(
        "--gpu-scale",
        choices=("off", "writer", "reader", "both"),
        default=os.environ.get("AVP_GPU_SCALE", "off"),
        help=(
            "Do the scaling and color conversion on the GPU instead of in "
            "swscale: the rescale_video node is replaced by "
            "hwupload,scale_cuda,hwdownload. Frames that are already on the "
            "GPU (--gpu-unpack) skip the upload, and frames headed for NVENC "
            "skip the download."
        ),
    )
    p.add_argument(
        "--cuda-interp",
        choices=("nearest", "bilinear", "bicubic", "lanczos"),
        default=os.environ.get("AVP_CUDA_INTERP"),
        help=(
            "scale_cuda interpolation algorithm, the GPU counterpart of "
            "--sws-flags. Unset leaves the filter's default."
        ),
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
        "--reader-encoder",
        choices=("auto", "mpeg4", "nvenc"),
        default=os.environ.get("AVP_READER_ENCODER", "auto"),
        help=(
            "Output encoder of the reader. 'auto' (default) follows "
            "--gpu-unpack: NVENC for GPU frames, mpeg4 otherwise. Naming one "
            "crosses the paths, which is what measures the download and the "
            "upload: GPU unpack into mpeg4 pays one hwdownload, CPU unpack "
            "into NVENC one hwupload."
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
    if args.reader_encoder == "auto":
        args.reader_encoder = "nvenc" if args.gpu_unpack else "mpeg4"
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


def _fps_period(fps: str) -> str:
    """Frame period as a timebase: '60000/1001' -> '1001/60000'."""
    num, _, den = _fps_ratio(fps).partition("/")
    return f"{den}/{num}"


def _fps_rounded(fps: str) -> int:
    """Nearest whole frame rate, for options that want one (GOP length)."""
    num, _, den = _fps_ratio(fps).partition("/")
    return max(1, round(float(num) / float(den)))


def _cuda_convert_graph(args: argparse.Namespace, dst_format: str, *,
                        upload: bool, download: bool,
                        width: int = 0, height: int = 0) -> str:
    """A libavfilter graph that converts on the GPU, as filter_video wants.

    `upload` and `download` bracket the conversion when the frames live in
    host memory on that side. Each is one PCIe transfer of a whole frame,
    in the format on that side of the conversion, so which of the two a
    leg pays for shows up directly in its CPU cost.
    """
    opts = []
    if width:
        opts.append(f"w={width}")
    if height:
        opts.append(f"h={height}")
    opts.append(f"format={dst_format}")
    if args.cuda_interp:
        opts.append(f"interp_algo={args.cuda_interp}")
    scale = "scale_cuda=" + ":".join(opts)
    stages = (["hwupload"] if upload else []) + [scale]
    if download:
        # hwdownload alone yields the frames context's sw_format, which is
        # what scale_cuda just produced; name it anyway so the graph fails
        # loudly rather than silently handing on another layout.
        stages += ["hwdownload", f"format={dst_format}"]
    return ",".join(stages)


def _needs_cuda(args: argparse.Namespace) -> bool:
    """Whether any node of the chosen graph wants the CUDA device."""
    if not args.reader_only and args.gpu_scale in ("writer", "both"):
        return True
    if args.writer_only:
        return False
    if args.reader_tail == "demux":         # never unpacks
        return False
    if args.gpu_unpack:
        return True
    if args.reader_tail == "unpack":        # CPU unpack, no tail to run
        return False
    return args.gpu_scale in ("reader", "both") or args.reader_encoder == "nvenc"


def _sws_flags(spec: str | None) -> list[str]:
    """'fast_bilinear' -> ['SWS_FAST_BILINEAR'], which rescale_video wants."""
    if not spec:
        return []
    flags = []
    for name in spec.split(","):
        name = name.strip().upper()
        flags.append(name if name.startswith("SWS_") else "SWS_" + name)
    return flags


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
        "name": "w_demux",
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
    # encoder's metadata chain is unambiguous. This up-convert is the
    # writer's single most expensive step, so it takes --sws-flags — or
    # --gpu-scale, which hands the whole conversion to scale_cuda.
    #
    # Either way the declared geometry is forced: the reader derives the
    # v210 row stride from --width, and packed v210 carries no dimensions,
    # so a source of another size would silently break the contract.
    if args.gpu_scale in ("writer", "both"):
        from pyplumber.node import FilterVideo

        avp.addNode(FilterVideo({
            "src": paced_src,
            "dst": "w_vscaled",
            "group": "w_in",
            "name": "w_scale",
            "graph": _cuda_convert_graph(args, "yuv422p10le",
                                         upload=True, download=True,
                                         width=args.width, height=args.height),
            "hwaccel": _HWACCEL,
            "auto_restart": "off",
        }))
    else:
        scale_params = {
            "src": paced_src,
            "dst": "w_vscaled",
            "group": "w_in",
            "name": "w_scale",
            "dst_pixel_format": "yuv422p10le",
            "dst_width": args.width,
            "dst_height": args.height,
            "auto_restart": "off",
        }
        if flags := _sws_flags(args.sws_flags):
            scale_params["flags"] = flags
        avp.addNode(RescaleVideo(scale_params))
    avp.addNode(AssumeVideoFormat({
        "src": "w_vscaled",
        "dst": "w_vscaled_assumed",
        "group": "w_in",
        "name": "w_assume",
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
        "name": "w_mux",
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
        "name": "w_output",
        "auto_restart": "off",
    }))


def _drop(avp: pyplumber.AVPlumber, src: str) -> None:
    """Sink an edge into nothing, which is what the measurement tails do."""
    from pyplumber.node import NullSink

    avp.addNode(NullSink({
        "src": src,
        "group": "r_in",
        "name": "r_sink",
        "auto_restart": "off",
    }))


def _build_reader(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> None:
    """mxl:// -> demux -> v210 unpack -> convert -> encode -> file.

    With `--gpu-unpack` the packed grains go straight to the GPU
    (`v210_to_cuda`) and NVENC writes the file; otherwise the CPU v210
    decoder plus swscale feed the mpeg4 encoder. `--gpu-unpack` and
    `--reader-encoder` pick those two ends independently, and the
    conversion in between follows from them. `--reader-tail` cuts the
    chain short after any of the three, for measuring them in isolation.
    """
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
        "name": "r_demux",
        "auto_restart": "off",
    }))
    if args.reader_tail == "demux":
        _drop(avp, "r_vpkt")
        return
    unpacked = (_build_gpu_unpack(avp, args) if args.gpu_unpack
                else _build_cpu_unpack(avp, args))
    if args.reader_tail == "unpack":
        _drop(avp, unpacked)
        return
    converted = _build_convert(avp, args, unpacked)
    if args.reader_tail == "scale":
        _drop(avp, converted)
        return
    _build_encoder(avp, args, converted)
    avp.addNode(Mux({
        "src": ["r_venc"],
        "dst": "r_mux",
        "group": "r_in",
        "name": "r_mux",
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
        "name": "r_output",
        "auto_restart": "off",
    }))


def _build_gpu_unpack(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> str:
    """r_vpkt -> v210_to_cuda -> r_vcuda, returning the frame edge.

    One packed frame per grain is exactly `v210_to_cuda`'s input
    contract, so the whole CPU v210 decoder and swscale drop out: the
    grain is copied once into pinned memory, unpacked to CUDA P210 by
    the PTX kernel, and never touches host memory again.
    """
    from pyplumber.node import V210ToCuda

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
        # Packed v210 has no timebase either, so the node picks one. 90 kHz
        # is the usual choice, but mpeg4 refuses any denominator above
        # 65535, so stamp the frame period when it is the output encoder.
        "timebase": ("1/90000" if args.reader_encoder == "nvenc"
                     else _fps_period(args.fps)),
        "format": "p210le",
        # Packed v210 carries no metadata at all, so the node stamps it.
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "color_range": "tv",
        "auto_restart": "off",
    }))
    return "r_vcuda"


def _build_cpu_unpack(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> str:
    """r_vpkt -> libavcodec v210 decoder -> r_vframe (yuv422p10le)."""
    avp.addNode(DecVideo({
        "src": "r_vpkt",
        "dst": "r_vframe",
        "group": "r_in",
        "name": "r_dec",
        "auto_restart": "off",
    }))
    return "r_vframe"


def _build_convert(avp: pyplumber.AVPlumber, args: argparse.Namespace,
                   src: str) -> str:
    """Bring the unpacked frames to what the output encoder takes.

    NVENC wants CUDA nv12 -- it only accepts 4:2:2 10-bit on
    Blackwell-class hardware -- and mpeg4 wants host yuv420p, so either
    way the 10-bit 4:2:2 flow is converted here. The frames arrive either
    on the GPU (`v210_to_cuda`) or in host memory (the `v210` decoder),
    which leaves four shapes for the same step: swscale, scale_cuda with
    an upload and a download around it (`--gpu-scale reader`), scale_cuda
    plus one download, or scale_cuda alone. It is the reader's most
    expensive step after the encoder itself.
    """
    from pyplumber.node import RescaleVideo, AssumeVideoFormat, FilterVideo

    gpu_in = args.gpu_unpack
    gpu_out = args.reader_encoder == "nvenc"
    dst_format = "nv12" if gpu_out else "yuv420p"
    if gpu_in or gpu_out or args.gpu_scale in ("reader", "both"):
        avp.addNode(FilterVideo({
            "src": src,
            "dst": "r_vconv",
            "group": "r_in",
            "name": "r_scale",
            "graph": _cuda_convert_graph(args, dst_format,
                                         upload=not gpu_in,
                                         download=not gpu_out),
            "hwaccel": _HWACCEL,
            "auto_restart": "off",
        }))
    else:
        scale_params = {
            "src": src,
            "dst": "r_vconv",
            "group": "r_in",
            "name": "r_scale",
            "dst_pixel_format": dst_format,
            "auto_restart": "off",
        }
        if flags := _sws_flags(args.sws_flags):
            scale_params["flags"] = flags
        avp.addNode(RescaleVideo(scale_params))
    avp.addNode(AssumeVideoFormat({
        "src": "r_vconv",
        "dst": "r_vassumed",
        "group": "r_in",
        "name": "r_assume",
        "width": args.width,
        "height": args.height,
        "pixel_format": "cuda" if gpu_out else dst_format,
        "real_pixel_format": dst_format,
        "auto_restart": "off",
    }))
    return "r_vassumed"


def _build_encoder(avp: pyplumber.AVPlumber, args: argparse.Namespace,
                   src: str) -> None:
    """The reader's output encoder: NVENC on the GPU, else mpeg4.

    NVENC sidesteps the CPU path's mpeg4 fallback, which the demo uses
    only because the image does not link libx264.
    """
    if args.reader_encoder != "nvenc":
        avp.addNode(EncVideo({
            "src": src,
            "dst": "r_venc",
            "group": "r_in",
            "name": "r_enc",
            "codec": "mpeg4",
            "auto_restart": "off",
        }))
        return
    avp.addNode(EncVideo({
        "src": src,
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


def _edge_counter(avp: pyplumber.AVPlumber, name: str, timeout: float = 30.0):
    """Total-enqueued reader for an edge, or None if it never appears.

    `startNodes()` only queues the group's state change, so the edges
    materialize on an avplumber worker thread some time after it returns —
    CUDA and NVENC init keep the reader busy for seconds. Wait for the
    edge instead of asking `getEdge()` for it: with no edge of that name
    yet, `getEdge()` *creates* one of the requested type, and the real
    node then fails forever with "Edge r_vcuda has type av::Packet, not
    av::VideoFrame". The stats JSON is the type-agnostic view, so it
    neither creates an edge nor has to guess the queue type.
    """
    deadline = time.monotonic() + timeout
    while not any(edge["name"] == name for edge in avp.edges.edgesStatsJson()):
        if time.monotonic() > deadline:
            print(f"bench: no edge {name} after {timeout:.0f} s: node failed?")
            return None
        time.sleep(0.1)
    edge = avp.getEdge(name)
    return lambda: edge.enqueued_total


def _reader_bench_edge(args: argparse.Namespace) -> str:
    """Last edge of the reader chain, which depends on --reader-tail."""
    if args.reader_tail == "encode":
        return "r_venc"
    if args.reader_tail == "scale":
        return "r_vassumed"
    if args.reader_tail == "demux":
        return "r_vpkt"
    return "r_vcuda" if args.gpu_unpack else "r_vframe"


_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _thread_cpu() -> dict[str, float]:
    """CPU seconds consumed so far, per thread name.

    Every avplumber node runs in a thread named after the node, so
    /proc/self/task is a per-node profiler that costs nothing on the media
    path. Linux keeps 15 characters of each name; nodes created without a
    "name" show up as "<type>@<address>".
    """
    totals: dict[str, float] = {}
    for task in os.scandir("/proc/self/task"):
        try:
            with open(os.path.join(task.path, "stat")) as stat_file:
                stat = stat_file.read()
        except OSError:                 # thread exited while we were reading
            continue
        name = stat[stat.index("(") + 1:stat.rindex(")")]
        # Fields after comm: state is 3, so utime (14) and stime (15) are
        # at index 11 and 12.
        fields = stat[stat.rindex(")") + 2:].split()
        cpu = (int(fields[11]) + int(fields[12])) / _CLK_TCK
        totals[name] = totals.get(name, 0.0) + cpu
    return totals


def _print_thread_cpu(before: dict[str, float], elapsed: float,
                      frames: dict[str, int]) -> None:
    """Per-node CPU over the measured window, most expensive node first.

    `frames` maps a node name prefix ("w_", "r_") to the number of frames
    that half of the demo produced in the window, which turns CPU seconds
    into the per-frame cost of each node.
    """
    rows = []
    after = _thread_cpu()
    for name, cpu in after.items():
        delta = cpu - before.get(name, 0.0)
        if delta >= 0.01:
            rows.append((name, delta))
    rows.sort(key=lambda row: -row[1])
    total = sum(delta for _, delta in rows)
    print(f"bench: per-node CPU over {elapsed:.1f} s, "
          f"{total:.1f} s total = {total / elapsed:.2f} cores")
    for name, delta in rows:
        count = frames.get(name[:2], 0)
        per_frame = f", {delta / count * 1000:6.2f} ms/frame" if count else ""
        print(f"bench:   {name:<16} {delta:6.2f} s, "
              f"{delta / elapsed:5.2f} cores{per_frame}")


def _bench(avp: pyplumber.AVPlumber, args: argparse.Namespace) -> int:
    """Sample per-second throughput on the writer and reader edges.

    `enqueued_total` on an edge is a plain counter, so polling it once a
    second costs nothing on the media path — unlike a wiretap callback,
    which would take the GIL per packet.
    """
    target = _fps_rounded(args.fps)
    counters = [(n, p, c) for n, p, c in (
        ("writer", "w_", None if args.reader_only else _edge_counter(avp, "w_venc")),
        ("reader", "r_", None if args.writer_only
         else _edge_counter(avp, _reader_bench_edge(args))),
    ) if c is not None]
    if not counters:
        print("bench: nothing to measure", file=sys.stderr)
        return 1

    print("bench: elapsed_s," + ",".join(f"{n}_fps" for n, _, _ in counters), flush=True)
    samples: dict[str, list[float]] = {n: [] for n, _, _ in counters}
    start = time.monotonic()
    prev_t = start
    prev = {n: c() for n, _, c in counters}
    # The per-node CPU window opens once the warmup is over, so that CUDA,
    # NVENC and MXL attach do not land in the per-frame figures.
    cpu_before: dict[str, float] | None = None
    frames_before: dict[str, int] = {}
    window_start = start
    while (elapsed := time.monotonic() - start) < args.bench_seconds:
        time.sleep(1.0)
        avp.heartbeat()
        now = time.monotonic()
        dt = now - prev_t
        row = []
        for name, _, counter in counters:
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
        if cpu_before is None and now - start > args.bench_warmup:
            cpu_before = _thread_cpu()
            frames_before = {p: c() for _, p, c in counters}
            window_start = now
        print(f"bench: {now - start:6.1f}," +
              ",".join(f"{fps:8.2f}" for fps in row), flush=True)

    print(f"bench: target {target} fps, "
          f"{args.width}x{args.height}, "
          f"pace {args.writer_pace}, "
          f"gpu_unpack {'on' if args.gpu_unpack else 'off'}, "
          f"zero_copy {'on' if args.zero_copy else 'off'}, "
          f"reader_tail {args.reader_tail}, "
          f"reader_encoder {args.reader_encoder}, "
          f"gpu_scale {args.gpu_scale}, "
          f"cuda_interp {args.cuda_interp or 'default'}, "
          f"sws_flags {args.sws_flags or 'default'}")
    for name, _, counter in counters:
        got = samples[name]
        if not got:
            continue
        mean = sum(got) / len(got)
        print(f"bench: {name}: mean {mean:.2f} fps "
              f"(min {min(got):.2f}, max {max(got):.2f}) "
              f"over {len(got)} s, "
              f"{mean / target * 100:.1f}% of target, "
              f"{counter()} frames total")
    if cpu_before is not None:
        window = time.monotonic() - window_start
        _print_thread_cpu(cpu_before, window,
                          {p: c() - frames_before[p] for _, p, c in counters})
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

    # Before either half is built: --gpu-scale puts CUDA filters in the
    # writer too, and a node's create() resolves the device by name.
    if _needs_cuda(args):
        avp.executeCommandsFromString(
            'hwaccel.init { "name": "%s", "type": "cuda" }' % _HWACCEL
        )
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
