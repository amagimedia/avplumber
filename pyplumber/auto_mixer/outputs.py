"""Program and Janus output graph builders for the auto mixer."""

from __future__ import annotations

import argparse

from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeAudioFormat,
    AssumeVideoFormat,
    EncAudio,
    EncVideo,
    ForceFPS,
    Mux,
    Output,
    ResampleAudio,
)

from .config import (
    AUDIO_CHANNEL_LAYOUT,
    AUDIO_SAMPLE_FORMAT,
    AUDIO_SAMPLE_RATE,
    CANVAS_H,
    CANVAS_W,
    FPS_DEN,
    FPS_NUM,
    HWACCEL,
    OPUS_SAMPLE_FORMAT,
    RTP_PKT_SIZE,
)


def output_format_for_url(url: str) -> str:
    return "flv" if url.startswith("rtmp://") else "mp4"


def rtp_url(host: str, port: int) -> str:
    return f"rtp://{host}:{port}?pkt_size={RTP_PKT_SIZE}"


def build_audio_output(
    avp: AVPlumber,
    audio_edge: str,
    codec: str = "aac",
    *,
    prefix: str = "program",
    group: str = "output",
    bitrate: str = "192k",
    sample_format: str = AUDIO_SAMPLE_FORMAT,
    options: dict | None = None,
    resample: bool = False,
) -> str:
    """Encode the selected program-audio edge.

    Returns the name of the encoded audio edge for use in Mux.
    """
    assumed_edge = f"{prefix}_a_program"
    encoded_edge = f"{prefix}_a_enc"
    enc_options = {"b": bitrate}
    if options:
        enc_options.update(options)
    if resample:
        avp.addNode(ResampleAudio({
            "name": f"{prefix}_resample_audio",
            "src": audio_edge,
            "dst": assumed_edge,
            "dst_sample_rate": AUDIO_SAMPLE_RATE,
            "dst_sample_format": sample_format,
            "dst_channel_layout": AUDIO_CHANNEL_LAYOUT,
            "compensation": 0,
            "group": group,
            "auto_restart": "panic",
        }))
    else:
        avp.addNode(AssumeAudioFormat({
            "name": f"{prefix}_assume_audio",
            "src": audio_edge,
            "dst": assumed_edge,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "sample_format": sample_format,
            "channel_layout": AUDIO_CHANNEL_LAYOUT,
            "group": group,
            "auto_restart": "panic",
        }))
    avp.addNode(EncAudio({
        "name": f"{prefix}_enc_audio",
        "src": assumed_edge,
        "dst": encoded_edge,
        "codec": codec,
        "options": enc_options,
        "group": group,
        "auto_restart": "panic",
    }))
    return encoded_edge


def build_video_output(
    avp: AVPlumber,
    video_edge: str,
    args: argparse.Namespace,
    *,
    prefix: str = "program",
    group: str = "output",
    codec: str | None = None,
    options: dict | None = None,
) -> str:
    """Add fps-normalizer, format hint, encoder.  Returns encoded video edge."""
    fps_edge = f"{prefix}_mixer_norm_fps"
    assumed_edge = f"{prefix}_mixer_norm"
    encoded_edge = f"{prefix}_v_enc"
    enc_options = {
        "b": "8000k",
        "maxrate": "14000k",
        "bufsize": "20000k",
        "g": 60,
        "bf": 0,
        "preset": "p3",
        "profile": "high",
    }
    if options:
        enc_options.update(options)
    avp.addNode(ForceFPS({
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "src": video_edge,
        "dst": fps_edge,
        "group": group,
        "on_error": "panic",
    }))
    avp.addNode(AssumeVideoFormat({
        "name": f"{prefix}_assume_video",
        "src": fps_edge,
        "dst": assumed_edge,
        "width": CANVAS_W,
        "height": CANVAS_H,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "group": group,
        "auto_restart": "panic",
    }))
    avp.addNode(EncVideo({
        "src": assumed_edge,
        "dst": encoded_edge,
        "name": f"{prefix}_enc_video",
        "codec": codec or args.codec,
        "hwaccel": HWACCEL,
        "group": group,
        "on_error": "panic",
        "options": enc_options,
    }))
    return encoded_edge


def build_mux_output(
    avp: AVPlumber,
    encoded_edges: list[str],
    *,
    mux_edge: str,
    output_url: str,
    output_format: str,
    group: str = "output",
    options: dict | None = None,
) -> None:
    avp.addNode(Mux({
        "name": f"{mux_edge}_mux",
        "src": encoded_edges,
        "dst": mux_edge,
        "group": group,
        "ts_sort_wait": 0,
        "on_error": "panic",
    }))
    output_params = {
        "name": f"{mux_edge}_output",
        "format": output_format,
        "url": output_url,
        "src": mux_edge,
        "group": group,
        "auto_restart": "panic",
    }
    if options:
        output_params["options"] = options
    avp.addNode(Output(output_params))


def build_janus_rtp_output(
    avp: AVPlumber,
    video_edge: str,
    audio_edge: str,
    args: argparse.Namespace,
) -> None:
    video_bitrate = f"{args.janus_video_bitrate_kbps}k"
    video_enc_edge = build_video_output(
        avp,
        video_edge,
        args,
        prefix="janus",
        codec=args.janus_video_codec,
        options={
            "b": video_bitrate,
            "maxrate": video_bitrate,
            "bufsize": video_bitrate,
            "g": 60,
            "bf": 0,
            "preset": "p3",
            "profile": "baseline",
            "tune": "ll",
            "zerolatency": 1,
            "delay": 0,
        },
    )
    audio_enc_edge = build_audio_output(
        avp,
        audio_edge,
        codec="opus",
        prefix="janus",
        bitrate=args.janus_audio_bitrate,
        sample_format=OPUS_SAMPLE_FORMAT,
        resample=True,
        options={
            "strict": "-2",
            "opus_delay": 20,
        },
    )
    build_mux_output(
        avp,
        [video_enc_edge],
        mux_edge="janus_video_rtp_mux",
        output_url=rtp_url(args.janus_host, args.janus_video_port),
        output_format="rtp",
        options={"payload_type": args.janus_video_pt},
    )
    build_mux_output(
        avp,
        [audio_enc_edge],
        mux_edge="janus_audio_rtp_mux",
        output_url=rtp_url(args.janus_host, args.janus_audio_port),
        output_format="rtp",
        options={"payload_type": args.janus_audio_pt},
    )
