"""Program and Janus output graph builders for the auto mixer."""

from __future__ import annotations

import argparse

from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeAudioFormat,
    AssumeVideoFormat,
    Bsf,
    EncAudio,
    EncVideo,
    ForceFPS,
    ForceKeyFrame,
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


def rtp_url(host: str, port: int, *, rtcp_port: int | None = None) -> str:
    query = f"pkt_size={RTP_PKT_SIZE}"
    if rtcp_port is not None:
        query += f"&rtcp_port={rtcp_port}"
    return f"rtp://{host}:{port}?{query}"


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
    force_keyframe_name: str | None = None,
    keyframe_interval_sec: str | int | float | None = None,
    repeat_keyframe_headers: bool = False,
) -> str:
    """Add fps-normalizer, format hint, encoder.  Returns encoded video edge."""
    fps_edge = f"{prefix}_mixer_norm_fps"
    keyframe_edge = f"{prefix}_keyframe_marked"
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

    format_source_edge = fps_edge
    if force_keyframe_name or keyframe_interval_sec is not None:
        keyframe_params = {
            "name": force_keyframe_name or f"{prefix}_force_keyframe",
            "src": fps_edge,
            "dst": keyframe_edge,
            "group": group,
            "auto_restart": "panic",
        }
        if keyframe_interval_sec is not None:
            keyframe_params["interval_sec"] = keyframe_interval_sec
        avp.addNode(ForceKeyFrame(keyframe_params))
        format_source_edge = keyframe_edge

    avp.addNode(AssumeVideoFormat({
        "name": f"{prefix}_assume_video",
        "src": format_source_edge,
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

    if repeat_keyframe_headers:
        repeat_headers_edge = f"{prefix}_v_repeat_headers"
        avp.addNode(Bsf({
            "name": f"{prefix}_repeat_headers",
            "src": encoded_edge,
            "dst": repeat_headers_edge,
            "bsf": "dump_extra=freq=keyframe",
            "group": group,
            "auto_restart": "panic",
        }))
        return repeat_headers_edge

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
    restart_on_clean_finish: bool = False,
) -> None:
    clean_finish_action = "on" if restart_on_clean_finish else "off"
    avp.addNode(Mux({
        "name": f"{mux_edge}_mux",
        "src": encoded_edges,
        "dst": mux_edge,
        "group": group,
        "ts_sort_wait": 0,
        "auto_restart": clean_finish_action,
        "on_error": "panic",
    }))
    output_params = {
        "name": f"{mux_edge}_output",
        "format": output_format,
        "url": output_url,
        "src": mux_edge,
        "group": group,
        "auto_restart": clean_finish_action,
        "on_error": "panic",
    }
    if options:
        output_params["options"] = options
    avp.addNode(Output(output_params))


def build_janus_rtp_output(
    avp: AVPlumber,
    video_edge: str,
    audio_edge: str | None,
    args: argparse.Namespace,
) -> None:
    video_bitrate = f"{args.janus_video_bitrate_kbps}k"
    video_enc_edge = build_video_output(
        avp,
        video_edge,
        args,
        prefix="janus",
        codec=args.janus_video_codec,
        force_keyframe_name="janus_force_keyframe",
        keyframe_interval_sec="1/1",
        repeat_keyframe_headers=True,
        options={
            "b": video_bitrate,
            "maxrate": video_bitrate,
            "bufsize": video_bitrate,
            "g": 30,
            "bf": 0,
            "preset": "p6",
            "profile": "baseline",
            "level": "4.0",
            "tune": "ull",
            "rc": "cbr",
            "rc-lookahead": 0,
            "zerolatency": 1,
            "delay": 0,
            "forced-idr": 1,
            "no-scenecut": 1,
            "strict_gop": 1,
            "aud": 1,
            "spatial-aq": 1,
            "temporal-aq": 0,
        },
    )
    build_mux_output(
        avp,
        [video_enc_edge],
        mux_edge="janus_video_rtp_mux",
        output_url=rtp_url(
            args.janus_host,
            args.janus_video_port,
            rtcp_port=args.janus_video_port + 1,
        ),
        output_format="rtp",
        options={
            "payload_type": args.janus_video_pt,
            "rtpflags": "skip_rtcp",
            "ssrc": args.janus_video_ssrc,
        },
        restart_on_clean_finish=True,
    )
    if audio_edge is not None:
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
            [audio_enc_edge],
            mux_edge="janus_audio_rtp_mux",
            output_url=rtp_url(
                args.janus_host,
                args.janus_audio_port,
                rtcp_port=args.janus_audio_port + 1,
            ),
            output_format="rtp",
            options={"payload_type": args.janus_audio_pt},
            restart_on_clean_finish=True,
        )
