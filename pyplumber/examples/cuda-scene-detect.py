#!/usr/bin/env python3
# CUDA-accelerated scene detection using pyplumber.scene_detect.
# All detection runs directly on NV12 CUDA frames — no hwdownload/CPU round-trip.
#
# Usage:
#   AVP_INPUT=/path/to/video.mp4 python3 pyplumber/examples/cuda-scene-detect.py
#
# Tunables (environment variables):
#
#   Input / output
#     AVP_INPUT                  input URL or local file (default: input.mp4)
#     AVP_OUTPUT                 optional MP4 output path with overlay
#     AVP_OUTPUT_WIDTH           output frame width  (default: 1920)
#     AVP_OUTPUT_HEIGHT          output frame height (default: 1080)
#     AVP_OUTPUT_BITRATE         NVENC target bitrate for file output (default: 6000k)
#     AVP_OUTPUT_FPS             output timebase (default: 25/1)
#     AVP_INPUT_LOOP             loop input (default: 1 when Janus output, else 0)
#
#   Detection algorithm
#     AVP_DETECTOR_TYPE          content | adaptive | histogram | threshold (default: content)
#     AVP_SCENE_THRESHOLD        cut threshold — meaning depends on detector:
#                                  content:   absolute HSV-delta score  (default: 27.0)
#                                  adaptive:  score/neighbour-avg ratio (default: 3.0)
#                                  histogram: luma correlation, lower=more sensitive (default: 0.95)
#                                  threshold: mean-luma level 0-255 (default: 12.0)
#     AVP_SCENE_MIN_LEN          minimum frames between cuts (default: 15)
#     AVP_CONFIRM_FRAMES         consecutive over-threshold frames required (default: 3,
#                                  recommended 1 for adaptive which has its own suppression)
#     AVP_HIST_BINS              histogram bins per channel (default: 256)
#     AVP_LUMA_ONLY              content/adaptive: use only V channel, skip H/S (default: false)
#     AVP_WINDOW_WIDTH           adaptive: rolling-window half-width in frames (default: 2)
#     AVP_MIN_CONTENT_VAL        adaptive: minimum raw score to trigger a cut (default: 15.0)
#     AVP_FADE_BIAS              threshold: bias toward fade-in (+) or fade-out (-) (default: 0.0)
#
#   Analysis resolution
#     AVP_SCENE_WIDTH            analysis frame width  — smaller is faster (default: 640)
#     AVP_SCENE_HEIGHT           analysis frame height (default: 360)
#
#   Janus WebRTC output
#     AVP_JANUS_OUTPUT           1 to stream to Janus (default: 0)
#     AVP_JANUS_HOST             Janus RTP host (default: 127.0.0.1)
#     AVP_JANUS_VIDEO_PORT       RTP video port; RTCP = port+1 (default: 5004)
#     AVP_JANUS_VIDEO_BITRATE_KBPS NVENC bitrate in kbps (default: 3000)
#
#   Metadata store
#     AVP_METADATA_KAFKA         1 to enable writing metadata to Kafka (default: 0)
#     AVP_METADATA_KAFKA_BROKERS Kafka bootstrap servers (default: localhost:9092)
#     AVP_METADATA_KAFKA_TOPIC   Kafka topic to publish metadata to (default: scene-metadata)
#     AVP_METADATA_KEYS          comma-separated metadata key names, or * for all (default: scene_overlay)
#
#   Monitoring
#     AVP_WEBUI_API              Web UI endpoint, e.g. http://127.0.0.1:22222
#     AVP_REMOTE_CONTROL_PORT    control server port for Web UI (required with AVP_WEBUI_API)
#     AVP_INSTANCE_NAME          Web UI instance label (default: scene-detect-cuda)
#     AVP_LOGFILE                write avplumber log to this file
#     AVP_IDLE_TIMEOUT           seconds without frames before treating as EOF (default: 3.0)

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    Bsf,
    DecVideo,
    Demux,
    DrawBBoxLabels,
    EncVideo,
    FilterVideo,
    ForceFPS,
    ForceKeyFrame,
    InputRec,
    JoinMetadata,
    Mux,
    OneToMany,
    Output,
    Realtime,
    SmoothTimestamps,
)
from pyplumber.scene_detect import CudaSceneDetectNode
from pyplumber.metadata_write import MetadataStoreNode
from pyplumber.auto_mixer.config import (
    JANUS_DEFAULT_HOST,
    JANUS_DEFAULT_VIDEO_BITRATE_KBPS,
    JANUS_DEFAULT_VIDEO_PORT,
    JANUS_DEFAULT_VIDEO_SSRC,
    RTP_PKT_SIZE,
)
from pyplumber.auto_mixer.rtcp_feedback import RtcpFeedbackListener


# ── Input / output ────────────────────────────────────────────────────────────
INPUT_URL       = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL      = os.environ.get("AVP_OUTPUT", "")
OUTPUT_WIDTH    = int(os.environ.get("AVP_OUTPUT_WIDTH",  "1920"))
OUTPUT_HEIGHT   = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
OUTPUT_BITRATE  = os.environ.get("AVP_OUTPUT_BITRATE", "6000k")
OUTPUT_FPS      = os.environ.get("AVP_OUTPUT_FPS", "25/1")
OVERLAY_KEY     = os.environ.get("AVP_SCENE_OVERLAY_METADATA_KEY", "scene_overlay")
WEBUI_API       = os.environ.get("AVP_WEBUI_API", "")
INSTANCE_NAME   = os.environ.get("AVP_INSTANCE_NAME", "scene-detect-cuda")
LOGFILE         = os.environ.get("AVP_LOGFILE", "")
REMOTE_CONTROL_PORT = int(os.environ.get("AVP_REMOTE_CONTROL_PORT", "0") or "0")
IDLE_TIMEOUT    = float(os.environ.get("AVP_IDLE_TIMEOUT", "3.0"))

# ── Detection algorithm ───────────────────────────────────────────────────────
DETECTOR_TYPE   = os.environ.get("AVP_DETECTOR_TYPE", "content").strip().lower()

# Per-detector threshold defaults; user override via AVP_SCENE_THRESHOLD
_THRESHOLD_DEFAULTS = {"content": 27.0, "adaptive": 3.0, "histogram": 0.95, "threshold": 12.0}
_thr_env = os.environ.get("AVP_SCENE_THRESHOLD")
SCENE_THRESHOLD = float(_thr_env) if _thr_env not in (None, "") \
                  else _THRESHOLD_DEFAULTS.get(DETECTOR_TYPE, 27.0)

SCENE_MIN_LEN   = int(os.environ.get("AVP_SCENE_MIN_LEN", "15"))
CONFIRM_FRAMES  = int(os.environ.get("AVP_CONFIRM_FRAMES", "3"))
HIST_BINS       = int(os.environ.get("AVP_HIST_BINS", "256"))
LUMA_ONLY       = os.environ.get("AVP_LUMA_ONLY", "false").strip().lower() in ("1", "true", "yes")
WINDOW_WIDTH    = int(os.environ.get("AVP_WINDOW_WIDTH", "2"))
MIN_CONTENT_VAL = float(os.environ.get("AVP_MIN_CONTENT_VAL", "15.0"))
FADE_BIAS       = float(os.environ.get("AVP_FADE_BIAS", "0.0"))

# ── Analysis resolution ───────────────────────────────────────────────────────
SCENE_WIDTH     = int(os.environ.get("AVP_SCENE_WIDTH",  "640"))
SCENE_HEIGHT    = int(os.environ.get("AVP_SCENE_HEIGHT", "360"))

# ── Janus ─────────────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

def _parse_int_auto_base(s: str) -> int:
    return int(s, 0)

JANUS_OUTPUT            = _env_bool("AVP_JANUS_OUTPUT")
JANUS_HOST              = os.environ.get("AVP_JANUS_HOST", JANUS_DEFAULT_HOST)
JANUS_VIDEO_PORT        = int(os.environ.get("AVP_JANUS_VIDEO_PORT", str(JANUS_DEFAULT_VIDEO_PORT)))
JANUS_VIDEO_PT          = int(os.environ.get("AVP_JANUS_VIDEO_PT", "96"))
JANUS_VIDEO_SSRC        = _parse_int_auto_base(
    os.environ.get("AVP_JANUS_VIDEO_SSRC", f"0x{JANUS_DEFAULT_VIDEO_SSRC:x}"))
JANUS_VIDEO_BITRATE_KBPS = int(
    os.environ.get("AVP_JANUS_VIDEO_BITRATE_KBPS", str(JANUS_DEFAULT_VIDEO_BITRATE_KBPS)))
JANUS_RTCP_FEEDBACK     = not _env_bool("AVP_DISABLE_JANUS_RTCP_FEEDBACK")
JANUS_RTCP_FEEDBACK_BIND = os.environ.get("AVP_JANUS_RTCP_FEEDBACK_BIND", "0.0.0.0")
JANUS_RTCP_FEEDBACK_PORT = int(os.environ.get("AVP_JANUS_RTCP_FEEDBACK_PORT", "0") or "0")
INPUT_LOOP              = _env_bool("AVP_INPUT_LOOP", JANUS_OUTPUT)

# ── Metadata store ────────────────────────────────────────────────────────────
METADATA_KAFKA         = _env_bool("AVP_METADATA_KAFKA")
METADATA_KAFKA_BROKERS = os.environ.get("AVP_METADATA_KAFKA_BROKERS", "localhost:9092")
METADATA_KAFKA_TOPIC   = os.environ.get("AVP_METADATA_KAFKA_TOPIC", "scene-metadata")
METADATA_KEYS          = os.environ.get("AVP_METADATA_KEYS", "scene_overlay")


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _cuda_decoder(src: str, dst: str, name: str) -> DecVideo:
    return DecVideo({
        "src": src, "dst": dst, "group": "scene", "name": name,
        "auto_restart": "off", "optional": True,
        "pixel_format": "?cuda", "hwaccel": "@gpu",
        "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
        "hwaccel_only_for_codecs": ["h264", "hevc"],
    })


def _analysis_scale_filter(src: str) -> FilterVideo:
    return FilterVideo({
        "src": src, "dst": "v_nv12_cuda",
        "group": "scene", "name": "Cuda_Scale_For_Scene_Detect",
        "graph": (
            f"scale_cuda=w={SCENE_WIDTH}:h={SCENE_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        ),
        "dst_width": SCENE_WIDTH, "dst_height": SCENE_HEIGHT,
        "dst_pixel_format": "cuda", "hwaccel": "@gpu",
        "auto_restart": "off",
    })


def _rtp_url(host: str, port: int, *, rtcp_port: int | None = None) -> str:
    q = f"pkt_size={RTP_PKT_SIZE}"
    if rtcp_port is not None:
        q += f"&rtcp_port={rtcp_port}"
    return f"rtp://{host}:{port}?{q}"


def _packet_realtime_node(src: str, dst: str) -> Realtime:
    return Realtime({
        "team": "scene-demux-rt",
        "negative_time_tolerance": 0.02, "negative_time_discard": 0.02,
        "jitter_margin": 0.1, "initial_jitter_margin": 0.1,
        "discontinuity_threshold": 3,
        "src": src, "dst": dst, "group": "in",
        "name": "Demux_Realtime", "auto_restart": "on",
    })


def _file_timestamp_node(src: str, dst: str) -> SmoothTimestamps:
    return SmoothTimestamps({
        "src": src, "dst": dst, "fps": OUTPUT_FPS,
        "drift_window": 0, "discontinuity_threshold": 86400,
        "group": "scene", "name": "File_Timestamps", "auto_restart": "on",
    })


def _janus_output_nodes(src_edge: str) -> list:
    bitrate = f"{JANUS_VIDEO_BITRATE_KBPS}k"
    return [
        ForceKeyFrame({
            "name": "janus_force_keyframe", "interval_sec": "1/1",
            "group": "scene", "src": src_edge, "dst": "janus_keyframed",
            "auto_restart": "panic",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda", "real_pixel_format": "nv12",
            "group": "scene", "src": "janus_keyframed", "dst": "janus_preenc",
            "auto_restart": "panic",
        }),
        EncVideo({
            "src": "janus_preenc", "dst": "janus_v_enc",
            "group": "scene", "name": "Janus_Encode_NVENC",
            "codec": "h264_nvenc", "hwaccel": "@gpu", "on_error": "panic",
            "options": {
                "b": bitrate, "maxrate": bitrate, "bufsize": bitrate,
                "g": 30, "bf": 0, "preset": "p6", "profile": "baseline",
                "level": "4.0", "tune": "ull", "rc": "cbr",
                "rc-lookahead": 0, "zerolatency": 1, "delay": 0,
                "forced-idr": 1, "no-scenecut": 1, "strict_gop": 1,
                "aud": 1, "spatial-aq": 1, "temporal-aq": 0,
            },
        }),
        Bsf({
            "name": "janus_repeat_headers", "bsf": "dump_extra=freq=keyframe",
            "src": "janus_v_enc", "dst": "janus_v_repeat_headers",
            "group": "scene", "auto_restart": "panic",
        }),
        Mux({
            "src": ["janus_v_repeat_headers"], "dst": "janus_video_rtp_mux",
            "group": "scene", "ts_sort_wait": 0,
            "auto_restart": "on", "on_error": "panic",
        }),
        Output({
            "format": "rtp",
            "url": _rtp_url(JANUS_HOST, JANUS_VIDEO_PORT, rtcp_port=JANUS_VIDEO_PORT + 1),
            "src": "janus_video_rtp_mux", "group": "scene",
            "name": "Janus_RTP_Output", "auto_restart": "on", "on_error": "panic",
            "options": {
                "payload_type": JANUS_VIDEO_PT,
                "rtpflags": "skip_rtcp",
                "ssrc": JANUS_VIDEO_SSRC,
            },
        }),
    ]


def _file_output_nodes(src_edge: str) -> list:
    return [
        ForceKeyFrame({
            "interval_sec": "2/1", "group": "scene",
            "src": src_edge, "dst": "v_keyframed",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda", "real_pixel_format": "nv12",
            "group": "scene", "src": "v_keyframed", "dst": "v_preenc",
            "auto_restart": "off",
        }),
        EncVideo({
            "src": "v_preenc", "dst": "v_outenc", "group": "scene",
            "name": "Encode_NVENC", "codec": "h264_nvenc", "hwaccel": "@gpu",
            "options": {
                "b": OUTPUT_BITRATE, "maxrate": OUTPUT_BITRATE, "bufsize": OUTPUT_BITRATE,
                "rc": "cbr", "g": 50, "bf": 0, "preset": "p1",
                "profile": "high", "level": "4.0", "forced-idr": 1,
                "no-scenecut": 1, "strict_gop": 1, "aud": 1,
            },
        }),
        Mux({"src": ["v_outenc"], "dst": "mux_v", "group": "scene", "ts_sort_wait": 0}),
        Output({
            "format": "mp4", "url": OUTPUT_URL,
            "src": "mux_v", "group": "scene", "name": "Output_MP4",
            "auto_restart": "off",
        }),
    ]


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    avp = pyplumber.AVPlumber()
    if LOGFILE:
        avp.setLogFile(LOGFILE)
    if REMOTE_CONTROL_PORT:
        avp.enableControlServer(REMOTE_CONTROL_PORT)
    if WEBUI_API:
        avp.registerWithWebUI(WEBUI_API, INSTANCE_NAME, LOGFILE)
    avp.edges.planCapacity("*", 15)

    render_output = bool(OUTPUT_URL) or JANUS_OUTPUT
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')

    split_src  = "v_file_ts"   if JANUS_OUTPUT else "v_dec_cuda"
    input_dst  = "in_mux_raw"  if JANUS_OUTPUT else "in_mux"

    nodes = [
        InputRec({
            "url": INPUT_URL, "dst": input_dst, "group": "in",
            "name": "input", "timeout": -1, "preseek": 0,
            "loop": INPUT_LOOP, "auto_restart": "off",
        }),
    ]
    if JANUS_OUTPUT:
        nodes.append(_packet_realtime_node(input_dst, "in_mux"))
    nodes.extend([
        Demux({
            "src": "in_mux", "routing": {"v:0": "v_pkt"},
            "group": "in", "auto_restart": "off",
        }),
        _cuda_decoder("v_pkt", "v_dec_cuda", "Decode_CUDA"),
    ])
    if JANUS_OUTPUT:
        nodes.append(_file_timestamp_node("v_dec_cuda", split_src))

    if render_output:
        nodes.extend([
            OneToMany({
                "src": split_src, "dst": ["v_full_cuda", "v_analysis_cuda"],
                "outputs": 3, "group": "scene",
                "name": "Split_Full_And_Scene_Detect", "auto_restart": "off",
            }),
            _analysis_scale_filter(src="v_analysis_cuda"),
        ])
    else:
        nodes.append(_analysis_scale_filter(src=split_src))

    # Determine detector dst edge and Kafka chain.
    # Each avplumber edge has exactly one consumer, so we must chain nodes
    # sequentially rather than tap the same edge from two nodes.
    #
    # render + kafka:   scale → detector(dst=v_scene_overlay_md)
    #                         → MetadataStore(src=v_scene_overlay_md, dst=v_scene_md_stored)
    #                         → JoinMetadata uses v_scene_md_stored
    #
    # no-render + kafka: scale → detector(dst=v_detect_pass)
    #                          → MetadataStore(src=v_detect_pass, no dst — sink)
    #
    # no kafka:          scale → detector(dst=v_scene_overlay_md or no dst)

    if render_output:
        detector_dst = "v_scene_overlay_md"
    elif METADATA_KAFKA:
        detector_dst = "v_detect_pass"
    else:
        detector_dst = None

    detector_params = {
        "src": "v_nv12_cuda",
        **({"dst": detector_dst} if detector_dst else {}),
        "group": "scene",
        "name": "cuda-scene-detect",
        "detector_type": DETECTOR_TYPE,
        "threshold": SCENE_THRESHOLD,
        "min_scene_len": SCENE_MIN_LEN,
        "confirm_frames": CONFIRM_FRAMES,
        "hist_bins": HIST_BINS,
        "luma_only": str(LUMA_ONLY).lower(),
        "window_width": WINDOW_WIDTH,
        "min_content_val": MIN_CONTENT_VAL,
        "fade_bias": FADE_BIAS,
        "metadata_key": OVERLAY_KEY if (render_output or METADATA_KAFKA) else "",
    }
    detector = CudaSceneDetectNode(detector_params)
    nodes.append(detector)

    _kafka_backend_cfg = {
        "type": "kafka",
        "bootstrap_servers": METADATA_KAFKA_BROKERS,
        "topic": METADATA_KAFKA_TOPIC,
    }
    if METADATA_KAFKA and render_output:
        # Pass-through: reads detector output, forwards to JoinMetadata
        nodes.append(MetadataStoreNode({
            "src": "v_scene_overlay_md",
            "dst": "v_scene_md_stored",
            "group": "scene",
            "name": "MetadataToKafka",
            "backend": _kafka_backend_cfg,
            "metadata": METADATA_KEYS,
        }))
        _meta_edge = "v_scene_md_stored"
    elif METADATA_KAFKA and not render_output:
        # Sink: reads detector pass-through, no downstream consumer
        nodes.append(MetadataStoreNode({
            "src": "v_detect_pass",
            "group": "scene",
            "name": "MetadataToKafka",
            "backend": _kafka_backend_cfg,
            "metadata": METADATA_KEYS,
        }))
        _meta_edge = "v_scene_overlay_md"
    else:
        _meta_edge = "v_scene_overlay_md"

    if render_output:
        nodes.extend([
            JoinMetadata({
                "src": ["v_full_cuda", _meta_edge],
                "dst": "v_full_with_scene_md",
                "group": "scene", "auto_restart": "off",
            }),
            DrawBBoxLabels({
                "src": "v_full_with_scene_md", "dst": "v_overlay_drawn",
                "group": "scene", "name": "Draw_Scene_Overlay",
                "metadata_key": OVERLAY_KEY,
                "label_template": "{label}", "min_conf": 0.0,
                "font_scale": 2, "line_spacing": 8, "glyph_preset": "10x14",
                "text_color": "white", "background_color": "black",
                "background_opacity": 0.70,
                "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT,
                "pixel_format": "cuda", "real_pixel_format": "nv12",
                "debug_log_every_n": 0,
            }),
            ForceFPS({
                "fps": OUTPUT_FPS, "group": "scene",
                "src": "v_overlay_drawn", "dst": "v_overlay_fps",
            }),
        ])
        if JANUS_OUTPUT:
            nodes.extend(_janus_output_nodes("v_overlay_fps"))
        elif OUTPUT_URL:
            nodes.extend(_file_output_nodes("v_overlay_fps"))

    for node in nodes:
        avp.addNode(node)
    return avp, detector


# ── Run loop ──────────────────────────────────────────────────────────────────

def _run_until_idle(avp, detector, started: float) -> None:
    while True:
        time.sleep(1)
        avp.heartbeat()
        if INPUT_LOOP and detector._frame_no > 0:
            continue
        if detector._frame_no > 0 and detector.idle_seconds() > IDLE_TIMEOUT:
            break
        if detector._frame_no == 0 and time.monotonic() - started > max(10.0, IDLE_TIMEOUT):
            print("No video frames decoded; check input path and codec support.")
            break


def main():
    if OUTPUT_URL and JANUS_OUTPUT:
        raise SystemExit("Set either AVP_OUTPUT for MP4 or AVP_JANUS_OUTPUT=1 for Janus, not both.")
    if WEBUI_API and not REMOTE_CONTROL_PORT:
        raise SystemExit("AVP_WEBUI_API requires AVP_REMOTE_CONTROL_PORT.")
    if DETECTOR_TYPE not in ("content", "adaptive", "histogram", "threshold"):
        raise SystemExit(f"Unknown AVP_DETECTOR_TYPE={DETECTOR_TYPE!r}. "
                         "Choose: content, adaptive, histogram, threshold")

    print(
        f"CUDA Scene Detection\n"
        f"  input     : {INPUT_URL!r}\n"
        f"  detector  : {DETECTOR_TYPE}  threshold={SCENE_THRESHOLD}"
        f"  min_len={SCENE_MIN_LEN}  confirm={CONFIRM_FRAMES}\n"
        f"  analysis  : {SCENE_WIDTH}x{SCENE_HEIGHT}"
        + (f"  luma_only=true" if LUMA_ONLY else "")
        + (f"\n  adaptive  : window={WINDOW_WIDTH}  min_content={MIN_CONTENT_VAL}"
           if DETECTOR_TYPE == "adaptive" else "")
        + (f"\n  threshold : fade_bias={FADE_BIAS}"
           if DETECTOR_TYPE == "threshold" else "")
    )
    if JANUS_OUTPUT:
        print(
            f"  Janus RTP : {JANUS_HOST}:{JANUS_VIDEO_PORT} "
            f"pt={JANUS_VIDEO_PT} ssrc=0x{JANUS_VIDEO_SSRC:x} {JANUS_VIDEO_BITRATE_KBPS}k"
        )
    if OUTPUT_URL:
        print(f"  output    : {OUTPUT_URL!r} ({OUTPUT_BITRATE})")
    if METADATA_KAFKA:
        print(f"  Kafka meta: {METADATA_KAFKA_BROKERS}  topic={METADATA_KAFKA_TOPIC!r}  keys={METADATA_KEYS!r}")
    if WEBUI_API:
        print(f"  Web UI    : {WEBUI_API!r} instance={INSTANCE_NAME!r} port={REMOTE_CONTROL_PORT}")

    avp, detector = build_graph()
    avp.group("in").startNodes()
    time.sleep(1)
    avp.group("scene").startNodes()
    if REMOTE_CONTROL_PORT:
        avp.setReady()

    rtcp_feedback_listener = None
    if JANUS_OUTPUT and JANUS_RTCP_FEEDBACK:
        def _trigger_janus_keyframe(request: str) -> None:
            try:
                avp.executeCommandsFromString("node.object.set janus_force_keyframe trigger true")
            except Exception as exc:
                print(f"[rtcp] keyframe trigger failed for {request}: {exc}")

        rtcp_feedback_listener = RtcpFeedbackListener(
            bind_host=JANUS_RTCP_FEEDBACK_BIND,
            bind_port=JANUS_RTCP_FEEDBACK_PORT,
            janus_host=JANUS_HOST,
            janus_rtcp_port=JANUS_VIDEO_PORT + 1,
            media_ssrc=JANUS_VIDEO_SSRC,
            on_keyframe_request=_trigger_janus_keyframe,
        )
        rtcp_feedback_listener.start()

    started = time.monotonic()
    try:
        _run_until_idle(avp, detector, started)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if rtcp_feedback_listener is not None:
            rtcp_feedback_listener.stop()
        avp.group("scene").stopNodes()
        avp.group("in").stopNodes()
        detector.finalize()


if __name__ == "__main__":
    main()
