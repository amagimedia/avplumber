#!/usr/bin/env python3
# Detect scene changes in an input file using PySceneDetect (https://www.scenedetect.com/).
#
# The AVPlumber graph opens the file, decodes the first video stream, and rescales
# each frame to packed BGR24 -- exactly the pixel layout PySceneDetect expects. A
# sink PythonNode wraps every frame as a numpy view and feeds it to a PySceneDetect
# ContentDetector. Every detected cut is printed to the console as it is found, and a
# final scene list is printed when the input reaches EOF.
#
# Usage (run from the repository root):
#
#     AVP_INPUT=/path/to/input.mp4 python3 pyplumber/examples/scene-detect.py
#
# Requires: pip install scenedetect numpy
#
# Tunables (environment variables):
#   AVP_INPUT             input URL or local media path (default: input.mp4)
#   AVP_SCENE_DETECTOR    detection algorithm (default: content). One of:
#                           content   - HSV content changes; general purpose
#                           adaptive  - content scores vs. a rolling average; robust to fast motion
#                           threshold - average pixel intensity; fade-in/out to black
#                           histogram - YUV luma histogram correlation
#                           hash      - perceptual hash difference
#   AVP_SCENE_THRESHOLD   detection threshold; meaning/scale depends on the detector.
#                         Unset = use that detector's own default. lower = more sensitive
#                         for content/adaptive; see PySceneDetect docs per algorithm.
#   AVP_SCENE_MIN_LEN     minimum scene length in frames (default: 15)
#   AVP_SCENE_WIDTH       maximum analysis frame width; smaller is faster (default: 640)
#   AVP_SCENE_HEIGHT      maximum analysis frame height; aspect is preserved (default: 640)
#   AVP_IDLE_TIMEOUT      seconds of no new frames that mark EOF (default: 3.0)
#   AVP_OUTPUT            optional MP4 output path with burned-in scene overlay
#   AVP_OUTPUT_WIDTH      output/full-resolution frame width (default: 608)
#   AVP_OUTPUT_HEIGHT     output/full-resolution frame height (default: 1080)
#   AVP_OUTPUT_BITRATE    NVENC target bitrate for AVP_OUTPUT (default: 6000k)
#   AVP_OUTPUT_FPS        output frame rate/timebase for encoding (default: 25/1)
#   AVP_JANUS_OUTPUT      send realtime H.264 RTP to Janus instead of MP4 (default: 0)
#   AVP_JANUS_HOST        Janus RTP ingest host (default: 127.0.0.1)
#   AVP_JANUS_VIDEO_PORT  Janus H.264 RTP port; RTCP uses port+1 (default: 5004)
#   AVP_WEBUI_API         optional Web UI API endpoint, e.g. http://127.0.0.1:22222
#   AVP_REMOTE_CONTROL_PORT optional line-based control server port for Web UI

import ctypes
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from scenedetect import FrameTimecode
    from scenedetect.detectors import ContentDetector
except Exception:
    FrameTimecode = None
    ContentDetector = None

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
    PythonNode,
    Realtime,
    SmoothTimestamps,
)
from pyplumber.auto_mixer.config import (
    JANUS_DEFAULT_HOST,
    JANUS_DEFAULT_VIDEO_BITRATE_KBPS,
    JANUS_DEFAULT_VIDEO_PORT,
    JANUS_DEFAULT_VIDEO_SSRC,
    RTP_PKT_SIZE,
)
from pyplumber.auto_mixer.rtcp_feedback import RtcpFeedbackListener


# Maps AVP_SCENE_DETECTOR values to the PySceneDetect class and the constructor
# keyword that receives AVP_SCENE_THRESHOLD (AdaptiveDetector names it differently).
_DETECTORS = {
    "content":   ("ContentDetector",   "threshold"),
    "adaptive":  ("AdaptiveDetector",  "adaptive_threshold"),
    "threshold": ("ThresholdDetector", "threshold"),
    "histogram": ("HistogramDetector", "threshold"),
    "hash":      ("HashDetector",      "threshold"),
}


def build_detector():
    """Instantiate the PySceneDetect detector selected by AVP_SCENE_DETECTOR."""
    import scenedetect.detectors as detectors

    if SCENE_DETECTOR not in _DETECTORS:
        choices = ", ".join(sorted(_DETECTORS))
        raise SystemExit(f"Unknown AVP_SCENE_DETECTOR {SCENE_DETECTOR!r}; choose one of: {choices}")

    cls_name, threshold_kw = _DETECTORS[SCENE_DETECTOR]
    cls = getattr(detectors, cls_name, None)
    if cls is None:
        raise SystemExit(
            f"{cls_name} is not available in your scenedetect version; "
            f"upgrade with 'pip install -U scenedetect' or pick another AVP_SCENE_DETECTOR."
        )

    kwargs = {"min_scene_len": SCENE_MIN_LEN}
    if SCENE_THRESHOLD is not None:
        kwargs[threshold_kw] = SCENE_THRESHOLD
    return cls(**kwargs)


INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "")
OUTPUT_WIDTH = int(os.environ.get("AVP_OUTPUT_WIDTH", "608"))
OUTPUT_HEIGHT = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
OUTPUT_BITRATE = os.environ.get("AVP_OUTPUT_BITRATE", "6000k")
OUTPUT_FPS = os.environ.get("AVP_OUTPUT_FPS", "25/1")
OVERLAY_METADATA_KEY = os.environ.get("AVP_SCENE_OVERLAY_METADATA_KEY", "scene_overlay")
WEBUI_API = os.environ.get("AVP_WEBUI_API", "")
INSTANCE_NAME = os.environ.get("AVP_INSTANCE_NAME", "scene-detect")
LOGFILE = os.environ.get("AVP_LOGFILE", "")
REMOTE_CONTROL_PORT = int(os.environ.get("AVP_REMOTE_CONTROL_PORT", "0") or "0")
SCENE_DETECTOR = os.environ.get("AVP_SCENE_DETECTOR", "content").strip().lower()
# None = let the chosen detector use its own default threshold (scales differ per algorithm).
_threshold_env = os.environ.get("AVP_SCENE_THRESHOLD")
SCENE_THRESHOLD = float(_threshold_env) if _threshold_env not in (None, "") else None
SCENE_MIN_LEN = int(os.environ.get("AVP_SCENE_MIN_LEN", "15"))
SCENE_WIDTH = int(os.environ.get("AVP_SCENE_WIDTH", "640"))
SCENE_HEIGHT = int(os.environ.get("AVP_SCENE_HEIGHT", "640"))
IDLE_TIMEOUT = float(os.environ.get("AVP_IDLE_TIMEOUT", "3.0"))
FALLBACK_FPS = float(os.environ.get("AVP_SCENE_FPS", "0")) or None

NOPTS = -9223372036854775808  # AV_NOPTS_VALUE


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_auto_base(value: str) -> int:
    return int(str(value), 0)


JANUS_OUTPUT = _env_bool("AVP_JANUS_OUTPUT")
JANUS_HOST = os.environ.get("AVP_JANUS_HOST", JANUS_DEFAULT_HOST)
JANUS_VIDEO_PORT = int(os.environ.get("AVP_JANUS_VIDEO_PORT", str(JANUS_DEFAULT_VIDEO_PORT)))
JANUS_VIDEO_PT = int(os.environ.get("AVP_JANUS_VIDEO_PT", "96"))
JANUS_VIDEO_SSRC = _parse_int_auto_base(os.environ.get("AVP_JANUS_VIDEO_SSRC", f"0x{JANUS_DEFAULT_VIDEO_SSRC:x}"))
JANUS_VIDEO_BITRATE_KBPS = int(os.environ.get("AVP_JANUS_VIDEO_BITRATE_KBPS", str(JANUS_DEFAULT_VIDEO_BITRATE_KBPS)))
JANUS_RTCP_FEEDBACK = not _env_bool("AVP_DISABLE_JANUS_RTCP_FEEDBACK")
JANUS_RTCP_FEEDBACK_BIND = os.environ.get("AVP_JANUS_RTCP_FEEDBACK_BIND", "0.0.0.0")
JANUS_RTCP_FEEDBACK_PORT = int(os.environ.get("AVP_JANUS_RTCP_FEEDBACK_PORT", "0") or "0")
INPUT_LOOP = _env_bool("AVP_INPUT_LOOP", JANUS_OUTPUT)


def _fmt_ts(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "??:??:??.???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_overlay_ts(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "NONE"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _frame_seconds(frame) -> float:
    pts = frame.pts
    ts = int(pts.timestamp)
    if ts == NOPTS:
        return float("nan")
    tb = pts.timebase
    if not tb or int(tb.den) == 0:
        return float(ts)
    return float(ts) * float(tb.num) / float(tb.den)


def _scene_overlay_payload(count: int, last_seconds: float | None) -> dict:
    labels = [
        ("SCENES", f"{count:03d}", 40),
        ("LAST", _fmt_overlay_ts(last_seconds), 82),
    ]
    return {
        "coord_space": "frame",
        "detections": [
            {
                "label": f"{prefix} {value}",
                "conf": 1.0,
                "xyxy": [24, y1, 300, y1 + 24],
            }
            for prefix, value, y1 in labels
        ],
    }


def _cuda_decoder(src: str, dst: str, name: str) -> DecVideo:
    return DecVideo({
        "src": src,
        "dst": dst,
        "group": "scene",
        "name": name,
        "auto_restart": "off",
        "optional": True,
        "pixel_format": "?cuda",
        "hwaccel": "@gpu",
        "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
        "hwaccel_only_for_codecs": ["h264", "hevc"],
    })


def _analysis_scale_filter(render_output: bool, src: str | None = None) -> FilterVideo:
    if render_output:
        graph = (
            f"scale_cuda=w={SCENE_WIDTH}:h={SCENE_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2,"
            "hwdownload,format=nv12,format=bgr24"
        )
        src = src or "v_dec_analysis_cuda"
        name = "Cuda_Scale_For_Scene_Detect"
        hwaccel = "@gpu"
    else:
        graph = (
            f"scale=w={SCENE_WIDTH}:h={SCENE_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2,"
            "format=bgr24"
        )
        src = src or "v_dec"
        name = "Scale_For_Scene_Detect"
        hwaccel = None

    params = {
        "src": src,
        "dst": "v_bgr",
        "group": "scene",
        "name": name,
        "graph": graph,
        "dst_width": SCENE_WIDTH,
        "dst_height": SCENE_HEIGHT,
        "dst_pixel_format": "bgr24",
        "auto_restart": "off",
    }
    if hwaccel:
        params["hwaccel"] = hwaccel
    return FilterVideo(params)


def _rtp_url(host: str, port: int, *, rtcp_port: int | None = None) -> str:
    query = f"pkt_size={RTP_PKT_SIZE}"
    if rtcp_port is not None:
        query += f"&rtcp_port={rtcp_port}"
    return f"rtp://{host}:{port}?{query}"


def _packet_realtime_node(src: str, dst: str) -> Realtime:
    return Realtime({
        "team": "scene-demux-rt",
        "negative_time_tolerance": 0.02,
        "negative_time_discard": 0.02,
        "jitter_margin": 0.1,
        "initial_jitter_margin": 0.1,
        "discontinuity_threshold": 3,
        "src": src,
        "dst": dst,
        "group": "in",
        "name": "Demux_Realtime",
        "auto_restart": "on",
    })


def _file_timestamp_node(src: str, dst: str) -> SmoothTimestamps:
    return SmoothTimestamps({
        "src": src,
        "dst": dst,
        "fps": OUTPUT_FPS,
        "drift_window": 0,
        "discontinuity_threshold": 86400,
        "group": "scene",
        "name": "File_Timestamps",
        "auto_restart": "on",
    })


def _file_output_nodes(src_edge: str) -> list:
    return [
        ForceKeyFrame({
            "interval_sec": "2/1",
            "group": "scene",
            "src": src_edge,
            "dst": "v_keyframed",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": "scene",
            "src": "v_keyframed",
            "dst": "v_preenc",
            "auto_restart": "off",
        }),
        EncVideo({
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": "scene",
            "name": "Encode_NVENC",
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "options": {
                "b": OUTPUT_BITRATE,
                "maxrate": OUTPUT_BITRATE,
                "bufsize": OUTPUT_BITRATE,
                "rc": "cbr",
                "g": 50,
                "bf": 0,
                "preset": "p1",
                "profile": "high",
                "level": "4.0",
                "forced-idr": 1,
                "no-scenecut": 1,
                "strict_gop": 1,
                "aud": 1,
                "spatial-aq": 0,
                "temporal-aq": 0,
            },
        }),
        Mux({
            "src": ["v_outenc"],
            "dst": "mux_v",
            "group": "scene",
            "ts_sort_wait": 0,
        }),
        Output({
            "format": "mp4",
            "url": OUTPUT_URL,
            "src": "mux_v",
            "group": "scene",
            "name": "Output_MP4",
            "auto_restart": "off",
        }),
    ]


def _janus_output_nodes(src_edge: str) -> list:
    video_bitrate = f"{JANUS_VIDEO_BITRATE_KBPS}k"
    return [
        ForceKeyFrame({
            "name": "janus_force_keyframe",
            "interval_sec": "1/1",
            "group": "scene",
            "src": src_edge,
            "dst": "janus_keyframed",
            "auto_restart": "panic",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": "scene",
            "src": "janus_keyframed",
            "dst": "janus_preenc",
            "auto_restart": "panic",
        }),
        EncVideo({
            "src": "janus_preenc",
            "dst": "janus_v_enc",
            "group": "scene",
            "name": "Janus_Encode_NVENC",
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "on_error": "panic",
            "options": {
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
        }),
        Bsf({
            "name": "janus_repeat_headers",
            "src": "janus_v_enc",
            "dst": "janus_v_repeat_headers",
            "bsf": "dump_extra=freq=keyframe",
            "group": "scene",
            "auto_restart": "panic",
        }),
        Mux({
            "src": ["janus_v_repeat_headers"],
            "dst": "janus_video_rtp_mux",
            "group": "scene",
            "ts_sort_wait": 0,
            "auto_restart": "on",
            "on_error": "panic",
        }),
        Output({
            "format": "rtp",
            "url": _rtp_url(JANUS_HOST, JANUS_VIDEO_PORT, rtcp_port=JANUS_VIDEO_PORT + 1),
            "src": "janus_video_rtp_mux",
            "group": "scene",
            "name": "Janus_RTP_Output",
            "auto_restart": "on",
            "on_error": "panic",
            "options": {
                "payload_type": JANUS_VIDEO_PT,
                "rtpflags": "skip_rtcp",
                "ssrc": JANUS_VIDEO_SSRC,
            },
        }),
    ]


class SceneDetectNode(PythonNode):
    """Sink node: feeds each BGR24 frame to a PySceneDetect ContentDetector."""

    def __init__(self, args):
        super().__init__(args)
        self._frame_no = 0
        self._last_activity = time.monotonic()
        self._cuts = []  # list of (frame_no, seconds)
        self._process_time_total = 0.0  # cumulative seconds spent in process_frame
        self._fps = FALLBACK_FPS
        self._pending = []  # (seconds, img) buffered until fps is known
        if ContentDetector is None:
            self._detector = None
        else:
            self._detector = build_detector()

    def _write_overlay_metadata(self, frame) -> None:
        if not OVERLAY_METADATA_KEY:
            return
        last_seconds = self._cuts[-1][1] if self._cuts else None
        frame.metadata[OVERLAY_METADATA_KEY] = json.dumps(_scene_overlay_payload(len(self._cuts), last_seconds))

    def _forward_if_configured(self, frame) -> None:
        dst = getattr(self, "_dst", None)
        if isinstance(dst, dict):
            for edge in dst.values():
                edge.enqueue(frame)
        elif dst is not None:
            dst.enqueue(frame)

    def _frame_to_bgr(self, frame):
        height = int(frame.height)
        width = int(frame.width)
        stride = int(frame.linesize[0]) if frame.linesize else 0
        ptr = int(frame.data_ptr[0]) if frame.data_ptr else 0
        if ptr <= 0 or stride <= 0 or width <= 0 or height <= 0:
            return None
        size = stride * height
        c_buf = (ctypes.c_uint8 * size).from_address(ptr)
        flat = np.frombuffer(c_buf, dtype=np.uint8, count=size).reshape(height, stride)
        return np.ascontiguousarray(flat[:, : width * 3].reshape(height, width, 3))

    def _record_cut(self, cut):
        # PySceneDetect 0.7 returns FrameTimecode objects for each cut.
        seconds = float(cut.seconds)
        frame = int(cut.frame_num)
        self._cuts.append((frame, seconds))
        print(f"  scene change #{len(self._cuts):<3} frame {frame:<8} at {_fmt_ts(seconds)}")
        sys.stdout.flush()

    def _feed(self, img):
        t0 = time.perf_counter()
        cuts = self._detector.process_frame(FrameTimecode(self._frame_no, self._fps), img)
        self._process_time_total += time.perf_counter() - t0
        for cut in cuts:
            self._record_cut(cut)
        self._frame_no += 1

    def process(self):
        p = self._src.tryGet(1000)
        if p is None:
            return  # timeout -- no frame ready; main loop watches for EOF
        self._last_activity = time.monotonic()

        if self._detector is None:
            self._forward_if_configured(p)
            return  # PySceneDetect missing; main loop reports the install hint

        img = self._frame_to_bgr(p)
        if img is None:
            self._forward_if_configured(p)
            return

        if self._fps is None:
            # ContentDetector needs an fps to build FrameTimecodes. Estimate it from
            # the gap between the first two frames' PTS, then flush the buffer.
            self._pending.append((_frame_seconds(p), img))
            if len(self._pending) < 2:
                return
            dt = self._pending[1][0] - self._pending[0][0]
            self._fps = (1.0 / dt) if (dt and dt == dt and dt > 0) else 25.0
            buffered, self._pending = self._pending, []
            for _, buf_img in buffered:
                self._feed(buf_img)
            self._write_overlay_metadata(p)
            self._forward_if_configured(p)
            return

        self._feed(img)
        self._write_overlay_metadata(p)
        self._forward_if_configured(p)

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def finalize(self):
        if self._detector is not None:
            # Flush frames still buffered for the fps estimate (very short inputs).
            if self._pending:
                if self._fps is None:
                    self._fps = 25.0
                buffered, self._pending = self._pending, []
                for _, buf_img in buffered:
                    self._feed(buf_img)
            for cut in self._detector.post_process(FrameTimecode(self._frame_no, self._fps or 25.0)):
                self._record_cut(cut)

        print("\n=== Scene detection summary ===")
        print(f"frames analyzed : {self._frame_no}")
        print(f"scene changes   : {len(self._cuts)}")
        print(f"scenes          : {len(self._cuts) + 1}")
        if self._frame_no > 0:
            avg_ms = (self._process_time_total / self._frame_no) * 1000.0
            print(f"avg frame time  : {avg_ms:.3f} ms")

        if not self._cuts:
            return
        print("\nscene boundaries:")
        start = 0.0
        for idx, (cut_frame, seconds) in enumerate(self._cuts, start=1):
            print(f"  scene {idx:<3} {_fmt_ts(start)} -> {_fmt_ts(seconds)} (cut at frame {cut_frame})")
            start = seconds
        print(f"  scene {len(self._cuts) + 1:<3} {_fmt_ts(start)} -> EOF")


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
    if render_output:
        avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')

    if render_output:
        split_src = "v_file_ts" if JANUS_OUTPUT else "v_dec_cuda"
        input_dst = "in_mux_raw" if JANUS_OUTPUT else "in_mux"
        nodes = [
            InputRec({
                "url": INPUT_URL,
                "dst": input_dst,
                "group": "in",
                "name": "input",
                "timeout": -1,
                "preseek": 0,
                "loop": INPUT_LOOP,
                "auto_restart": "off",
            }),
        ]
        if JANUS_OUTPUT:
            nodes.append(_packet_realtime_node(input_dst, "in_mux"))
        nodes.extend([
            Demux({
                "src": "in_mux",
                "routing": {"v:0": "v_pkt"},
                "group": "in",
                "auto_restart": "off",
            }),
            _cuda_decoder("v_pkt", "v_dec_cuda", "Decode_CUDA"),
        ])
        if JANUS_OUTPUT:
            nodes.append(_file_timestamp_node("v_dec_cuda", split_src))
        nodes.extend([
            OneToMany({
                "src": split_src,
                "dst": ["v_full_cuda", "v_analysis_cuda"],
                "outputs": 3,
                "group": "scene",
                "name": "Split_Full_And_Scene_Detect",
                "auto_restart": "off",
            }),
            _analysis_scale_filter(True, src="v_analysis_cuda"),
        ])
    else:
        nodes = [
            InputRec({
                "url": INPUT_URL,
                "dst": "in_mux",
                "group": "in",
                "name": "input",
                "timeout": -1,
                "preseek": 0,
                "loop": INPUT_LOOP,
                "auto_restart": "off",
            }),
            Demux({
                "src": "in_mux",
                "routing": {"v:0": "v_pkt"},
                "group": "in",
                "auto_restart": "off",
            }),
            DecVideo({
                "src": "v_pkt",
                "dst": "v_dec",
                "group": "scene",
                "name": "Video_Decode",
                "auto_restart": "off",
            }),
            _analysis_scale_filter(False),
        ]

    detector = SceneDetectNode({
        "src": "v_bgr",
        **({"dst": "v_scene_overlay_md"} if render_output else {}),
        "group": "scene",
        "name": "scene-detect",
    })
    nodes.append(detector)

    if render_output:
        nodes.extend([
            JoinMetadata({
                "src": ["v_full_cuda", "v_scene_overlay_md"],
                "dst": "v_full_with_scene_md",
                "group": "scene",
                "auto_restart": "off",
            }),
            DrawBBoxLabels({
                "src": "v_full_with_scene_md",
                "dst": "v_overlay_drawn",
                "group": "scene",
                "name": "Draw_Scene_Overlay",
                "metadata_key": OVERLAY_METADATA_KEY,
                "label_template": "{label}",
                "min_conf": 0.0,
                "font_scale": 2,
                "line_spacing": 8,
                "glyph_preset": "10x14",
                "text_color": "white",
                "background_color": "black",
                "background_opacity": 0.70,
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "pixel_format": "cuda",
                "real_pixel_format": "nv12",
                "debug_log_every_n": 0,
            }),
            ForceFPS({
                "fps": OUTPUT_FPS,
                "group": "scene",
                "src": "v_overlay_drawn",
                "dst": "v_overlay_fps",
            }),
        ])
        if JANUS_OUTPUT:
            nodes.extend(_janus_output_nodes("v_overlay_fps"))
        elif OUTPUT_URL:
            nodes.extend(_file_output_nodes("v_overlay_fps"))

    for node in nodes:
        avp.addNode(node)
    return avp, detector


def _run_until_idle(avp, detector, started: float) -> None:
    while True:
        time.sleep(1)
        avp.heartbeat()
        if INPUT_LOOP and detector._frame_no > 0:
            continue
        # EOF: frames were processed, then none arrived for IDLE_TIMEOUT seconds.
        if detector._frame_no > 0 and detector.idle_seconds() > IDLE_TIMEOUT:
            break
        # Guard against a file that never produced a frame (bad path/codec).
        if detector._frame_no == 0 and time.monotonic() - started > max(10.0, IDLE_TIMEOUT):
            print("No video frames were decoded; check the input path and codec support.")
            break


def main():
    if ContentDetector is None:
        print("PySceneDetect is not installed. Install it with:\n    pip install scenedetect")
        sys.exit(1)
    if OUTPUT_URL and JANUS_OUTPUT:
        raise SystemExit("Set either AVP_OUTPUT for MP4 or AVP_JANUS_OUTPUT=1 for Janus RTP, not both.")
    if WEBUI_API and not REMOTE_CONTROL_PORT:
        raise SystemExit("AVP_WEBUI_API requires AVP_REMOTE_CONTROL_PORT so Web UI can inspect/control the graph.")

    threshold_disp = SCENE_THRESHOLD if SCENE_THRESHOLD is not None else "default"
    print(
        f"Analyzing {INPUT_URL!r} (detector={SCENE_DETECTOR}, "
        f"threshold={threshold_disp}, min_scene_len={SCENE_MIN_LEN})"
    )
    if OUTPUT_URL:
        print(f"Rendering overlay MP4 to {OUTPUT_URL!r} ({OUTPUT_BITRATE})")
    if JANUS_OUTPUT:
        print(
            "Streaming realtime Janus RTP to "
            f"{JANUS_HOST}:{JANUS_VIDEO_PORT} pt={JANUS_VIDEO_PT} "
            f"ssrc=0x{JANUS_VIDEO_SSRC:x} bitrate={JANUS_VIDEO_BITRATE_KBPS}k"
        )
    if WEBUI_API:
        print(
            f"Registering Web UI instance {INSTANCE_NAME!r} at {WEBUI_API!r} "
            f"(control port {REMOTE_CONTROL_PORT})"
        )
    avp, detector = build_graph()
    avp.group("in").startNodes()
    # Let the demux read enough of the container to expose the video stream
    # parameters before the decoder group starts, avoiding a noisy init retry.
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
                print(f"[rtcp] failed to trigger Janus keyframe for {request}: {exc}")

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
