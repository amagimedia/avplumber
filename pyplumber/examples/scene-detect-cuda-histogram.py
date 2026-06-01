#!/usr/bin/env python3
# CUDA-accelerated histogram-based scene detection.
# Works on CUDA NV12 frames directly — no hwdownload/CPU round-trip.

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
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False

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


INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "")
OUTPUT_WIDTH = int(os.environ.get("AVP_OUTPUT_WIDTH", "608"))
OUTPUT_HEIGHT = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
OUTPUT_BITRATE = os.environ.get("AVP_OUTPUT_BITRATE", "6000k")
OUTPUT_FPS = os.environ.get("AVP_OUTPUT_FPS", "25/1")
OVERLAY_METADATA_KEY = os.environ.get("AVP_SCENE_OVERLAY_METADATA_KEY", "scene_overlay")
WEBUI_API = os.environ.get("AVP_WEBUI_API", "")
INSTANCE_NAME = os.environ.get("AVP_INSTANCE_NAME", "scene-detect-cuda")
LOGFILE = os.environ.get("AVP_LOGFILE", "")
REMOTE_CONTROL_PORT = int(os.environ.get("AVP_REMOTE_CONTROL_PORT", "0") or "0")
SCENE_THRESHOLD = float(os.environ.get("AVP_SCENE_THRESHOLD", "0.05"))
SCENE_MIN_LEN = int(os.environ.get("AVP_SCENE_MIN_LEN", "15"))
SCENE_WIDTH = int(os.environ.get("AVP_SCENE_WIDTH", "640"))
SCENE_HEIGHT = int(os.environ.get("AVP_SCENE_HEIGHT", "360"))
IDLE_TIMEOUT = float(os.environ.get("AVP_IDLE_TIMEOUT", "3.0"))
FALLBACK_FPS = float(os.environ.get("AVP_SCENE_FPS", "0")) or None
HIST_BINS = int(os.environ.get("AVP_HIST_BINS", "256"))

NOPTS = -9223372036854775808


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
    if seconds != seconds:
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


def _analysis_scale_filter_cuda(src: str = "v_dec_analysis_cuda") -> FilterVideo:
    """CUDA scale filter that outputs NV12 in CUDA memory — no hwdownload."""
    graph = (
        f"scale_cuda=w={SCENE_WIDTH}:h={SCENE_HEIGHT}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    return FilterVideo({
        "src": src,
        "dst": "v_nv12_cuda",
        "group": "scene",
        "name": "Cuda_Scale_For_Scene_Detect",
        "graph": graph,
        "dst_width": SCENE_WIDTH,
        "dst_height": SCENE_HEIGHT,
        "dst_pixel_format": "cuda",
        "hwaccel": "@gpu",
        "auto_restart": "off",
    })


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


class CudaHistogramSceneDetector(PythonNode):
    """CUDA-accelerated histogram-based scene detector.
    
    Operates directly on NV12 CUDA frames. Computes luma histogram
    correlation between consecutive frames using cupy.
    """

    def __init__(self, args):
        super().__init__(args)
        self._frame_no = 0
        self._last_activity = time.monotonic()
        self._cuts = []
        self._process_time_total = 0.0
        self._fps = FALLBACK_FPS
        self._last_hist = None
        self._frames_since_cut = 0
        
        if not HAVE_CUPY:
            raise RuntimeError("cupy is required for CUDA histogram detector. Install with: pip install cupy-cuda12x")

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

    def _nv12_cuda_to_luma_cupy(self, frame):
        """Extract Y plane from NV12 CUDA frame as cupy array."""
        height = int(frame.height)
        width = int(frame.width)
        stride = int(frame.linesize[0]) if frame.linesize else 0
        ptr = int(frame.data_ptr[0]) if frame.data_ptr else 0
        
        if ptr <= 0 or stride <= 0 or width <= 0 or height <= 0:
            return None
        
        # NV12: Y plane is height rows of stride bytes each
        y_size = stride * height
        
        # Wrap CUDA pointer as cupy array
        y_plane_flat = cp.ndarray(
            shape=(y_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, y_size, owner=frame), 0)
        )
        
        # Reshape to 2D and extract valid width
        y_plane_strided = y_plane_flat.reshape(height, stride)
        y_plane = y_plane_strided[:, :width]
        
        return y_plane

    def _compute_histogram_cupy(self, y_plane):
        """Compute normalized histogram on GPU."""
        hist = cp.histogram(y_plane, bins=HIST_BINS, range=(0, 256))[0]
        hist_norm = hist.astype(cp.float32) / hist.sum()
        return hist_norm

    def _histogram_correlation(self, hist1, hist2):
        """Compute correlation between two histograms."""
        mean1 = hist1.mean()
        mean2 = hist2.mean()
        
        numerator = ((hist1 - mean1) * (hist2 - mean2)).sum()
        denom1 = cp.sqrt(((hist1 - mean1) ** 2).sum())
        denom2 = cp.sqrt(((hist2 - mean2) ** 2).sum())
        
        if denom1 == 0 or denom2 == 0:
            return 1.0
        
        corr = numerator / (denom1 * denom2)
        return float(corr)

    def _record_cut(self, frame_no: int, seconds: float):
        self._cuts.append((frame_no, seconds))
        print(f"  scene change #{len(self._cuts):<3} frame {frame_no:<8} at {_fmt_ts(seconds)} (corr dropped below {SCENE_THRESHOLD})")
        sys.stdout.flush()

    def _detect(self, frame):
        t0 = time.perf_counter()
        
        y_plane = self._nv12_cuda_to_luma_cupy(frame)
        if y_plane is None:
            self._process_time_total += time.perf_counter() - t0
            return
        
        hist = self._compute_histogram_cupy(y_plane)
        
        if self._last_hist is not None:
            corr = self._histogram_correlation(self._last_hist, hist)
            
            # Low correlation = scene change
            if corr < SCENE_THRESHOLD and self._frames_since_cut >= SCENE_MIN_LEN:
                seconds = _frame_seconds(frame)
                self._record_cut(self._frame_no, seconds)
                self._frames_since_cut = 0
            else:
                self._frames_since_cut += 1
        
        self._last_hist = hist
        self._frame_no += 1
        self._process_time_total += time.perf_counter() - t0

    def process(self):
        p = self._src.tryGet(1000)
        if p is None:
            return
        
        self._last_activity = time.monotonic()
        self._detect(p)
        self._write_overlay_metadata(p)
        self._forward_if_configured(p)

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def finalize(self):
        print("\n=== CUDA Histogram Scene Detection Summary ===")
        print(f"frames analyzed : {self._frame_no}")
        print(f"scene changes   : {len(self._cuts)}")
        print(f"scenes          : {len(self._cuts) + 1}")
        print(f"threshold       : {SCENE_THRESHOLD}")
        print(f"min scene len   : {SCENE_MIN_LEN} frames")
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
        _analysis_scale_filter_cuda(src="v_analysis_cuda"),
    ])

    detector = CudaHistogramSceneDetector({
        "src": "v_nv12_cuda",
        **({"dst": "v_scene_overlay_md"} if render_output else {}),
        "group": "scene",
        "name": "cuda-histogram-detect",
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

    for node in nodes:
        avp.addNode(node)
    return avp, detector


def _run_until_idle(avp, detector, started: float) -> None:
    while True:
        time.sleep(1)
        avp.heartbeat()
        if INPUT_LOOP and detector._frame_no > 0:
            continue
        if detector._frame_no > 0 and detector.idle_seconds() > IDLE_TIMEOUT:
            break
        if detector._frame_no == 0 and time.monotonic() - started > max(10.0, IDLE_TIMEOUT):
            print("No video frames were decoded; check the input path and codec support.")
            break


def main():
    if not HAVE_CUPY:
        print("cupy is not installed. Install it with:")
        print("    pip install cupy-cuda12x")
        sys.exit(1)
    
    if WEBUI_API and not REMOTE_CONTROL_PORT:
        raise SystemExit("AVP_WEBUI_API requires AVP_REMOTE_CONTROL_PORT")

    print(
        f"CUDA Histogram Scene Detection on {INPUT_URL!r}\n"
        f"  threshold={SCENE_THRESHOLD}, min_scene_len={SCENE_MIN_LEN}, bins={HIST_BINS}\n"
        f"  analysis size={SCENE_WIDTH}x{SCENE_HEIGHT}"
    )
    if JANUS_OUTPUT:
        print(
            f"  Janus RTP: {JANUS_HOST}:{JANUS_VIDEO_PORT} "
            f"pt={JANUS_VIDEO_PT} ssrc=0x{JANUS_VIDEO_SSRC:x} {JANUS_VIDEO_BITRATE_KBPS}k"
        )
    if WEBUI_API:
        print(f"  Web UI: {WEBUI_API!r} instance={INSTANCE_NAME!r} port={REMOTE_CONTROL_PORT}")
    
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
