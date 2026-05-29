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
#   AVP_SCENE_WIDTH       analysis frame width; smaller is faster (default: 640)
#   AVP_SCENE_HEIGHT      analysis frame height (default: 360)
#   AVP_IDLE_TIMEOUT      seconds of no new frames that mark EOF (default: 3.0)

import ctypes
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
from pyplumber.node import DecVideo, Demux, InputRec, PythonNode, RescaleVideo


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
SCENE_DETECTOR = os.environ.get("AVP_SCENE_DETECTOR", "content").strip().lower()
# None = let the chosen detector use its own default threshold (scales differ per algorithm).
_threshold_env = os.environ.get("AVP_SCENE_THRESHOLD")
SCENE_THRESHOLD = float(_threshold_env) if _threshold_env not in (None, "") else None
SCENE_MIN_LEN = int(os.environ.get("AVP_SCENE_MIN_LEN", "15"))
SCENE_WIDTH = int(os.environ.get("AVP_SCENE_WIDTH", "1920"))
SCENE_HEIGHT = int(os.environ.get("AVP_SCENE_HEIGHT", "1080"))
IDLE_TIMEOUT = float(os.environ.get("AVP_IDLE_TIMEOUT", "3.0"))
FALLBACK_FPS = float(os.environ.get("AVP_SCENE_FPS", "0")) or None

NOPTS = -9223372036854775808  # AV_NOPTS_VALUE


def _fmt_ts(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "??:??:??.???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _frame_seconds(frame) -> float:
    pts = frame.pts
    ts = int(pts.timestamp)
    if ts == NOPTS:
        return float("nan")
    tb = pts.timebase
    if not tb or int(tb.den) == 0:
        return float(ts)
    return float(ts) * float(tb.num) / float(tb.den)


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
            return  # PySceneDetect missing; main loop reports the install hint

        img = self._frame_to_bgr(p)
        if img is None:
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
            return

        self._feed(img)

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
    avp.edges.planCapacity("*", 15)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux",
            "group": "in",
            "name": "input",
            "timeout": -1,
            "preseek": 0,
            "loop": False,
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
        RescaleVideo({
            "src": "v_dec",
            "dst": "v_bgr",
            "group": "scene",
            "name": "Scale_For_Scene_Detect",
            "dst_width": SCENE_WIDTH,
            "dst_height": SCENE_HEIGHT,
            "dst_pixel_format": "bgr24",
            "flags": ["SWS_BILINEAR"],
            "auto_restart": "off",
        }),
    ]

    detector = SceneDetectNode({
        "src": "v_bgr",
        "group": "scene",
        "name": "scene-detect",
    })
    nodes.append(detector)

    for node in nodes:
        avp.addNode(node)
    return avp, detector


def main():
    if ContentDetector is None:
        print("PySceneDetect is not installed. Install it with:\n    pip install scenedetect")
        sys.exit(1)

    threshold_disp = SCENE_THRESHOLD if SCENE_THRESHOLD is not None else "default"
    print(
        f"Analyzing {INPUT_URL!r} (detector={SCENE_DETECTOR}, "
        f"threshold={threshold_disp}, min_scene_len={SCENE_MIN_LEN})"
    )
    avp, detector = build_graph()
    avp.group("in").startNodes()
    # Let the demux read enough of the container to expose the video stream
    # parameters before the decoder group starts, avoiding a noisy init retry.
    time.sleep(1)
    avp.group("scene").startNodes()

    started = time.monotonic()
    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
            # EOF: frames were processed, then none arrived for IDLE_TIMEOUT seconds.
            if detector._frame_no > 0 and detector.idle_seconds() > IDLE_TIMEOUT:
                break
            # Guard against a file that never produced a frame (bad path/codec).
            if detector._frame_no == 0 and time.monotonic() - started > max(10.0, IDLE_TIMEOUT):
                print("No video frames were decoded; check the input path and codec support.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        avp.group("scene").stopNodes()
        avp.group("in").stopNodes()
        detector.finalize()


if __name__ == "__main__":
    main()
