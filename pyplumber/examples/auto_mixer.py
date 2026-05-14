"""Automatic scene switcher with face reframer and multi-modal speech detection.

Each input fans out to:
  - an original 16:9 leg (for PiP / multiviewer / vstack layouts)
  - a face-tracked 9:16 leg (for fullscreen and videoconference layouts)

Speech detection uses two complementary signals:
  - Audio: Silero neural VAD (far more accurate than RMS energy thresholds).
    Requires audio resampled to 16 kHz / mono.  Events are bridged into the
    Speaker registry via SileroVadRegistryBridge.
  - Visual: lip-motion analysis via FaceAnchoredMouthTrackerNode +
    VisualSpeechGateNode.  Taps the raw YOLO output (before PlayerTracker) so
    that Mouth / Nose bounding boxes are available even though only the "face"
    label is used for viewport tracking.  Updates Speaker.visual_speaking via
    VisualSpeechRegistryNode.
The AutoSwitcher triggers a scene change only when both signals are active.

All inputs are mixed on a 1080x1920 (9:16) canvas.

Available scene types
---------------------
  full_face_{i}       Dominant-speaker 9:16 reframed portrait, full canvas.
  videoconf_{i}       Dominant speaker 1:1 (static crop of 9:16) in the top
                      1080×1080 slot, up to 5 other cameras as portrait
                      thumbnails below — videoconference-style layout.
  vstack3_{a}_{b}_{c} Three 16:9 sources stacked vertically (3×1080×608).
  vstack_{a}_{b}      Two 16:9 sources stacked vertically (2×1080×608).
  pip_{i}_{j}         Camera i fullscreen + camera j thumbnail (top-right).
  multiviewer         2×2 grid of face portraits (≥ 3 inputs, up to 4).

Usage
-----
    python auto_mixer.py \\
        --inputs rtmp://host/a rtmp://host/b \\
        --output rtmp://host/out \\
        --face-engine /opt/tly/engines/yolo_face.plan \\
        [--codec h264_nvenc] \\
        [--input-start-ts 00:10] \\
        [--silero-model /opt/tly/models/silero_vad.jit] \\
        [--silero-device cpu] \\
        [--remote-control-port 7777] \\
        [--logfile /tmp/auto_mixer.log] \\
        [--webui-api http://localhost:22222] \\
        [--instance-name auto-mixer]

Environment variables
---------------------
    AVP_FACE_ENGINE      default face TRT engine path (overridden by --face-engine)
    AVP_SILERO_MODEL     Silero VAD .jit model path (optional; downloads from hub if unset)
    AVP_SILERO_REPO      torch.hub repo for Silero (default: snakers4/silero-vad)
    AVPLUMBER_UI_HEARTBEAT_INTERVAL
                         Web UI heartbeat interval in seconds
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, ".")

from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeAudioFormat,
    AssumeVideoFormat,
    CropMetadataCuda,
    CudaInferYolo,
    DecAudio,
    DecVideo,
    Demux,
    EncAudio,
    EncVideo,
    FilterVideo,
    ForceFPS,
    InputRec,
    JoinMetadata,
    Mux,
    NullSink,
    Output,
    PlayerTracker,
    Realtime,
    ResampleAudio,
    SmoothCropViewport,
    SmoothTimestamps,
    Split,
)
from pyplumber.mixer import MixerGraphBuilder
from pyplumber.audio_vad import SileroVadRegistryBridge, Speaker, VisualSpeechRegistryNode
from pyplumber.auto_switcher import AutoSwitcher
from pyplumber.mouth_tracker import FaceAnchoredMouthTrackerNode
from pyplumber.vad import SileroVADNode
from pyplumber.visual_speech import VisualSpeechGateNode

# ------------------------------------------------------------------
# Canvas and pipeline constants
# ------------------------------------------------------------------

CANVAS_W = 1080          # portrait 9:16
CANVAS_H = 1920
FPS_NUM = 30
FPS_DEN = 1
HWACCEL = "@gpu"

# Face detection model input size.
FACE_MODEL_W = 960
FACE_MODEL_H = 544       # 540 content + 2 px padding top/bottom
FACE_MODEL_CONTENT_H = 540

# Face tracking metadata key names (must match smooth_crop_viewport / crop_metadata_cuda).
FACE_METADATA_KEY = "yolo_faces"
VIEWPORT_METADATA_KEY = "smoothed_crop_viewport_v1"

# 9:16 portrait crop from a 1920x1080 frame (608 is the closest even number to
# 1080 * 9/16 = 607.5, giving a <0.1 % aspect-ratio rounding error when scaled).
FACE_CROP_W = 608
FACE_CROP_H = 1080

# YOLO face-part class labels used by the face-recognition-1.2 model.
FACE_CLASS_NAMES = ["Eye", "Face", "MakeUp", "Mouth", "Nose", "Tooth", "Topping"]
FACE_TRACKED_LABELS = ["Face"]

AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNEL_LAYOUT = "stereo"
AUDIO_SAMPLE_FORMAT = "fltp"

# Silero VAD requires 16 kHz mono float audio.
VAD_SAMPLE_RATE = 16000
MIN_ACTIVE_AUDIO_LEVEL_DBFS = -60.0
RENE_INPUT_INDEX = 4
RENE_REQUIRED_LEAD_DB = 3.0

SERGIO_INPUT_NAME = "sergio"
RENE_INPUT_NAME = "rene"

def input_basename(url: str) -> str:
    """Return a lowercase input basename for both URL and local path inputs."""
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme else url
    return Path(path).name.lower()
def find_named_input(inputs: list[str], name: str) -> int | None:
    name = name.lower()
    for i, url in enumerate(inputs):
        if name in input_basename(url):
            return i
    return None

# Visual speech metadata key names (per-input, so no collisions on the same frame).
VS_MOUTH_KEY_PREFIX = "vs_mouth_rois"
VS_VISUAL_KEY_PREFIX = "vs_visual_speech"


def default_face_engine() -> str:
    """Resolve the face TRT engine used by local dev images and remote test hosts."""
    env_engine = os.environ.get("AVP_FACE_ENGINE")
    if env_engine:
        return env_engine

    model_dir_env = os.environ.get("AVP_MODEL_DIR")
    candidates = []
    if model_dir_env:
        model_dir = Path(model_dir_env)
        candidates.extend([
            model_dir / "face-recognition_960x544.plan",
            model_dir / "face-recognition_960x544.engine",
            model_dir / "best.plan",
            model_dir / "best.engine",
        ])
    candidates.extend([
        Path("/opt/tly/engines/yolo_face.plan"),
        Path("/home/fedora/models/face-recognition-1.2/face-recognition_960x544.plan"),
        Path("/home/user/tensorrt/face-recognition-1.2/face-recognition_960x544.plan"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])

# ------------------------------------------------------------------
# Per-input subgraph
# ------------------------------------------------------------------

def build_input_subgraph(
    avp: AVPlumber,
    idx: int,
    url: str,
    face_engine: str,
    input_start_ts: str | None = None,
    sync_team: str = "",
    silero_model: str | None = None,
    silero_repo: str = "snakers4/silero-vad",
    silero_device: str = "cpu",
    silero_threshold: float = 0.5,
) -> dict:
    """Build decode + face-detection + audio chain for one input.

    Returns a dict with the edge names that the caller needs:
        orig_edge           -- 1920x1080 CUDA edge (after smooth_crop, with face metadata)
        face_edge           -- 608x1080 CUDA edge (face-cropped portrait)
        program_audio_edge  -- 48k/stereo/fltp audio for optional program output
        vad_audio_edge      -- 16k/mono/fltp audio for SileroVADNode
        vad_events_edge     -- MetadataFrame edge from SileroVADNode (speech events)
        visual_speech_edge  -- video edge from VisualSpeechGateNode (speaking metadata attached)
        input_group         -- avplumber group name for this input
    """
    g = f"input_{idx}"

    # ---- Input / demux ----
    input_rec_args = {
        "name": f"input_{idx}",
        "url": url,
        "dst": f"in{idx}_pkt",
        "group": g,
        "loop": True,
        "initial_timeout": 20,
        "timeout": 3_942_000_000,
    }
    if input_start_ts:
        input_rec_args["start_ts"] = input_start_ts
    avp.addNode(InputRec(input_rec_args))
    avp.addNode(Demux({
        "name": f"demux_{idx}",
        "src": f"in{idx}_pkt",
        "routing": {"?v:0": f"v{idx}_pkt", "?a:0": f"a{idx}_pkt"},
        "wait_for_keyframe": False,
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Video decode → realtime → force_fps ----
    avp.addNode(DecVideo({
        "name": f"dec_v{idx}",
        "src": f"v{idx}_pkt",
        "dst": f"v{idx}_dec",
        "pixel_format": "?cuda",
        "hwaccel": HWACCEL,
        "group": g,
        "auto_restart": "group",
    }))
    rt_kwargs: dict = {"set_pts": True, "group": g, "auto_restart": "group"}
    if sync_team:
        rt_kwargs["team"] = sync_team
    avp.addNode(Realtime({
        "name": f"rt_{idx}",
        "src": f"v{idx}_dec",
        "dst": f"v{idx}_rt",
        **rt_kwargs,
    }))
    avp.addNode(ForceFPS({
        "name": f"ffps_{idx}",
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "src": f"v{idx}_rt",
        "dst": f"v{idx}_fps",
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Video fan-out: full-res leg + YOLO leg + visual-speech full-res copy ----
    avp.addNode(Split({
        "name": f"split_v{idx}",
        "src": f"v{idx}_fps",
        "dst": [f"v{idx}_fullres", f"v{idx}_for_yolo", f"v{idx}_fullres_vs"],
        "group": g,
    }))

    # ---- YOLO face detection branch ----
    avp.addNode(FilterVideo({
        "name": f"yolo_scale_{idx}",
        "src": f"v{idx}_for_yolo",
        "dst": f"v{idx}_yolo_in",
        "graph": (
            f"scale_cuda=w={FACE_MODEL_W}:h={FACE_MODEL_CONTENT_H},"
            f"pad_cuda={FACE_MODEL_W}:{FACE_MODEL_H}:0:2"
        ),
        "hwaccel": HWACCEL,
        "group": g,
        "auto_restart": "on",
    }))
    avp.addNode(CudaInferYolo({
        "name": f"yolo_{idx}",
        "src": f"v{idx}_yolo_in",
        "dst": f"v{idx}_yolo_raw",
        "metadata_key_detection": FACE_METADATA_KEY,
        "models": [{
            "engine": face_engine,
            "task_type": "detection",
            "class_names": FACE_CLASS_NAMES,
            "output_box_format": "end2end_xyxy",
        }],
        "group": g,
        "auto_restart": "group",
    }))
    # Split raw YOLO output: one copy for face-tracking and one for visual-speech.
    # PlayerTracker filters to target_labels=["Face"] and may discard Mouth/Nose
    # detections; the visual-speech branch therefore needs the unmodified output.
    avp.addNode(Split({
        "name": f"split_yolo_{idx}",
        "src": f"v{idx}_yolo_raw",
        "dst": [f"v{idx}_yolo_for_tracker", f"v{idx}_yolo_for_vs"],
        "group": g,
    }))
    avp.addNode(PlayerTracker({
        "name": f"tracker_{idx}",
        "src": f"v{idx}_yolo_for_tracker",
        "dst": f"v{idx}_tracked",
        "metadata_key": FACE_METADATA_KEY,
        "target_labels": FACE_TRACKED_LABELS,
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Merge YOLO metadata back onto the full-res frame ----
    # join_metadata uses exact-PTS matching; both branches share the same
    # upstream force_fps PTS grid.
    avp.addNode(JoinMetadata({
        "name": f"join_{idx}",
        "src": [f"v{idx}_fullres", f"v{idx}_tracked"],
        "dst": f"v{idx}_fullres_md",
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Smooth viewport (produces portrait crop coordinates) ----
    avp.addNode(SmoothCropViewport({
        "name": f"smooth_vp_{idx}",
        "src": f"v{idx}_fullres_md",
        "dst": f"v{idx}_smooth",
        "metadata_key_ins": [FACE_METADATA_KEY],
        "metadata_key_out": VIEWPORT_METADATA_KEY,
        "viewport_dst_width": FACE_CROP_W,
        "viewport_dst_height": FACE_CROP_H,
        "focus_mode": "label_priority",
        "filter_type": "kalman",
        "lost_target": "hold_last",
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Split smooth output into: orig leg + crop-input leg ----
    avp.addNode(Split({
        "name": f"split_legs_{idx}",
        "src": f"v{idx}_smooth",
        "dst": [f"v{idx}_orig_raw", f"v{idx}_for_crop"],
        "group": g,
    }))
    # Smooth timestamps on the orig leg so the OTM always receives a well-formed
    # monotonic PTS sequence regardless of any irregularities introduced by the
    # face-detection chain (JoinMetadata frame-drops, SmoothCropViewport holds, …).
    avp.addNode(SmoothTimestamps({
        "name": f"smooth_ts_orig_{idx}",
        "src": f"v{idx}_orig_raw",
        "dst": f"v{idx}_orig",
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "group": g,
        "auto_restart": "group",
    }))
    # ---- Face crop: 1920x1080 → 608x1080 portrait ----
    avp.addNode(CropMetadataCuda({
        "name": f"face_crop_{idx}",
        "src": f"v{idx}_for_crop",
        "dst": f"v{idx}_face_916_raw",
        "metadata_key": VIEWPORT_METADATA_KEY,
        "offset_log_path": "/dev/null",
        "group": g,
        "auto_restart": "group",
    }))
    # Same smoothing on the face-crop leg.
    avp.addNode(SmoothTimestamps({
        "name": f"smooth_ts_face_{idx}",
        "src": f"v{idx}_face_916_raw",
        "dst": f"v{idx}_face_916",
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Visual-speech branch ----
    # Attach raw YOLO detections (Face, Mouth, Nose, Eye) onto a full-res copy
    # so that FaceAnchoredMouthTrackerNode can locate the mouth bounding box.
    # If the face engine doesn't detect sub-parts (Mouth/Nose), the tracker
    # falls back to geometric estimation, still providing a motion signal.
    mouth_key = f"{VS_MOUTH_KEY_PREFIX}_{idx}"
    vs_key = f"{VS_VISUAL_KEY_PREFIX}_{idx}"
    avp.addNode(JoinMetadata({
        "name": f"join_vs_{idx}",
        "src": [f"v{idx}_fullres_vs", f"v{idx}_yolo_for_vs"],
        "dst": f"v{idx}_vs_md",
        "group": g,
        "auto_restart": "group",
    }))
    avp.addNode(FaceAnchoredMouthTrackerNode({
        "name": f"mouth_tracker_{idx}",
        "src": f"v{idx}_vs_md",
        "dst": f"v{idx}_vs_mouth",
        "group": g,
        "source": f"input_{idx}",
        "input_metadata_key": FACE_METADATA_KEY,
        "output_metadata_key": mouth_key,
        "targets": [{"name": "primary"}],
        "run_in_wrapper_thread": True,
        "auto_restart": "group",
    }))
    avp.addNode(VisualSpeechGateNode({
        "name": f"vs_gate_{idx}",
        "src": f"v{idx}_vs_mouth",
        "dst": f"v{idx}_vs_out",
        "group": g,
        "source": f"input_{idx}",
        "mouth_metadata_key": mouth_key,
        "output_metadata_key": vs_key,
        "targets": [{"name": "primary"}],
        "run_in_wrapper_thread": True,
        "auto_restart": "group",
    }))

    # ---- Audio: decode → realtime → resample to fltp ----
    avp.addNode(DecAudio({
        "name": f"dec_a{idx}",
        "src": f"a{idx}_pkt",
        "dst": f"a{idx}_dec",
        "group": g,
        "auto_restart": "group",
    }))
    audio_rt_kwargs: dict = {"set_pts": True, "group": g, "auto_restart": "group"}
    if sync_team:
        audio_rt_kwargs["team"] = sync_team
    avp.addNode(Realtime({
        "name": f"rt_a{idx}",
        "src": f"a{idx}_dec",
        "dst": f"a{idx}_rt",
        **audio_rt_kwargs,
    }))
    avp.addNode(ResampleAudio({
        "name": f"resamp_{idx}",
        "src": f"a{idx}_rt",
        "dst": f"a{idx}_fltp",
        "dst_sample_rate": AUDIO_SAMPLE_RATE,
        "dst_channel_layout": AUDIO_CHANNEL_LAYOUT,
        "dst_sample_format": AUDIO_SAMPLE_FORMAT,
        "compensation": 0,
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Audio fan-out: Silero-VAD tap + optional program-audio output ----
    avp.addNode(Split({
        "name": f"split_a{idx}",
        "src": f"a{idx}_fltp",
        "dst": [f"a{idx}_vad_48k", f"a{idx}_program"],
        "group": g,
    }))
    # Silero requires 16 kHz / mono / float32 (flt or fltp).
    avp.addNode(ResampleAudio({
        "name": f"resamp_vad_{idx}",
        "src": f"a{idx}_vad_48k",
        "dst": f"a{idx}_vad_16k",
        "dst_sample_rate": VAD_SAMPLE_RATE,
        "dst_channel_layout": "mono",
        "dst_sample_format": "fltp",
        "compensation": 0,
        "group": g,
        "auto_restart": "group",
    }))
    # SileroVADNode reads the 16 kHz audio and emits speech-segment events.
    silero_args: dict = {
        "name": f"silero_{idx}",
        "src": f"a{idx}_vad_16k",
        "dst": f"a{idx}_vad_events",
        "group": g,
        "source": f"input_{idx}",
        "sample_rate": VAD_SAMPLE_RATE,
        "threshold": silero_threshold,
        "emit_state_events": True,
        "emit_state_updates": True,
        "repo_or_dir": silero_repo,
        "device": silero_device,
        "auto_restart": "group",
    }
    if silero_model:
        silero_args["model_path"] = silero_model
    avp.addNode(SileroVADNode(silero_args))

    return {
        "orig_edge": f"v{idx}_orig",
        "face_edge": f"v{idx}_face_916",
        "program_audio_edge": f"a{idx}_program",
        "vad_events_edge": f"a{idx}_vad_events",
        "visual_speech_edge": f"v{idx}_vs_out",
        "vs_key": vs_key,
        "input_group": g,
    }


# ------------------------------------------------------------------
# Scene definitions for a 9:16 canvas
# ------------------------------------------------------------------

def define_auto_scenes(mx: MixerGraphBuilder, n: int) -> None:
    """Register scenes for *n* inputs on the 1080x1920 (9:16) canvas.

    Source names follow the convention used in build_input_subgraph:
        face_{i}  -- 608x1080 portrait crop from input i (face-tracked 9:16)
        orig_{i}  -- 1920x1080 landscape frame from input i

    Scenes registered
    -----------------
    full_face_{i}
        Camera *i* face-tracked 9:16 portrait scaled to fill the full canvas.

    videoconf_{i}
        Camera *i* as a top 1:1 crop in the top 1080×1080 slot, with up
        to 5 other cameras shown as portrait thumbnails in the bottom strip.

    vstack3_{a}_{b}_{c}
        Three landscape sources (orig_a / orig_b / orig_c) stacked vertically,
        each scaled to 1080×608 (16:9).  Three tiles occupy 1824 px; the
        remaining 96 px are split evenly as top/bottom margins.

    vstack_{a}_{b}
        Two landscape sources stacked vertically (same as above but 2×16:9).

    pip_{i}_{j}
        Camera *i* face portrait fullscreen + camera *j* landscape thumbnail
        in the top-right corner.

    multiviewer
        2×2 grid of face portraits (up to 4 inputs, ≥ 3 required).
    """
    W, H = CANVAS_W, CANVAS_H

    # ------------------------------------------------------------------
    # 1. Full screen: dominant speaker 9:16 (face-tracked portrait)
    # ------------------------------------------------------------------
    for i in range(n):
        mx.add_scene(f"full_face_{i}", {
            f"face_{i}": {
                "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                "dst_x": 0,
                "dst_y": 0,
            }
        })

    # ------------------------------------------------------------------
    # 2. Videoconference: dominant speaker 1:1 (top) + others portrait below
    #
    # Top slot (1080 × 1080 — 1:1):
    #   face_{i} (608 × 1080) is cropped from the top to 608 × 608, then scaled.
    #
    # Bottom strip (1080 × 840):
    #   Up to 5 other cameras tiled horizontally as 9:16 portrait thumbnails,
    #   centred within the strip.
    # ------------------------------------------------------------------
    CONF_TOP_H = W                          # 1080 — square (1:1) top slot
    CONF_BOT_H = H - CONF_TOP_H            # 840
    CONF_MAIN_CROP_Y = 0

    for i in range(n):
        others = [j for j in range(n) if j != i][:5]
        n_oth = len(others)

        cell_w = (W // n_oth) & ~1
        cell_h = (cell_w * 16 // 9) & ~1
        if cell_h > CONF_BOT_H:
            # Portrait cells taller than the strip; constrain height and width.
            cell_h = CONF_BOT_H & ~1
            cell_w = (cell_h * 9 // 16) & ~1

        x_off = (W - n_oth * cell_w) // 2          # centre cells horizontally
        y_off = CONF_TOP_H + (CONF_BOT_H - cell_h) // 2  # centre in strip

        sources: dict = {
            f"face_{i}": {
                "graph": (
                    f"crop_cuda=w={FACE_CROP_W}:h={FACE_CROP_W}:x=0:y={CONF_MAIN_CROP_Y},"
                    f"scale_cuda=w={W}:h={CONF_TOP_H}:interp_algo=lanczos"
                ),
                "dst_x": 0,
                "dst_y": 0,
            }
        }
        for k, j in enumerate(others):
            sources[f"face_{j}"] = {
                "graph": f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos",
                "dst_x": x_off + k * cell_w,
                "dst_y": y_off,
            }
        mx.add_scene(f"videoconf_{i}", sources)

    # ------------------------------------------------------------------
    # 3. Vertical stack: 3 × 16:9 landscape sources
    #
    # Each tile: 1080 × 608 (16:9; 1080 × 9/16 = 607.5 → 608 < 0.1 % error).
    # Three tiles: 1824 px total; top/bottom margins of 48 px each.
    # ------------------------------------------------------------------
    if n >= 3:
        tile_w3 = W           # 1080
        tile_h3 = 608
        top3 = (H - 3 * tile_h3) // 2  # 48
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                for c in range(n):
                    if c == a or c == b:
                        continue
                    mx.add_scene(f"vstack3_{a}_{b}_{c}", {
                        f"orig_{a}": {
                            "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                            "dst_x": 0,
                            "dst_y": top3,
                        },
                        f"orig_{b}": {
                            "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                            "dst_x": 0,
                            "dst_y": top3 + tile_h3,
                        },
                        f"orig_{c}": {
                            "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                            "dst_x": 0,
                            "dst_y": top3 + 2 * tile_h3,
                        },
                    })

    # ------------------------------------------------------------------
    # PiP: face i fullscreen + orig j as thumbnail in top-right corner.
    # ------------------------------------------------------------------
    if n >= 2:
        pip_w = W // 3           # 360
        pip_h = (pip_w * 9 // 16) & ~1  # 202
        pip_x = W - pip_w - 16
        pip_y = 16
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mx.add_scene(f"pip_{i}_{j}", {
                    f"face_{i}": {
                        "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                        "dst_x": 0,
                        "dst_y": 0,
                    },
                    f"orig_{j}": {
                        "graph": f"scale_cuda=w={pip_w}:h={pip_h}:interp_algo=lanczos",
                        "dst_x": pip_x,
                        "dst_y": pip_y,
                    },
                })

    # ------------------------------------------------------------------
    # Vertical stack: 2 × 16:9 landscape sources.
    # Each tile: 1080 × 608.  Two tiles: 1216 px; margins of 352 px each.
    # ------------------------------------------------------------------
    if n >= 2:
        tile_w = W    # 1080
        tile_h = 608
        gap = (H - 2 * tile_h) // 2  # 352
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                mx.add_scene(f"vstack_{a}_{b}", {
                    f"orig_{a}": {
                        "graph": f"scale_cuda=w={tile_w}:h={tile_h}:interp_algo=lanczos",
                        "dst_x": 0,
                        "dst_y": gap,
                    },
                    f"orig_{b}": {
                        "graph": f"scale_cuda=w={tile_w}:h={tile_h}:interp_algo=lanczos",
                        "dst_x": 0,
                        "dst_y": gap + tile_h,
                    },
                })

    # ------------------------------------------------------------------
    # Multiviewer: 2×2 grid of face portraits (up to 4 inputs).
    # Each cell: 540 × 960 (9:16).  Two rows of two fill 1080 × 1920 exactly.
    # ------------------------------------------------------------------
    if n >= 3:
        cols = 2
        cell_w = W // cols          # 540
        cell_h = cell_w * 16 // 9  # 960
        grid_n = min(n, 4)
        sources_mv: dict = {}
        for j in range(grid_n):
            row, col = j // cols, j % cols
            sources_mv[f"face_{j}"] = {
                "graph": f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos",
                "dst_x": col * cell_w,
                "dst_y": row * cell_h,
            }
        mx.add_scene("multiviewer", sources_mv)


# ------------------------------------------------------------------
# Audio output
# ------------------------------------------------------------------

def build_audio_output(avp: AVPlumber, audio_edge: str, codec: str = "aac") -> str:
    """Encode the selected program-audio edge.

    Returns the name of the encoded audio edge for use in Mux.
    """
    avp.addNode(AssumeAudioFormat({
        "name": "assume_audio",
        "src": audio_edge,
        "dst": "a_program",
        "sample_rate": AUDIO_SAMPLE_RATE,
        "sample_format": AUDIO_SAMPLE_FORMAT,
        "channel_layout": AUDIO_CHANNEL_LAYOUT,
        "group": "output",
        "auto_restart": "panic",
    }))
    avp.addNode(EncAudio({
        "name": "enc_audio",
        "src": "a_program",
        "dst": "a_enc",
        "codec": codec,
        "options": {"b": "192k"},
        "group": "output",
        "auto_restart": "panic",
    }))
    return "a_enc"


# ------------------------------------------------------------------
# Video output
# ------------------------------------------------------------------

def build_video_output(avp: AVPlumber, video_edge: str, args: argparse.Namespace) -> str:
    """Add fps-normalizer, format hint, encoder.  Returns encoded video edge."""
    g = "output"
    avp.addNode(ForceFPS({
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "src": video_edge,
        "dst": "mixer_norm_fps",
        "group": g,
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "mixer_norm_fps",
        "dst": "mixer_norm",
        "width": CANVAS_W,
        "height": CANVAS_H,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "group": g,
        "auto_restart": "panic",
    }))
    avp.addNode(EncVideo({
        "src": "mixer_norm",
        "dst": "v_enc",
        "name": "enc_video",
        "codec": args.codec,
        "hwaccel": HWACCEL,
        "group": g,
        "options": {
            "b": "8000k",
            "maxrate": "14000k",
            "bufsize": "20000k",
            "g": 60,
            "bf": 0,
            "preset": "p3",
            "profile": "high",
        },
    }))
    return "v_enc"


# ------------------------------------------------------------------
# main()
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input URLs (at least 2)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output RTMP URL or file path",
    )
    parser.add_argument(
        "--face-engine",
        default=default_face_engine(),
        help="Path to the face detection TensorRT engine (.plan)",
    )
    parser.add_argument(
        "--codec", default="h264_nvenc",
        help="Video encoder codec (default: h264_nvenc)",
    )
    parser.add_argument(
        "--audio-codec", default="aac",
        help="Audio encoder codec (default: aac)",
    )
    parser.add_argument(
        "--sync-team", default="syncgroup",
        help="realtime sync_team name for live SRT sources (empty = independent)",
    )
    parser.add_argument(
        "--input-start-ts",
        help="Seek each input to this start timestamp (ms, MM:SS[.mmm], or HH:MM:SS[.mmm])",
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="Declare wipe subgraph (needed for mixer.wipe transitions)",
    )
    parser.add_argument(
        "--fade", default=0.6, type=float,
        help="Crossfade duration in seconds for auto-switching (default: 0.6)",
    )
    parser.add_argument(
        "--min-dwell", default=2.5, type=float,
        help="Minimum program dwell time in seconds before switching (default: 2.5)",
    )
    parser.add_argument(
        "--silero-model",
        default=os.environ.get("AVP_SILERO_MODEL"),
        help="Path to Silero VAD .jit model (downloads from torch.hub if unset)",
    )
    parser.add_argument(
        "--silero-repo",
        default=os.environ.get("AVP_SILERO_REPO", "snakers4/silero-vad"),
        help="torch.hub repo or local dir for Silero VAD (default: snakers4/silero-vad)",
    )
    parser.add_argument(
        "--silero-device", default="cpu",
        help="Device for Silero inference: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--silero-threshold", default=0.5, type=float,
        help="Silero speech probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--vad-holdoff", default=1.5, type=float,
        help="Deprecated; retained for CLI compatibility",
    )
    parser.add_argument(
        "--remote-control-port", default=0, type=int, metavar="PORT",
        help="TCP port for the avplumber remote control API (0 = disabled)",
    )
    parser.add_argument(
        "--logfile", default="",
        help="Write avplumber messages to this file and expose it to the Web UI",
    )
    parser.add_argument(
        "--webui-api", default="",
        help="Web UI server API endpoint URL for auto-registration (e.g. http://localhost:22222)",
    )
    parser.add_argument(
        "--instance-name", default="",
        help="Instance name for Web UI registration",
    )
    parser.add_argument(
        "--disable-auto-switcher", action="store_true",
        help="Start the graph without automatic scene changes",
    )
    args = parser.parse_args()

    n = len(args.inputs)
    if n < 2:
        parser.error("At least 2 inputs are required.")
    sergio_input_index = find_named_input(args.inputs, SERGIO_INPUT_NAME)
    rene_input_index = find_named_input(args.inputs, RENE_INPUT_NAME)

    avp = AVPlumber()
    avp.setLogFile(args.logfile)
    if args.remote_control_port:
        avp.enableControlServer(args.remote_control_port)
    if args.webui_api:
        avp.registerWithWebUI(args.webui_api, args.instance_name, args.logfile)
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    avp.edges.planCapacity("*", 4)

    # ---- Per-input subgraphs ----
    subgraphs = []
    for i, url in enumerate(args.inputs):
        sg = build_input_subgraph(
            avp, i, url,
            face_engine=args.face_engine,
            input_start_ts=args.input_start_ts,
            sync_team=args.sync_team,
            silero_model=args.silero_model,
            silero_repo=args.silero_repo,
            silero_device=args.silero_device,
            silero_threshold=args.silero_threshold,
        )
        subgraphs.append(sg)

    # ---- Mixer ----
    mx = MixerGraphBuilder(
        avp,
        name="mixer",
        canvas=(CANVAS_W, CANVAS_H),
        fps=(FPS_NUM, FPS_DEN),
        hwaccel=HWACCEL,
        enable_wipe=args.wipe,
    )

    for i, sg in enumerate(subgraphs):
        # Register the face-crop 9:16 source.
        mx.add_source(
            f"face_{i}",
            pre_otm_edge=sg["face_edge"],
            input_group=sg["input_group"],
        )
        # Register the original 16:9 source.
        mx.add_source(
            f"orig_{i}",
            pre_otm_edge=sg["orig_edge"],
            input_group=sg["input_group"],
        )

    define_auto_scenes(mx, n)
    mx.set_initial_scene("videoconf_0", slot="A")
    video_out_edge = mx.build()

    # ---- Audio output ----
    if rene_input_index is None:
        parser.error(f'Program audio input "{RENE_INPUT_NAME}" was not found in --inputs.')
    program_audio_edge = subgraphs[rene_input_index]["program_audio_edge"]
    for i, sg in enumerate(subgraphs):
        if i == rene_input_index:
            continue
        avp.addNode(NullSink({
            "name": f"program_audio_sink_{i}",
            "src": sg["program_audio_edge"],
            "group": "output",
            "auto_restart": "panic",
        }))
    audio_enc_edge = build_audio_output(avp, program_audio_edge, codec=args.audio_codec)

    # ---- Video output ----
    video_enc_edge = build_video_output(avp, video_out_edge, args)

    # ---- Mux + output ----
    avp.addNode(Mux({
        "src": [video_enc_edge, audio_enc_edge],
        "dst": "mux_out",
        "group": "output",
        "ts_sort_wait": 0,
    }))
    out_fmt = "flv" if args.output.startswith("rtmp://") else "mp4"
    avp.addNode(Output({
        "format": out_fmt,
        "url": args.output,
        "src": "mux_out",
        "group": "output",
        "auto_restart": "panic",
    }))

    # ---- Speech detection: Silero audio VAD + visual lip-motion ----
    speaker_registry = Speaker()
    for i, sg in enumerate(subgraphs):
        g = sg["input_group"]

        # Audio: SileroVADNode is already wired inside build_input_subgraph().
        # This bridge converts its live speech_start / speech_stop metadata events
        # to registry updates.
        avp.addNode(SileroVadRegistryBridge(
            {
                "name": f"vad_bridge_{i}",
                "src": sg["vad_events_edge"],
                "group": g,
                "auto_restart": "group",
            },
            index=i,
            registry=speaker_registry,
            holdoff_s=args.vad_holdoff,
        ))

        # Visual: VisualSpeechGateNode output is already wired in build_input_subgraph().
        avp.addNode(VisualSpeechRegistryNode(
            {
                "name": f"vs_registry_{i}",
                "src": sg["visual_speech_edge"],
                "group": g,
                "auto_restart": "group",
            },
            index=i,
            registry=speaker_registry,
            visual_metadata_key=sg["vs_key"],
        ))

    # ---- Start all groups ----
    for i in range(n):
        avp.group(f"input_{i}").startNodes()
    mx.start_groups()
    avp.group("output").startNodes()

    print(f"[auto_mixer] Graph started with {n} input(s) → {args.output}")
    print(f"[auto_mixer] Canvas: {CANVAS_W}x{CANVAS_H}, face engine: {args.face_engine}")
    if sergio_input_index is not None:
        print(f"[auto_mixer] Sergio input detected: {sergio_input_index} ({input_basename(args.inputs[sergio_input_index])})")
    if rene_input_index is not None:
        print(f"[auto_mixer] Rene input detected: {rene_input_index} ({input_basename(args.inputs[rene_input_index])})")
    print(f"[auto_mixer] Scenes: {mx.scenes()}")

    # ---- Auto-switcher ----
    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=n,
        full_face_scene=lambda i: f"videoconf_{i}",
        fade_duration_s=args.fade,
        min_dwell_program_s=args.min_dwell,
        min_active_level_db=MIN_ACTIVE_AUDIO_LEVEL_DBFS,
        special_speaker_index=rene_input_index,
        special_speaker_margin_db=RENE_REQUIRED_LEAD_DB,
        vad_only_priority_speaker_index=sergio_input_index,
    )
    if not args.disable_auto_switcher:
        switcher.start()

    # Unlock the control server so remote clients can issue commands.
    avp.setReady()

    # ---- Run until interrupted ----
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        print("\n[auto_mixer] Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            # Heartbeat / status log every 10 s.
            if int(time.monotonic()) % 10 == 0:
                entries = speaker_registry.all()
                for e in entries:
                    audio_s = "AUDIO" if e.speaking else "     "
                    visual_s = "VIS" if e.visual_speaking else "   "
                    duration_s = time.monotonic() - e.last_change_ts if e.speaking else e.speaking_duration_s
                    print(
                        f"[vad] input {e.index}: {audio_s} {visual_s}  "
                        f"{e.level_db:+.1f} dB  "
                        f"dur={duration_s:.1f}s"
                    )
                print(f"[mixer] PGM: {mx.current_scene}")
    finally:
        if not args.disable_auto_switcher:
            switcher.stop()
        print("[auto_mixer] Done.")


if __name__ == "__main__":
    main()
