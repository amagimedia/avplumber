"""Automatic scene switcher with face reframer and audio VAD.

Each input fans out to:
  - an original 16:9 leg (for PiP / multiviewer / vstack layouts)
  - a face-tracked 9:16 leg (for fullscreen and videoconference layouts)

Audio RMS VAD determines the active speaker and drives mixer.fade().
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
        [--rtmp-output rtmp://out] \\
        [--codec h264_nvenc]

Environment variables
---------------------
    AVP_FACE_ENGINE   default face TRT engine path (overridden by --face-engine)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

sys.path.insert(0, ".")

from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeVideoFormat,
    CropMetadataCuda,
    CudaInferYolo,
    DecAudio,
    DecVideo,
    Demux,
    EncAudio,
    EncVideo,
    FilterAudio,
    FilterVideo,
    ForceFPS,
    InputRec,
    JoinMetadata,
    Mux,
    Output,
    PlayerTracker,
    Realtime,
    ResampleAudio,
    SmoothCropViewport,
    Split,
)
from pyplumber.mixer import MixerGraphBuilder
from pyplumber.audio_vad import RmsVadNode, Speaker
from pyplumber.auto_switcher import AutoSwitcher

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

# YOLO face class labels used by the model.
FACE_CLASS_NAMES = ["face", "Eye", "Nose", "Mouth"]
FACE_TRACKED_LABELS = ["face"]

AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNEL_LAYOUT = "stereo"
AUDIO_SAMPLE_FORMAT = "fltp"

# ------------------------------------------------------------------
# Per-input subgraph
# ------------------------------------------------------------------

def build_input_subgraph(
    avp: AVPlumber,
    idx: int,
    url: str,
    face_engine: str,
    sync_team: str = "",
) -> dict:
    """Build decode + face-detection + audio chain for one input.

    Returns a dict with the edge names that the caller needs:
        orig_edge   -- 1920x1080 CUDA edge (after smooth_crop, with face metadata)
        face_edge   -- 608x1080 CUDA edge (face-cropped portrait)
        amix_edge   -- 48k/stereo/fltp audio for the amix bus
        vad_edge    -- 48k/stereo/fltp audio for the VAD node
    """
    g = f"input_{idx}"

    # ---- Input / demux ----
    avp.addNode(InputRec({
        "name": f"input_{idx}",
        "url": url,
        "dst": f"in{idx}_pkt",
        "group": g,
        "loop": True,
        "initial_timeout": 20,
        "timeout": 3_942_000_000,
    }))
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
    rt_kwargs: dict = {"set_pts": True, "group": g}
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
    }))

    # ---- Video fan-out: full-res leg + YOLO leg ----
    avp.addNode(Split({
        "name": f"split_v{idx}",
        "src": f"v{idx}_fps",
        "dst": [f"v{idx}_fullres", f"v{idx}_for_yolo"],
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
        "dst": f"v{idx}_yolo_out",
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
    avp.addNode(PlayerTracker({
        "name": f"tracker_{idx}",
        "src": f"v{idx}_yolo_out",
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
        "dst": [f"v{idx}_orig", f"v{idx}_for_crop"],
        "group": g,
    }))

    # ---- Face crop: 1920x1080 → 608x1080 portrait ----
    avp.addNode(CropMetadataCuda({
        "name": f"face_crop_{idx}",
        "src": f"v{idx}_for_crop",
        "dst": f"v{idx}_face_916",
        "metadata_key": VIEWPORT_METADATA_KEY,
        "offset_log_path": "/dev/null",
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Audio: decode → resample to fltp ----
    avp.addNode(DecAudio({
        "name": f"dec_a{idx}",
        "src": f"a{idx}_pkt",
        "dst": f"a{idx}_dec",
        "group": g,
        "auto_restart": "group",
    }))
    avp.addNode(ResampleAudio({
        "name": f"resamp_{idx}",
        "src": f"a{idx}_dec",
        "dst": f"a{idx}_fltp",
        "dst_sample_rate": AUDIO_SAMPLE_RATE,
        "dst_channel_layout": AUDIO_CHANNEL_LAYOUT,
        "dst_sample_format": AUDIO_SAMPLE_FORMAT,
        "compensation": 0,
        "group": g,
        "auto_restart": "group",
    }))

    # ---- Audio fan-out: VAD tap + amix bus ----
    avp.addNode(Split({
        "name": f"split_a{idx}",
        "src": f"a{idx}_fltp",
        "dst": [f"a{idx}_vad", f"a{idx}_amix"],
        "group": g,
    }))

    return {
        "orig_edge": f"v{idx}_orig",
        "face_edge": f"v{idx}_face_916",
        "amix_edge": f"a{idx}_amix",
        "vad_edge": f"a{idx}_vad",
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
        Camera *i* as a static 1:1 crop in the top 1080×1080 slot, with up to
        5 other cameras shown as portrait thumbnails in the bottom strip.
        The 1:1 area is produced by scaling the face-tracked portrait to fill
        1080 px wide; the canvas clips the bottom, leaving the head/shoulders
        region visible — no extra face-tracking pass required.

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
    #   face_{i} (608 × 1080) scaled to fill 1080 px wide → ~1918 px tall.
    #   The canvas clips the bottom, so only the upper 1080 px are rendered —
    #   a static 1:1 crop of the already face-tracked portrait.
    #
    # Bottom strip (1080 × 840):
    #   Up to 5 other cameras tiled horizontally as 9:16 portrait thumbnails,
    #   centred within the strip.
    # ------------------------------------------------------------------
    CONF_TOP_H = W                          # 1080 — square (1:1) top slot
    CONF_BOT_H = H - CONF_TOP_H            # 840
    # Height when face portrait (608 × 1080) is scaled to fill 1080 px wide.
    # Ceiling-divide then round to even to avoid any letterbox gap at the edge.
    _face_fill_h = (W * FACE_CROP_H // FACE_CROP_W + 1) & ~1  # 1918

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
                "graph": f"scale_cuda=w={W}:h={_face_fill_h}:interp_algo=lanczos",
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
# Audio output: amix all inputs
# ------------------------------------------------------------------

def build_audio_output(avp: AVPlumber, amix_edges: list, codec: str = "aac") -> str:
    """Wire all per-input audio splits into a single amix node and encode.

    Returns the name of the encoded audio edge for use in Mux.
    """
    n = len(amix_edges)
    avp.addNode(FilterAudio({
        "name": "amix",
        "src": amix_edges,
        "dst": "a_mixed",
        "graph": f"amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0",
        "group": "output",
        "auto_restart": "panic",
    }))
    avp.addNode(EncAudio({
        "name": "enc_audio",
        "src": "a_mixed",
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
        default=os.environ.get("AVP_FACE_ENGINE", "/opt/tly/engines/yolo_face.plan"),
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
        "--sync-team", default="",
        help="realtime sync_team name for live SRT sources (empty = independent)",
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
    args = parser.parse_args()

    n = len(args.inputs)
    if n < 2:
        parser.error("At least 2 inputs are required.")

    avp = AVPlumber()
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    avp.edges.planCapacity("*", 4)

    # ---- Per-input subgraphs ----
    subgraphs = []
    for i, url in enumerate(args.inputs):
        sg = build_input_subgraph(
            avp, i, url,
            face_engine=args.face_engine,
            sync_team=args.sync_team,
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
    mx.set_initial_scene("full_face_0", slot="A")
    video_out_edge = mx.build()

    # ---- Audio output ----
    amix_edges = [sg["amix_edge"] for sg in subgraphs]
    audio_enc_edge = build_audio_output(avp, amix_edges, codec=args.audio_codec)

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

    # ---- VAD nodes ----
    speaker_registry = Speaker()
    for i, sg in enumerate(subgraphs):
        vad_node = RmsVadNode(
            {
                "name": f"vad_{i}",
                "src": sg["vad_edge"],
                "group": sg["input_group"],
                "auto_restart": "group",
            },
            index=i,
            registry=speaker_registry,
        )
        avp.addNode(vad_node)

    # ---- Start all groups ----
    for i in range(n):
        avp.group(f"input_{i}").startNodes()
    mx.start_groups()
    avp.group("output").startNodes()

    print(f"[auto_mixer] Graph started with {n} input(s) → {args.output}")
    print(f"[auto_mixer] Canvas: {CANVAS_W}x{CANVAS_H}, face engine: {args.face_engine}")
    print(f"[auto_mixer] Scenes: {mx.scenes()}")

    # ---- Auto-switcher ----
    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=n,
        full_face_scene=lambda i: f"full_face_{i}",
        fade_duration_s=args.fade,
        min_dwell_program_s=args.min_dwell,
    )
    switcher.start()

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
                    state = "SPEAKING" if e.speaking else "silent "
                    print(
                        f"[vad] input {e.index}: {state}  "
                        f"{e.level_db:+.1f} dB  "
                        f"dur={e.speaking_duration_s:.1f}s"
                    )
                print(f"[mixer] PGM: {mx.current_scene}")
    finally:
        switcher.stop()
        print("[auto_mixer] Done.")


if __name__ == "__main__":
    main()
