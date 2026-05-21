"""Per-input graph construction for the auto mixer."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from pyplumber import AVPlumber
from pyplumber.audio_vad import Speaker
from pyplumber.mouth_tracker import FaceAnchoredMouthTrackerNode
from pyplumber.node import (
    AssumeVideoFormat,
    CropMetadataCuda,
    CudaInferYolo,
    DecAudio,
    DecVideo,
    Demux,
    DrawBBox,
    DrawBBoxLabels,
    FilterVideo,
    ForceFPS,
    InputRec,
    JoinMetadata,
    PlayerTracker,
    Realtime,
    ResampleAudio,
    SmoothCropViewport,
    SmoothTimestamps,
    Split,
)
from pyplumber.vad import SileroVADNode
from pyplumber.visual_speech import VisualSpeechGateNode

from .config import (
    AUDIO_CHANNEL_LAYOUT,
    AUDIO_SAMPLE_FORMAT,
    AUDIO_SAMPLE_RATE,
    FACE_CLASS_NAMES,
    FACE_CROP_H,
    FACE_CROP_W,
    FACE_METADATA_KEY,
    FACE_MODEL_CONTENT_H,
    FACE_MODEL_H,
    FACE_MODEL_W,
    FACE_TRACKED_LABELS,
    FPS_DEN,
    FPS_NUM,
    HWACCEL,
    STATIC_VIEWPORT_METADATA_KEY,
    VAD_SAMPLE_RATE,
    VIEWPORT_METADATA_KEY,
)
from .debug_overlay import (
    DEBUG_AUDIO_SPEAKING_LABELS,
    DEBUG_MOUTH_LABEL_COLORS,
    DEBUG_MOUTH_LABELS,
    DEBUG_VIDEO_SPEAKING_LABELS,
    SpeakingStatusLabelNode,
    StaticViewportMetadataNode,
    VS_MOUTH_KEY_PREFIX,
    VS_SPEAKING_LABEL_KEY_PREFIX,
    VS_VISUAL_KEY_PREFIX,
)


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


def default_face_engine() -> str:
    """Resolve the face TRT engine from explicit config or packaged defaults."""
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
    face_use_cuda_graph: bool = False,
    input_start_ts: str | None = None,
    sync_team: str = "",
    silero_model: str | None = None,
    silero_repo: str = "snakers4/silero-vad",
    silero_device: str = "cpu",
    silero_threshold: float = 0.5,
    static_face_crop: bool = False,
    debug_mouth_rois: bool = False,
    speaker_registry: Speaker | None = None,
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
        "drop": True,
        "group": g,
    }))

    # ---- YOLO face detection branch ----
    avp.addNode(FilterVideo({
        "name": f"yolo_scale_{idx}",
        "src": f"v{idx}_for_yolo",
        "dst": f"v{idx}_yolo_in",
        "graph": (
            f"scale_cuda=w={FACE_MODEL_W}:h={FACE_MODEL_CONTENT_H},"
            f"pad_cuda=w={FACE_MODEL_W}:h={FACE_MODEL_H}:x=0:y=2"
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
        "use_cuda_graph": face_use_cuda_graph,
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
    # PlayerTracker tracks only Face but preserves non-target detections as
    # passthrough metadata; the visual-speech branch still needs the unmodified
    # output for raw Mouth/Nose detections.
    yolo_split_dsts = [f"v{idx}_yolo_for_tracker", f"v{idx}_yolo_for_vs"]
    if debug_mouth_rois:
        yolo_split_dsts.append(f"v{idx}_yolo_for_debug")
    avp.addNode(Split({
        "name": f"split_yolo_{idx}",
        "src": f"v{idx}_yolo_raw",
        "dst": yolo_split_dsts,
        "drop": True,
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
        "allowed_labels": FACE_TRACKED_LABELS,
        "label_priority": FACE_TRACKED_LABELS,
        "filter_type": "kalman",
        "lost_target": "hold_last",
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
    speaking_label_key = f"{VS_SPEAKING_LABEL_KEY_PREFIX}_{idx}"
    vs_gate_dst = f"v{idx}_vs_out_raw" if debug_mouth_rois else f"v{idx}_vs_out"
    debug_visual_edge = vs_gate_dst
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
        "dst": vs_gate_dst,
        "group": g,
        "source": f"input_{idx}",
        "mouth_metadata_key": mouth_key,
        "output_metadata_key": vs_key,
        "targets": [{"name": "primary"}],
        "run_in_wrapper_thread": True,
        "auto_restart": "group",
    }))
    if debug_mouth_rois:
        debug_visual_edge = f"v{idx}_vs_debug"
        avp.addNode(Split({
            "name": f"split_vs_debug_{idx}",
            "src": vs_gate_dst,
            "dst": [f"v{idx}_vs_out", debug_visual_edge],
            "drop": True,
            "group": g,
        }))

    # ---- Optional visible mouth ROI debug overlay ----
    visible_src_edge = f"v{idx}_smooth"
    if debug_mouth_rois:
        avp.addNode(JoinMetadata({
            "name": f"join_mouth_debug_{idx}",
            "src": [f"v{idx}_smooth", f"v{idx}_yolo_for_debug"],
            "dst": f"v{idx}_debug_mouth_md",
            "group": g,
            "auto_restart": "group",
        }))
        avp.addNode(AssumeVideoFormat({
            "name": f"assume_debug_mouth_{idx}",
            "src": f"v{idx}_debug_mouth_md",
            "dst": f"v{idx}_debug_mouth_fmt",
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": g,
            "auto_restart": "group",
        }))
        avp.addNode(DrawBBox({
            "name": f"draw_mouth_debug_boxes_{idx}",
            "src": f"v{idx}_debug_mouth_fmt",
            "dst": f"v{idx}_debug_mouth_boxes",
            "group": g,
            "metadata_key": FACE_METADATA_KEY,
            "bbox_thickness": 4,
            "min_conf": 0.0,
            "allowed_labels": DEBUG_MOUTH_LABELS,
            "label_colors": DEBUG_MOUTH_LABEL_COLORS,
            "model_content_width": FACE_MODEL_W,
            "model_content_height": FACE_MODEL_H,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 300,
            "auto_restart": "group",
        }))
        if speaker_registry is None:
            raise ValueError("speaker_registry is required when debug_mouth_rois is enabled")
        avp.addNode(SpeakingStatusLabelNode({
            "name": f"speaking_label_metadata_{idx}",
            "src": f"v{idx}_debug_mouth_boxes",
            "dst": f"v{idx}_debug_speaking_md",
            "group": g,
            "visual_metadata_key": vs_key,
            "viewport_metadata_key": VIEWPORT_METADATA_KEY,
            "output_metadata_key": speaking_label_key,
            "model_width": FACE_MODEL_W,
            "model_height": FACE_MODEL_H,
            "static_face_crop": static_face_crop,
            "auto_restart": "group",
        }, index=idx, registry=speaker_registry))
        avp.addNode(DrawBBoxLabels({
            "name": f"draw_video_speaking_debug_label_{idx}",
            "src": f"v{idx}_debug_speaking_md",
            "dst": f"v{idx}_debug_video_speaking_label",
            "group": g,
            "metadata_key": speaking_label_key,
            "label_template": "{label}",
            "allowed_labels": DEBUG_VIDEO_SPEAKING_LABELS,
            "min_conf": 0.0,
            "model_content_width": FACE_MODEL_W,
            "model_content_height": FACE_MODEL_H,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "text_color": "white",
            "background_color": "green",
            "font_scale": 1,
            "debug_log_every_n": 300,
            "auto_restart": "group",
        }))
        avp.addNode(DrawBBoxLabels({
            "name": f"draw_audio_speaking_debug_label_{idx}",
            "src": f"v{idx}_debug_video_speaking_label",
            "dst": f"v{idx}_debug_speaking_labels",
            "group": g,
            "metadata_key": speaking_label_key,
            "label_template": "{label}",
            "allowed_labels": DEBUG_AUDIO_SPEAKING_LABELS,
            "min_conf": 0.0,
            "model_content_width": FACE_MODEL_W,
            "model_content_height": FACE_MODEL_H,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "text_color": "white",
            "background_color": "light_blue",
            "font_scale": 1,
            "debug_log_every_n": 300,
            "auto_restart": "group",
        }))
        visible_src_edge = f"v{idx}_debug_speaking_labels"

    # ---- Split visible full-res output into: orig leg + crop-input leg ----
    avp.addNode(Split({
        "name": f"split_legs_{idx}",
        "src": visible_src_edge,
        "dst": [f"v{idx}_orig_raw", f"v{idx}_for_crop"],
        "drop": True,
        "group": g,
    }))
    # Smooth timestamps on the orig leg so the OTM always receives a well-formed
    # monotonic PTS sequence regardless of any irregularities introduced by the
    # face-detection chain (JoinMetadata frame-drops, SmoothCropViewport holds, ...).
    avp.addNode(SmoothTimestamps({
        "name": f"smooth_ts_orig_{idx}",
        "src": f"v{idx}_orig_raw",
        "dst": f"v{idx}_orig",
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "group": g,
        "auto_restart": "group",
    }))
    if static_face_crop:
        # For this input, keep the tracker-side graph shape but ignore the
        # tracked viewport when producing the portrait source.
        avp.addNode(StaticViewportMetadataNode({
            "name": f"static_vp_{idx}",
            "src": f"v{idx}_for_crop",
            "dst": f"v{idx}_static_vp",
            "metadata_key": STATIC_VIEWPORT_METADATA_KEY,
            "viewport_dst_width": FACE_CROP_W,
            "viewport_dst_height": FACE_CROP_H,
            "group": g,
            "auto_restart": "group",
        }))
        avp.addNode(CropMetadataCuda({
            "name": f"face_crop_{idx}",
            "src": f"v{idx}_static_vp",
            "dst": f"v{idx}_face_916_raw",
            "metadata_key": STATIC_VIEWPORT_METADATA_KEY,
            "offset_log_path": "/dev/null",
            "group": g,
            "auto_restart": "group",
        }))
    else:
        # ---- Face crop: 1920x1080 -> 608x1080 portrait ----
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
        "drop": True,
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
