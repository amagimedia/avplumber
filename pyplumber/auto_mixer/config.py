"""Shared constants for the auto mixer graph."""

from __future__ import annotations

# Portrait 9:16 program canvas.
CANVAS_W = 1080
CANVAS_H = 1920
FPS_NUM = 30
FPS_DEN = 1
HWACCEL = "@gpu"

# Face detection model input size.
FACE_MODEL_W = 960
FACE_MODEL_H = 544
FACE_MODEL_CONTENT_H = 540

# Face tracking metadata key names.
FACE_METADATA_KEY = "yolo_faces"
VIEWPORT_METADATA_KEY = "smoothed_crop_viewport_v1"
STATIC_VIEWPORT_METADATA_KEY = "static_crop_viewport_v1"

# 9:16 portrait crop from a 1920x1080 frame.
FACE_CROP_W = 608
FACE_CROP_H = 1080

# YOLO face-part class labels used by the face-recognition-1.2 model.
FACE_CLASS_NAMES = ["Eye", "Face", "MakeUp", "Mouth", "Nose", "Tooth", "Topping"]
FACE_TRACKED_LABELS = ["Face"]

AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNEL_LAYOUT = "stereo"
AUDIO_SAMPLE_FORMAT = "fltp"
OPUS_SAMPLE_FORMAT = "fltp"

# Silero VAD requires 16 kHz mono float audio.
VAD_SAMPLE_RATE = 16000
MIN_ACTIVE_AUDIO_LEVEL_DBFS = -60.0

JANUS_DEFAULT_HOST = "127.0.0.1"
JANUS_DEFAULT_VIDEO_PORT = 5004
JANUS_DEFAULT_AUDIO_PORT = 5002
JANUS_DEFAULT_VIDEO_BITRATE_KBPS = 3000
RTP_PKT_SIZE = 1200

SAMPLED_MANUAL_SCENE_COUNT = 5
PIP_SCENE_SAMPLE_SEED = 20260518
VSTACK2_SCENE_SAMPLE_SEED = 20260519
VSTACK3_SCENE_SAMPLE_SEED = 20260520
