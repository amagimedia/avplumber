#!/usr/bin/env python3

import os
import sys
import time
import json
from pathlib import Path

sys.path.append("../..")
import pyplumber  # pyright: ignore[reportMissingImports]
from pyplumber.node import (  # pyright: ignore[reportMissingImports]
    PythonNode,
    AssumeVideoFormat,
    BallHandler,
    BallTracker,
    CudaInferYolo,
    DecVideo,
    Demux,
    DrawBBox,
    DrawBBoxLabels,
    DrawTrail,
    EncVideo,
    FilterVideo,
    ForceFPS,
    InputRec,
    JoinMetadata,
    MediaPipeAutoflipCropMetadata,
    Mux,
    Output,
    PlayerTracker,
    Realtime,
    ShotClassifier,
    Split,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")
MODELS_DIR = os.environ.get("AVP_MODELS_DIR", "/home/user/tensorrt/")
BALL_TRACK_DUMP = os.environ.get("AVP_BALL_TRACK_DUMP", "track-dump.txt")
CROP_DUMP = os.environ.get("AVP_CROP_DUMP", "/tmp/crop_coords.txt")

IS_RTP = OUTPUT_FORMAT == "rtp" or OUTPUT_URL.startswith("rtp://")

ENC_OPTIONS = (
    {"b": "4000k", "maxrate": "4000k", "bufsize": "4000k",
     "rc": "cbr", "g": 25, "bf": 0, "preset": "p6",
     "profile": "baseline", "level": "4.0", "tune": "ull",
     "zerolatency": 1, "delay": 0}
    if IS_RTP else
    {"b": "8000k", "maxrate": "8000k", "bufsize": "8000k",
     "rc": "cbr", "g": 75, "bf": 0, "preset": "p7",
     "tune": "ll", "profile": "high", "multipass": "disabled",
     "zerolatency": 1, "spatial_aq": 1, "temporal_aq": 1}
)
OUTPUT_OPTIONS = (
    {"payload_type": 96, "rtpflags": "skip_rtcp", "ssrc": 1096093697}
    if IS_RTP else {}
)


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def main() -> None:
    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 7)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux0",
            "group": "in",
            "name": "input",
            "initial_timeout": 20,
            "timeout": 10,
            "loop": True,
            "auto_restart": "group",
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": "in",
            "auto_restart": "group",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "in",
            "name": "Video_Dec",
            "auto_restart": "group",
            "optional": True,
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }),
        Realtime({
            "src": "v_dec_cuda",
            "dst": "v_dec_rt",
            "group": "in",
            "auto_restart": "group",
            "set_pts": True,
        }),
        ForceFPS({
            "fps": "25/1",
            "group": "in",
            "src": "v_dec_rt",
            "dst": "v_dec_30fps",
            "auto_restart": "group",
        }),
        Split({
            "src": "v_dec_30fps",
            "dst": ["v_dec_1080p", "v_dec_for_yolo"],
            "group": "in",
            "auto_restart": "group",
        }),
        FilterVideo({
            "graph": "scale_cuda=w=960:h=540,pad_cuda=960:544:0:2",
            "src": "v_dec_for_yolo",
            "dst": "v_pre_yolo",
            "group": "in",
            "name": "Scale_Yolo",
            "auto_restart": "group",
            "dst_width": 960,
            "dst_height": 544,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        Split({
            "src": "v_pre_yolo",
            "dst": ["v_for_players", "v_for_ball", "v_for_seg"],
            "group": "in",
            "auto_restart": "group",
        }),
        CudaInferYolo({
            "src": "v_for_players",
            "dst": "v_post_players_to_python",
            "group": "in",
            "name": "Yolo_Players",
            "auto_restart": "group",
            "input_format": "RGB",
            "conf_thresh": 0.25,
            "max_det": 40,
            "infer_every_n": 1,
            "metadata_key_detection": "yolo_players",
            "debug_log_metadata": True,
            "debug_log_every_n": 30,
            "mask_gpu_every_n": 0,
            "mask_cpu_every_n": 0,
            "models": [{
                "engine": model_path("basketball-players-full_960x544.plan"),
                "task_type": "detection",
                "class_names": [
                    "_suppress",
                    "Hoop",
                    "Period",
                    "Player",
                    "Ref",
                    "Shot Clock",
                    "Team Name",
                    "Team Points",
                    "Time Remaining",
                ],
                "output_box_format": "end2end_xyxy",
            }],
        }),
        CudaInferYolo({
            "src": "v_for_ball",
            "dst": "v_post_ball",
            "group": "in",
            "name": "Yolo_Ball",
            "auto_restart": "group",
            "input_format": "RGB",
            "conf_thresh": 0.04,
            "max_det": 10,
            "infer_every_n": 1,
            "metadata_key_detection": "yolo_ball",
            "debug_log_metadata": True,
            "debug_log_every_n": 30,
            "mask_gpu_every_n": 0,
            "mask_cpu_every_n": 0,
            "models": [{
                "engine": model_path("ball_960x544.plan"),
                "task_type": "detection",
                "class_names": ["basketball"],
                "output_box_format": "end2end_xyxy",
            }],
        }),
        CudaInferYolo({
            "src": "v_for_seg",
            "dst": "v_post_seg",
            "group": "in",
            "name": "Yolo_Seg",
            "auto_restart": "group",
            "input_format": "RGB",
            "conf_thresh": 0.25,
            "max_det": 10,
            "infer_every_n": 1,
            "metadata_key_detection": "yolo_seg_det",
            "metadata_key_segmentation": "yolo_seg",
            "debug_log_metadata": True,
            "debug_log_every_n": 30,
            "mask_gpu_every_n": 1,
            "mask_cpu_every_n": 1,
            "models": [{
                "engine": model_path("court-segmentation_960x544.plan"),
                "task_type": "segmentation",
                "class_names": ["three point line", "basketball-court"],
                "output_box_format": "end2end_xyxy",
                "include_in_detection_metadata": True,
            }],
        }),
        JoinMetadata({
            "src": ["v_post_players_to_python", "v_post_ball"],
            "dst": "v_players_ball",
            "group": "in",
            "auto_restart": "group",
        }),
        JoinMetadata({
            "src": ["v_players_ball", "v_post_seg"],
            "dst": "v_inferred",
            "group": "in",
            "auto_restart": "group",
        }),
        ShotClassifier({
            "src": "v_inferred",
            "dst": "v_classified",
            "group": "in",
            "seg_metadata_key": "yolo_seg",
            "player_metadata_key": "yolo_players",
            "player_labels": ["Player"],
            "court_class_indices": [0, 1],
            "wide_court_threshold": 0.25,
            "closeup_court_threshold": 0.05,
            "ambiguous_min_players": 3,
            "high_player_override": 5,
            "player_height_fraction": 0.25,
            "player_height_tolerance": 0.45,
            "player_min_aspect_ratio": 0.75,
            "min_stable_frames": 3,
            "debug_log_every_n": 1,
            "auto_restart": "group",
        }),
        PlayerTracker({
            "src": "v_classified",
            "dst": "v_tracked_players",
            "group": "in",
            "metadata_key": "yolo_players",
            "target_labels": ["Player"],
            "frame_rate": 25,
            "track_buffer": 90,
            "predict_on_empty": True,
            "track_thresh": 0.2,
            "high_thresh": 0.85,
            "shot_metadata_key": "shot_info",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        BallTracker({
            "src": "v_tracked_players",
            "dst": "v_ball_tracked",
            "group": "in",
            "metadata_key": "yolo_ball",
            "target_label": "basketball",
            "coast": True,
            "min_conf": 0.04,
            "shot_metadata_key": "shot_info",
            "coast_edge_jump_veto_enabled": True,
            "coast_edge_zone_rel": 0.14,
            "coast_edge_jump_rel": 0.18,
            "coast_edge_confirm_frames": 9,
            "debug_log_every_n": 1,
            "dump_file": BALL_TRACK_DUMP,
            "auto_restart": "group",
        }),
        BallHandler({
            "src": "v_ball_tracked",
            "dst": "v_tracked",
            "group": "in",
            "ball_metadata_key": "yolo_ball",
            "player_metadata_key": "yolo_players",
            "output_metadata_key": "ball_handler",
            "ball_label": "basketball",
            "player_labels": ["Player"],
            "max_distance_px": 35,
            "hysteresis_frames": 12,
            "shot_metadata_key": "shot_info",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        JoinMetadata({
            "src": ["v_dec_1080p", "v_tracked"],
            "dst": "v_1080p_with_md",
            "group": "in",
            "auto_restart": "group",
        }),
        DrawTrail({
            "src": "v_1080p_with_md",
            "dst": "v_trail_drawn",
            "group": "in",
            "name": "Draw_Trail",
            "metadata_key": "yolo_ball",
            "color": "red",
            "thickness": 2,
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_trail_drawn",
            "dst": "v_bbox_players",
            "group": "in",
            "name": "Draw_Players",
            "metadata_key": "yolo_players",
            "bbox_thickness": 2,
            "min_conf": 0.25,
            "allowed_labels": [
                "Hoop",
                "Period",
                "Player",
                "Ref",
                "Shot Clock",
                "Team Name",
                "Team Points",
                "Time Remaining",
            ],
            "label_colors": {
                "Hoop": "yellow",
                "Period": "cyan",
                "Player": "green",
                "Ref": "orange",
                "Shot Clock": "purple",
                "Team Name": "light_blue",
                "Team Points": "magenta",
                "Time Remaining": "white",
            },
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        DrawBBoxLabels({
            "src": "v_bbox_players",
            "dst": "v_labels_drawn",
            "group": "in",
            "name": "Draw_Labels",
            "metadata_key": "yolo_players",
            "label_template": "ID:{track_id}",
            "allowed_labels": ["Player"],
            "min_conf": 0.25,
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "text_color": "white",
            "background_color": "black",
            "font_scale": 2,
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_labels_drawn",
            "dst": "v_bbox_ball",
            "group": "in",
            "name": "Draw_Ball",
            "metadata_key": "yolo_ball",
            "bbox_thickness": 2,
            "min_conf": 0.04,
            "allowed_labels": ["basketball"],
            "label_colors": {"basketball": "red"},
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_bbox_ball",
            "dst": "v_bbox_handler",
            "group": "in",
            "name": "Draw_Handler",
            "metadata_key": "ball_handler",
            "bbox_thickness": 4,
            "min_conf": 0.0,
            "allowed_labels": ["BallHandler"],
            "label_colors": {"BallHandler": "magenta"},
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        MediaPipeAutoflipCropMetadata({
            "src": "v_bbox_handler",
            "dst": "v_viewport_md",
            "name": "Autoflip_Kinematic",
            "metadata_key_out": "smoothed_crop_viewport_v1",
            "crop_w": 608,
            "crop_h": 1080,
            "model_content_width": 960,
            "model_content_height": 544,
            "model_content_offset_x": 0,
            "model_content_offset_y": 2,
            "saliency": [
                {"metadata_key": "ball_handler", "role": "preferred", "weight": 100,
                 "min_conf": 0.30, "optional_input": True},
                {"metadata_key": "yolo_ball",    "role": "preferred", "weight": 70,
                 "min_conf": 0.10},
                {"metadata_key": "yolo_players", "role": "preferred", "weight": 60,
                 "min_conf": 0.25, "optional_input": True,
                 "shot_type_key": "shot_info", "shot_type_field": "camera_shot",
                 "shot_type_value": "closeup"},
                {"metadata_key": "yolo_players", "role": "preferred", "weight": 60,
                 "min_conf": 0.25, "optional_input": True,
                 "fallback_when_empty": True},
            ],
            "min_motion_to_reframe": 0.001,
            "max_velocity": 2000,
            "reset_on_scene_cut": False,
            "debug_log_every_n": 30,
            "group": "in",
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_viewport_md",
            "dst": "v_viewport_drawn",
            "group": "in",
            "name": "Draw_Viewport",
            "metadata_key": "smoothed_crop_viewport_v1",
            "bbox_thickness": 2,
            "min_conf": 0.0,
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        ForceFPS({
            "fps": "25/1",
            "group": "in",
            "src": "v_annotated_cuda",
            "dst": "v_annotated_fps",
            "auto_restart": "group",
        }),
        FilterVideo({
            "graph": "scale_cuda=w=1920:h=1080",
            "src": "v_annotated_fps",
            "dst": "v_scaled_out",
            "group": "in",
            "name": "Scale_Output_1080p",
            "auto_restart": "group",
            "dst_width": 1920,
            "dst_height": 1080,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        AssumeVideoFormat({
            "width": 1920,
            "height": 1080,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": "in",
            "src": "v_scaled_out",
            "dst": "v_preenc",
            "auto_restart": "group",
        }),
        EncVideo({
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": "in",
            "name": "Video_Encode_NVENC",
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "options": ENC_OPTIONS,
            "auto_restart": "group",
        }),
        Mux({
            "src": ["v_outenc"],
            "dst": "mux_v",
            "group": "in",
            "ts_sort_wait": 0,
            "auto_restart": "group",
        }),
        Output({
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "src": "mux_v",
            "group": "in",
            **({"options": OUTPUT_OPTIONS} if OUTPUT_OPTIONS else {}),
            "auto_restart": "group",
        }),
    ]

    class CropDumpNode(PythonNode):
        def __init__(self, params):
            super().__init__(params)
            self._file = open(CROP_DUMP, "w", buffering=1)
            self._frame = 0

        def process(self):
            frm = self._src.get()
            if frm:
                self._frame += 1
                vp_raw = frm.metadata.as_dict.get("smoothed_crop_viewport_v1", "")
                if vp_raw:
                    try:
                        vp = json.loads(vp_raw)
                        bbox = vp.get("viewport_bbox", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            self._file.write(
                                f"{self._frame} {x1} {y1} {x2} {y2}\n"
                            )
                    except Exception:
                        pass
                self._dst.enqueue(frm)

    nodes.append(
        CropDumpNode({
            "src": "v_viewport_drawn",
            "dst": "v_annotated_cuda",
            "name": "Crop_Dump",
            "group": "in",
            "auto_restart": "group",
        })
    )

    for node in nodes:
        avp.addNode(node)

    print("Starting group: in")
    avp.group("in").startNodes()

    while True:
        time.sleep(1)
        avp.heartbeat()


if __name__ == "__main__":
    main()
