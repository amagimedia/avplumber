#!/usr/bin/env python3
"""Lean basketball tactical-view pipeline.

Players + ball + court/player segmentation + tracker + feet + team
classification, segmentation-driven court calibration (court_calibration.py,
no pose model) and the draw_tactical_court overlay. No metadata dumping, no
scoreboard/game-state nodes.

Run (inside the CUDA/TensorRT runtime container):
  python3 pyplumber/examples/tactical_view.py \
      --input /media/input.mp4 --output /artifacts/tactical.ts
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pyplumber
from pyplumber.court_calibration import CourtCalibrationNode

MODELS_DIR = os.environ.get("AVP_MODELS_DIR", "/models")


def graph_script(input_url, output_url, models):
    m = models
    return f"""
queue.plan_capacity * 14

hwaccel.init {{ "name": "@gpu", "type": "cuda" }}

node.add {{ "type": "input_rec", "url": "{input_url}", "dst": "in_mux0", "group": "in", "name": "input", "initial_timeout": 20, "timeout": 10, "loop": false, "on_error": "panic" }}
node.add {{ "type": "demux", "src": "in_mux0", "wait_for_keyframe": false, "routing": {{ "?v:0": "v_pkt" }}, "group": "in" }}
node.add {{ "type": "dec_video", "src": "v_pkt", "dst": "v_dec_cuda", "group": "in", "name": "Video_Dec", "optional": true, "pixel_format": "?cuda", "hwaccel": "@gpu", "codec_map": {{ "h264": "h264_cuvid", "hevc": "hevc_cuvid" }}, "hwaccel_only_for_codecs": ["h264", "hevc"] }}
node.add {{ "type": "force_fps", "fps": "25/1", "group": "in", "src": "v_dec_cuda", "dst": "v_dec_25fps" }}

node.add {{ "type": "split", "src": "v_dec_25fps", "dst": ["v_dec_1080p_main", "v_dec_for_yolo"], "group": "in" }}
node.add {{ "type": "filter_video", "graph": "scale_cuda=w=960:h=540,pad_cuda=960:544:0:2", "src": "v_dec_for_yolo", "dst": "v_pre_yolo", "group": "in", "name": "Scale_Yolo", "dst_width": 960, "dst_height": 544, "dst_pixel_format": "cuda", "hwaccel": "@gpu" }}
node.add {{ "type": "split", "src": "v_pre_yolo", "dst": ["v_for_players", "v_for_ball", "v_for_seg", "v_for_player_seg", "v_for_pose"], "group": "in" }}

node.add {{ "type": "cuda_infer_yolo", "src": "v_for_pose", "dst": "v_post_pose", "group": "in", "name": "Yolo_Pose", "input_format": "RGB", "conf_thresh": 0.18, "max_det": 1, "infer_every_n": 1, "metadata_key_pose": "yolo_pose", "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{{ "engine": "{m}/pose-small/pose-small.plan", "task_type": "pose", "class_names": ["court"], "num_classes": 1, "nms_iou_thresh": 0.25, "output_box_format": "raw_cxcywh", "include_in_detection_metadata": false }}] }}

node.add {{ "type": "cuda_infer_yolo", "src": "v_for_players", "dst": "v_post_players", "group": "in", "name": "Yolo_Players", "input_format": "RGB", "conf_thresh": 0.25, "max_det": 40, "infer_every_n": 1, "metadata_key_detection": "yolo_players", "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{{ "engine": "{m}/basketball-players-full_960x544.plan", "task_type": "detection", "class_names": ["_suppress", "Hoop", "_suppress", "Player", "Ref", "_suppress", "_suppress", "_suppress", "_suppress"], "output_box_format": "end2end_xyxy" }}] }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_ball", "dst": "v_post_ball", "group": "in", "name": "Yolo_Ball", "input_format": "RGB", "conf_thresh": 0.04, "max_det": 10, "infer_every_n": 1, "metadata_key_detection": "yolo_ball", "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{{ "engine": "{m}/ball_960x544.plan", "task_type": "detection", "class_names": ["basketball"], "output_box_format": "end2end_xyxy" }}] }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_seg", "dst": "v_post_seg", "group": "in", "name": "Yolo_Seg", "input_format": "RGB", "conf_thresh": 0.25, "max_det": 10, "infer_every_n": 1, "metadata_key_detection": "yolo_seg_det", "metadata_key_segmentation": "yolo_seg", "mask_gpu_every_n": 1, "mask_cpu_every_n": 1, "mask_cpu_resolution": 272, "models": [{{ "engine": "{m}/court-segmentation_960x544.plan", "task_type": "segmentation", "class_names": ["basketball-court", "three point line"], "output_box_format": "end2end_xyxy", "include_in_detection_metadata": false }}] }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_player_seg", "dst": "v_post_player_seg", "group": "in", "name": "Yolo_Player_Seg", "input_format": "RGB", "conf_thresh": 0.18, "max_det": 20, "infer_every_n": 1, "metadata_key_detection": "yolo_players_seg_det", "metadata_key_segmentation": "yolo_players_seg", "mask_gpu_every_n": 1, "mask_cpu_every_n": 1, "side_data_slot": 1, "models": [{{ "engine": "{m}/player-seg/player-seg_960x544.plan", "task_type": "segmentation", "class_names": ["player"], "output_box_format": "end2end_xyxy", "include_in_detection_metadata": false }}] }}

node.add {{ "type": "split", "src": "v_post_player_seg", "dst": ["v_post_player_seg_torso", "v_post_player_seg_main"], "group": "in" }}
node.add {{ "type": "player_torso_seg", "src": "v_post_player_seg_torso", "dst": "v_post_torso_seg", "group": "in", "name": "Player_Torso_Seg", "metadata_key": "yolo_players_seg", "output_metadata_key": "yolo_players_torso_seg", "target_labels": ["player"], "input_side_data_slot": 1, "output_side_data_slot": 2, "mask_threshold": 0.5, "torso_x_margin_rel": 0.10, "torso_y_start_rel": 0.16, "torso_y_end_rel": 0.60, "sample_inner_x_margin_rel": 0.18, "sample_top_y_exclusion_rel": 0.12, "skin_filter": true }}
node.add {{ "type": "jersey_color_extract", "src": "v_post_torso_seg", "dst": "v_post_torso_color", "group": "in", "name": "Torso_Color_Extract", "metadata_key": "yolo_players_torso_seg", "target_labels": ["torso"], "mask_threshold": 0.5, "min_pixels": 32, "body_region": "full", "side_data_slot": 2 }}

node.add {{ "type": "join_metadata", "src": ["v_post_players", "v_post_ball"], "dst": "v_players_ball", "group": "in" }}
node.add {{ "type": "join_metadata", "src": ["v_players_ball", "v_post_seg"], "dst": "v_players_ball_court", "group": "in" }}
node.add {{ "type": "join_metadata", "src": ["v_players_ball_court", "v_post_player_seg_main"], "dst": "v_inferred_nopose", "group": "in" }}
node.add {{ "type": "join_metadata", "src": ["v_inferred_nopose", "v_post_pose"], "dst": "v_inferred", "group": "in" }}

node.add {{ "type": "shot_classifier", "src": "v_inferred", "dst": "v_classified", "group": "in", "seg_metadata_key": "yolo_seg", "player_metadata_key": "yolo_players", "player_labels": ["Player"], "court_class_indices": [0], "wide_court_threshold": 0.25, "closeup_court_threshold": 0.05, "ambiguous_min_players": 3, "high_player_override": 5, "player_height_fraction": 0.25, "player_height_tolerance": 0.45, "player_min_aspect_ratio": 0.75, "min_stable_frames": 3 }}
node.add {{ "type": "player_tracker", "src": "v_classified", "dst": "v_tracked_players", "group": "in", "metadata_key": "yolo_players", "target_labels": ["Player"], "frame_rate": 25, "track_buffer": 90, "predict_on_empty": true, "track_thresh": 0.2, "high_thresh": 0.85, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "player_feet_seg", "src": "v_tracked_players", "dst": "v_tracked_players_feet", "group": "in", "name": "Player_Feet_Seg", "metadata_key": "yolo_players_seg", "player_metadata_key": "yolo_players", "output_metadata_key": "player_feet", "target_labels": ["player"], "input_side_data_slot": 1, "output_side_data_slot": 3, "mask_threshold": 0.5, "foot_y_start_rel": 0.70, "foot_x_margin_rel": 0.02, "min_pixels": 10 }}
node.add {{ "type": "ball_tracker", "src": "v_tracked_players_feet", "dst": "v_ball_tracked", "group": "in", "metadata_key": "yolo_ball", "target_label": "basketball", "coast": true, "min_conf": 0.04, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "ball_handler", "src": "v_ball_tracked", "dst": "v_tracked", "group": "in", "ball_metadata_key": "yolo_ball", "player_metadata_key": "yolo_players", "output_metadata_key": "ball_handler", "ball_label": "basketball", "player_labels": ["Player"], "max_distance_px": 35, "hysteresis_frames": 12, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "join_metadata", "src": ["v_tracked", "v_post_torso_color"], "dst": "v_team_inputs", "group": "in" }}
node.add {{ "type": "torso_team_classifier", "src": "v_team_inputs", "dst": "v_teams", "group": "in", "name": "Torso_Team_Classifier", "player_metadata_key": "yolo_players", "torso_metadata_key": "yolo_players_torso_seg", "player_seg_metadata_key": "yolo_players_seg", "output_player_metadata_key": "yolo_players", "camera_shot_metadata_key": "camera_shot_info", "player_labels": ["Player"], "torso_labels": ["torso"], "require_wide_shot": true, "iou_match_threshold": 0.10, "min_jersey_pixels": 32, "bootstrap_frames": 10, "rewrite_torso_cls": true, "write_back_to_player_seg": true, "tracker_fallback_enabled": true }}

node.add {{ "type": "join_metadata", "src": ["v_dec_1080p_main", "v_teams"], "dst": "v_1080p_md", "group": "in" }}

node.add {{ "type": "draw_segmask", "src": "v_calib", "dst": "v_court_seg_drawn", "group": "in", "name": "Draw_Court_Seg", "metadata_key": "yolo_seg", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "light_blue", "opacity": 0.65, "threshold": 0.5, "min_conf": 0.0, "class_colors": {{ "0": "light_blue", "1": "yellow" }}, "class_opacities": {{ "0": 1.0, "1": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 20, "overlay_fade_frames": 10, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_segmask", "src": "v_court_seg_drawn", "dst": "v_torso_seg_drawn", "group": "in", "name": "Draw_Torso_Teams", "metadata_key": "yolo_players_torso_seg", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "red", "opacity": 0.60, "threshold": 0.5, "min_conf": 0.0, "class_colors": {{ "-1": "red", "0": "light_blue", "1": "green" }}, "class_opacities": {{ "-1": 0.35, "0": 1.0, "1": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 5, "overlay_fade_frames": 3, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "side_data_slot": 2 }}
node.add {{ "type": "draw_segmask", "src": "v_torso_seg_drawn", "dst": "v_feet_seg_drawn", "group": "in", "name": "Draw_Feet_Seg", "metadata_key": "player_feet", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "red", "opacity": 1.0, "threshold": 0.5, "min_conf": 0.05, "class_opacities": {{ "0": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 5, "overlay_fade_frames": 3, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "side_data_slot": 3 }}
node.add {{ "type": "draw_keypoints", "src": "v_feet_seg_drawn", "dst": "v_court_proj_drawn", "group": "in", "name": "Draw_Court_Proj", "metadata_key": "court_proj", "color": "magenta", "radius": 3, "min_conf": 0.0, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "metadata_dump", "src": "v_court_proj_drawn", "dst": "v_dumped", "group": "in", "name": "Metadata_Dump", "output_metadata_key": "frame_dump", "video_label": "input", "fps": 25, "dump_every_n": 1 }}
node.add {{ "type": "draw_bbox", "src": "v_dumped", "dst": "v_bbox_players", "group": "in", "name": "Draw_Players", "metadata_key": "yolo_players", "bbox_thickness": 1, "min_conf": 0.25, "allowed_labels": ["Hoop", "Player"], "label_colors": {{ "Hoop": "yellow", "Player": "green" }}, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_tactical_court", "src": "v_bbox_players", "dst": "v_tactical", "group": "in", "name": "Draw_Tactical_Court", "metadata_key": "frame_dump", "court_seg_metadata_key": "yolo_seg", "pose_metadata_key": "yolo_pose", "court_seg_slot": 0, "require_wide_shot": true, "panel_width": 360, "panel_height": 220, "padding_left": 28, "padding_bottom": 116, "inner_padding": 14, "background_opacity": 0.58, "max_players_per_team": 5, "max_player_dots": 12, "show_unknown_players": true, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "debug_log_every_n": 250 }}

node.add {{ "type": "force_fps", "fps": "25/1", "group": "in", "src": "v_tactical", "dst": "v_out_fps" }}
node.add {{ "type": "assume_video_format", "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "group": "in", "src": "v_out_fps", "dst": "v_preenc" }}
node.add {{ "type": "enc_video", "src": "v_preenc", "dst": "v_outenc", "group": "in", "name": "Video_Encode_NVENC", "codec": "h264_nvenc", "hwaccel": "@gpu", "options": {{ "b": "12000k", "maxrate": "12000k", "bufsize": "12000k", "rc": "cbr", "g": 75, "bf": 0, "preset": "p5", "profile": "high" }} }}
node.add {{ "type": "mux", "src": ["v_outenc"], "dst": "mux_v", "group": "in", "ts_sort_wait": 0 }}
node.add {{ "type": "output", "format": "mpegts", "url": "{output_url}", "src": "mux_v", "group": "in", "auto_restart": "exit", "on_error": "panic" }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="tactical.ts")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--debug-every-n", type=int, default=125)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many wall-clock seconds (0 = run until killed)")
    args = ap.parse_args()

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString(graph_script(args.input, args.output, args.models_dir))

    calib = CourtCalibrationNode({
        "src": "v_1080p_md",
        "dst": "v_calib",
        "group": "in",
        "name": "Court_Calibration",
        # Visibility-first tuning: hold the last accepted homography across
        # gate dropouts (the wide camera moves slowly) and accept rougher
        # fits — players should be on the panel most of the time.
        "hold_frames": 10,
        "gate_median_ft": 3.0,
        "min_coverage": 0.25,
        "min_coverage_no_hoop": 0.35,
        "min_coverage_cold_no_hoop": 0.5,
        "hoop_gate_ft": 15.0,
        "sideline_gate_ft": 6.0,
        "min_region_area": 0.03,
        # Polish the template camera in PTZ space (pan/tilt/zoom point-to-arc
        # LM; removes the coarse-grid depth offset of the magenta arc dots),
        # clean the 3-pt region mask (kills court-into-arc spillover), and
        # reject degenerate baseline-only fits via arc angular coverage.
        "relative_refine": os.environ.get("AVP_RELATIVE_REFINE", "1") == "1",
        "clean_tpl_mask": True,
        # Reclaim 3-pt area lost to court-class spillover into the arc interior
        # (the first-half "blue floods the arc" blocker) using the prior
        # homography, before the region-area gate — so flooded frames still fit
        # instead of collapsing to region_small and riding a stale hold.
        # Env override lets an A/B harness flip just this flag.
        "reconcile_spillover": os.environ.get("AVP_RECONCILE_SPILLOVER", "1") == "1",
        # Reclaim only the deep interior; keep a band (ft) next to painted lines.
        "reconcile_margin_ft": float(os.environ.get("AVP_RECONCILE_MARGIN_FT", "2.0")),
        # Sector-consensus spillover rejection for the relative path: bin
        # boundary points along each segment under the template prior, drop
        # bins deviating from the segment consensus by more than this (ft).
        "sector_outlier_ft": float(os.environ.get("AVP_SECTOR_OUTLIER_FT", "2.0")),
        "min_arc_coverage_deg": 40.0,
        "calib_log": os.environ.get("AVP_CALIB_LOG", ""),
        "debug_log_every_n": args.debug_every_n,
    })
    avp.addNode(calib)

    avp.executeCommandsFromString("group.start in")

    started = time.time()
    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
            if args.max_seconds > 0 and time.time() - started > args.max_seconds:
                print("tactical_view: max seconds reached, exiting")
                break
    finally:
        sys.stdout.flush()
        os._exit(0)  # node threads are non-daemon; exit hard


if __name__ == "__main__":
    main()
