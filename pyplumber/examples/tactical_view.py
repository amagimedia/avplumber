#!/usr/bin/env python3
"""Lean basketball tactical-view pipeline.

Players + ball + court/player segmentation + tracker + feet + team
classification, segmentation-driven court calibration (court_calibration.py,
no pose model) and the draw_tactical_court overlay. No metadata dumping,
scoreboard, or game-state nodes.

Run (inside the CUDA/TensorRT runtime container):
  python3 pyplumber/examples/tactical_view.py \
      --input /media/input.mp4 --output /artifacts/tactical.ts
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pyplumber
from pyplumber.court_calibration import CourtCalibrationNode

MODELS_DIR = os.environ.get("AVP_MODELS_DIR", "/models")


def _ts(seconds):
    ms_total = int(round(seconds * 1000.0))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def graph_script(input_url, output_url, models, debug_every_n,
                 start_seconds=0.0, seconds=0.0):
    m = models
    # Per-head inference decimation (the five 960x544 TRT engines are the
    # throughput wall on a T4): skipped frames pass through without fresh
    # metadata and the trackers/holds downstream coast across them.
    en_players = int(os.environ.get("AVP_EVERY_N_PLAYERS", "1"))
    en_ball = int(os.environ.get("AVP_EVERY_N_BALL", "1"))
    en_seg = int(os.environ.get("AVP_EVERY_N_SEG", "1"))
    en_pseg = int(os.environ.get("AVP_EVERY_N_PLAYER_SEG", "2"))
    calib_every_n = int(os.environ.get("AVP_CALIB_EVERY_N", "1"))
    court_cpu_mask_every_n = int(os.environ.get(
        "AVP_COURT_CPU_MASK_EVERY_N", str(calib_every_n)))
    emit_luma = "true" if os.environ.get("AVP_EMIT_LUMA", "1") == "1" else "false"
    dot_log = os.environ.get("AVP_DOT_LOG", "")
    dot_log_param = f', "dot_log": "{dot_log}"' if dot_log else ""
    input_time_params = ""
    if start_seconds > 0.0:
        input_time_params += f', "start_ts": "{_ts(start_seconds)}"'
    if seconds > 0.0:
        input_time_params += f', "stop_ts": "{_ts(start_seconds + seconds)}"'
    return f"""
queue.plan_capacity * 14

hwaccel.init {{ "name": "@gpu", "type": "cuda" }}

node.add {{ "type": "input_rec", "url": "{input_url}", "dst": "in_mux0", "group": "in", "name": "input", "initial_timeout": 20, "timeout": 10, "loop": false, "on_error": "panic"{input_time_params} }}
node.add {{ "type": "demux", "src": "in_mux0", "wait_for_keyframe": false, "routing": {{ "?v:0": "v_pkt" }}, "group": "in" }}
node.add {{ "type": "dec_video", "src": "v_pkt", "dst": "v_dec_cuda", "group": "in", "name": "Video_Dec", "optional": true, "pixel_format": "?cuda", "hwaccel": "@gpu", "codec_map": {{ "h264": "h264_cuvid", "hevc": "hevc_cuvid" }}, "hwaccel_only_for_codecs": ["h264", "hevc"] }}
node.add {{ "type": "force_fps", "fps": "25/1", "group": "in", "src": "v_dec_cuda", "dst": "v_dec_25fps" }}

node.add {{ "type": "split", "src": "v_dec_25fps", "dst": ["v_dec_1080p_main", "v_dec_for_yolo"], "group": "in" }}
node.add {{ "type": "filter_video", "graph": "scale_cuda=w=960:h=540,pad_cuda=960:544:0:2", "src": "v_dec_for_yolo", "dst": "v_pre_yolo", "group": "in", "name": "Scale_Yolo", "dst_width": 960, "dst_height": 544, "dst_pixel_format": "cuda", "hwaccel": "@gpu" }}
node.add {{ "type": "split", "src": "v_pre_yolo", "dst": ["v_for_players", "v_for_ball", "v_for_seg", "v_for_player_seg"], "group": "in" }}

node.add {{ "type": "cuda_infer_yolo", "src": "v_for_players", "dst": "v_post_players", "group": "in", "name": "Yolo_Players", "input_format": "RGB", "conf_thresh": 0.25, "max_det": 40, "infer_every_n": {en_players}, "metadata_key_detection": "yolo_players", "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{{ "engine": "{m}/basketball-players-full_960x544.plan", "task_type": "detection", "class_names": ["_suppress", "Hoop", "_suppress", "Player", "Ref", "_suppress", "_suppress", "_suppress", "_suppress"], "output_box_format": "end2end_xyxy" }}] }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_ball", "dst": "v_post_ball", "group": "in", "name": "Yolo_Ball", "input_format": "RGB", "conf_thresh": 0.04, "max_det": 10, "infer_every_n": {en_ball}, "metadata_key_detection": "yolo_ball", "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{{ "engine": "{m}/ball_960x544.plan", "task_type": "detection", "class_names": ["basketball"], "output_box_format": "end2end_xyxy" }}] }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_seg", "dst": "v_post_seg", "group": "in", "name": "Yolo_Seg", "input_format": "RGB", "conf_thresh": 0.25, "max_det": 10, "infer_every_n": {en_seg}, "metadata_key_detection": "yolo_seg_det", "metadata_key_segmentation": "yolo_seg", "mask_gpu_every_n": 1, "mask_cpu_every_n": 0, "mask_cpu_resolution": 272, "models": [{{ "engine": "{m}/court-segmentation_960x544.plan", "task_type": "segmentation", "class_names": ["basketball-court", "three point line"], "output_box_format": "end2end_xyxy", "include_in_detection_metadata": false }}] }}
node.add {{ "type": "court_seg_evidence_cuda", "src": "v_post_seg", "dst": "v_post_seg_evidence", "group": "in", "name": "Court_Seg_Evidence_Cuda", "side_data_slot": 0, "output_side_data_slot": 0, "mask_resolution": 272, "mask_threshold": 0.5, "emit_cpu_every_n": {court_cpu_mask_every_n}, "emit_luma": {emit_luma}, "output_metadata_key": "court_seg_evidence" }}
node.add {{ "type": "cuda_infer_yolo", "src": "v_for_player_seg", "dst": "v_post_player_seg", "group": "in", "name": "Yolo_Player_Seg", "input_format": "RGB", "conf_thresh": 0.18, "max_det": 20, "infer_every_n": {en_pseg}, "metadata_key_detection": "yolo_players_seg_det", "metadata_key_segmentation": "yolo_players_seg", "mask_gpu_every_n": 1, "mask_cpu_every_n": 0, "side_data_slot": 1, "models": [{{ "engine": "{m}/player-seg/player-seg_960x544.plan", "task_type": "segmentation", "class_names": ["player"], "output_box_format": "end2end_xyxy", "include_in_detection_metadata": false }}] }}

node.add {{ "type": "split", "src": "v_post_player_seg", "dst": ["v_post_player_seg_torso", "v_post_player_seg_main"], "group": "in" }}
node.add {{ "type": "player_torso_seg", "src": "v_post_player_seg_torso", "dst": "v_post_torso_seg", "group": "in", "name": "Player_Torso_Seg", "metadata_key": "yolo_players_seg", "output_metadata_key": "yolo_players_torso_seg", "target_labels": ["player"], "input_side_data_slot": 1, "output_side_data_slot": 2, "mask_threshold": 0.5, "torso_x_margin_rel": 0.10, "torso_y_start_rel": 0.16, "torso_y_end_rel": 0.60, "sample_inner_x_margin_rel": 0.18, "sample_top_y_exclusion_rel": 0.12, "skin_filter": true }}
node.add {{ "type": "jersey_color_extract", "src": "v_post_torso_seg", "dst": "v_post_torso_color", "group": "in", "name": "Torso_Color_Extract", "metadata_key": "yolo_players_torso_seg", "target_labels": ["torso"], "mask_threshold": 0.5, "min_pixels": 32, "body_region": "full", "side_data_slot": 2 }}

node.add {{ "type": "join_metadata", "src": ["v_post_players", "v_post_ball"], "dst": "v_players_ball", "group": "in" }}
node.add {{ "type": "join_metadata", "src": ["v_players_ball", "v_post_seg_evidence"], "dst": "v_players_ball_court", "group": "in" }}
node.add {{ "type": "join_metadata", "src": ["v_players_ball_court", "v_post_player_seg_main"], "dst": "v_inferred", "group": "in" }}

node.add {{ "type": "shot_classifier", "src": "v_inferred", "dst": "v_classified", "group": "in", "seg_metadata_key": "yolo_seg", "seg_evidence_metadata_key": "court_seg_evidence", "player_metadata_key": "yolo_players", "player_labels": ["Player"], "court_class_indices": [0], "wide_court_threshold": 0.25, "closeup_court_threshold": 0.05, "ambiguous_min_players": 3, "high_player_override": 5, "player_height_fraction": 0.25, "player_height_tolerance": 0.45, "player_min_aspect_ratio": 0.75, "min_stable_frames": 3, "debug_log_every_n": {debug_every_n} }}
node.add {{ "type": "player_tracker", "src": "v_classified", "dst": "v_tracked_players", "group": "in", "metadata_key": "yolo_players", "target_labels": ["Player"], "frame_rate": 25, "track_buffer": 90, "predict_on_empty": true, "track_thresh": 0.2, "high_thresh": 0.85, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "player_feet_seg", "src": "v_tracked_players", "dst": "v_tracked_players_feet", "group": "in", "name": "Player_Feet_Seg", "metadata_key": "yolo_players_seg", "player_metadata_key": "yolo_players", "output_metadata_key": "player_feet", "target_labels": ["player"], "input_side_data_slot": 1, "output_side_data_slot": 3, "mask_threshold": 0.5, "foot_y_start_rel": 0.70, "foot_x_margin_rel": 0.02, "min_pixels": 10, "emit_unmatched_player_fallbacks": true, "debug_log_every_n": {debug_every_n} }}
node.add {{ "type": "ball_tracker", "src": "v_tracked_players_feet", "dst": "v_ball_tracked", "group": "in", "metadata_key": "yolo_ball", "target_label": "basketball", "coast": true, "min_conf": 0.04, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "ball_handler", "src": "v_ball_tracked", "dst": "v_tracked", "group": "in", "ball_metadata_key": "yolo_ball", "player_metadata_key": "yolo_players", "output_metadata_key": "ball_handler", "ball_label": "basketball", "player_labels": ["Player"], "max_distance_px": 35, "hysteresis_frames": 12, "camera_shot_metadata_key": "camera_shot_info" }}
node.add {{ "type": "join_metadata", "src": ["v_tracked", "v_post_torso_color"], "dst": "v_team_inputs", "group": "in" }}
node.add {{ "type": "torso_team_classifier", "src": "v_team_inputs", "dst": "v_teams", "group": "in", "name": "Torso_Team_Classifier", "player_metadata_key": "yolo_players", "torso_metadata_key": "yolo_players_torso_seg", "player_seg_metadata_key": "yolo_players_seg", "output_player_metadata_key": "yolo_players", "camera_shot_metadata_key": "camera_shot_info", "player_labels": ["Player"], "torso_labels": ["torso"], "require_wide_shot": true, "iou_match_threshold": 0.10, "min_jersey_pixels": 32, "bootstrap_frames": 10, "rewrite_torso_cls": true, "write_back_to_player_seg": true, "tracker_fallback_enabled": true }}

node.add {{ "type": "join_metadata", "src": ["v_dec_1080p_main", "v_teams"], "dst": "v_1080p_md", "group": "in" }}

node.add {{ "type": "draw_segmask", "src": "v_calib", "dst": "v_court_seg_drawn", "group": "in", "name": "Draw_Court_Seg", "metadata_key": "yolo_seg", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "light_blue", "opacity": 0.30, "threshold": 0.5, "min_conf": 0.0, "class_colors": {{ "0": "light_blue", "1": "yellow" }}, "class_opacities": {{ "0": 1.0, "1": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 20, "overlay_fade_frames": 10, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_segmask", "src": "v_court_seg_drawn", "dst": "v_torso_seg_drawn", "group": "in", "name": "Draw_Torso_Teams", "metadata_key": "yolo_players_torso_seg", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "red", "opacity": 0.60, "threshold": 0.5, "min_conf": 0.0, "class_colors": {{ "-1": "red", "0": "light_blue", "1": "green" }}, "class_opacities": {{ "-1": 0.35, "0": 1.0, "1": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 5, "overlay_fade_frames": 3, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "side_data_slot": 2 }}
node.add {{ "type": "draw_segmask", "src": "v_torso_seg_drawn", "dst": "v_feet_seg_drawn", "group": "in", "name": "Draw_Feet_Seg", "metadata_key": "player_feet", "camera_shot_metadata_key": "camera_shot_info", "mask_color": "red", "opacity": 1.0, "threshold": 0.5, "min_conf": 0.05, "class_opacities": {{ "0": 1.0 }}, "require_wide_shot": true, "overlay_hold_frames": 5, "overlay_fade_frames": 3, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "side_data_slot": 3 }}
node.add {{ "type": "draw_keypoints", "src": "v_feet_seg_drawn", "dst": "v_court_proj_drawn0", "group": "in", "name": "Draw_Court_Proj", "metadata_key": "court_proj", "color": "magenta", "radius": 3, "min_conf": 0.0, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_keypoints", "src": "v_court_proj_drawn0", "dst": "v_court_proj_drawn", "group": "in", "name": "Draw_Court_Evidence", "metadata_key": "court_evidence", "color": "red", "radius": 2, "min_conf": 0.0, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_bbox", "src": "v_court_proj_drawn", "dst": "v_bbox_players", "group": "in", "name": "Draw_Players", "metadata_key": "yolo_players", "bbox_thickness": 1, "min_conf": 0.25, "allowed_labels": ["Hoop", "Player"], "label_colors": {{ "Hoop": "yellow", "Player": "green" }}, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12" }}
node.add {{ "type": "draw_tactical_court", "src": "v_bbox_players", "dst": "v_tactical", "group": "in", "name": "Draw_Tactical_Court", "feet_metadata_key": "player_feet", "player_metadata_key": "yolo_players", "player_seg_metadata_key": "yolo_players_seg", "torso_metadata_key": "yolo_players_torso_seg", "team_a_color": "light_blue", "team_b_color": "green", "unknown_color": "red", "panel_width": 540, "panel_height": 330, "padding_left": 28, "padding_bottom": 116, "inner_padding": 21, "line_thickness": 3, "background_opacity": 0.58, "max_players_per_team": 5, "max_player_dots": 12, "show_unknown_players": true, "use_fallback_feet": true, "player_min_seen_frames": 1, "player_hold_frames": 20{dot_log_param}, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "debug_log_every_n": {debug_every_n} }}

node.add {{ "type": "force_fps", "fps": "25/1", "group": "in", "src": "v_tactical", "dst": "v_out_fps" }}
node.add {{ "type": "assume_video_format", "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "group": "in", "src": "v_out_fps", "dst": "v_preenc" }}
node.add {{ "type": "enc_video", "src": "v_preenc", "dst": "v_outenc", "group": "in", "name": "Video_Encode_NVENC", "codec": "h264_nvenc", "hwaccel": "@gpu", "options": {{ "b": "12000k", "maxrate": "12000k", "bufsize": "12000k", "rc": "cbr", "g": 75, "bf": 0, "preset": "p5", "profile": "high" }} }}
node.add {{ "type": "mux", "src": ["v_outenc"], "dst": "mux_v", "group": "in", "ts_sort_wait": 0 }}
node.add {{ "type": "output", "format": "mpegts", "url": "{output_url}", "src": "mux_v", "group": "in", "name": "Output", "auto_restart": "exit", "on_error": "panic" }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="tactical.ts")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--debug-every-n", type=int, default=0)
    ap.add_argument("--start-seconds", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="process this much input and exit after the output finishes (0 = unlimited)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many wall-clock seconds (0 = run until killed)")
    args = ap.parse_args()

    calib_every_n = int(os.environ.get("AVP_CALIB_EVERY_N", "1"))

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString(graph_script(
        args.input, args.output, args.models_dir, args.debug_every_n,
        args.start_seconds, args.seconds))

    calib = CourtCalibrationNode({
        "src": "v_1080p_md",
        "dst": "v_calib",
        "group": "in",
        "name": "Court_Calibration",
        "gate_median_ft": 3.0,
        "min_coverage": 0.25,
        "hoop_gate_ft": 15.0,
        "sideline_gate_ft": 6.0,
        "min_region_area": 0.03,
        # letterbox geometry: masks cover the 960x544 model frame whose
        # content is 960x540 at y=2 (scale_cuda + pad_cuda 0:2)
        "mask_model_w": 960.0,
        "mask_model_h": 544.0,
        "mask_pad_x": 0.0,
        "mask_pad_y": 2.0,
        "mask_content_w": 960.0,
        "mask_content_h": 540.0,
        # Polish the template camera in PTZ space (pan/tilt/zoom point-to-arc
        # LM; removes the coarse-grid depth offset of the magenta arc dots),
        # clean the 3-pt region mask (kills court-into-arc spillover), and
        # reject degenerate baseline-only fits via arc angular coverage.
        "relative_refine": os.environ.get("AVP_RELATIVE_REFINE", "1") == "1",
        "clean_tpl_mask": True,
        "tpl_max_source_area": float(os.environ.get("AVP_TPL_MAX_SOURCE_AREA", "0.0")),
        "min_arc_coverage_deg": 40.0,
        "court_proj_smooth_alpha": float(os.environ.get(
            "AVP_COURT_PROJ_SMOOTH_ALPHA", "0.45")),
        "court_proj_reset_px": float(os.environ.get(
            "AVP_COURT_PROJ_RESET_PX", "90.0")),
        "court_proj_max_gap_frames": int(os.environ.get(
            "AVP_COURT_PROJ_MAX_GAP_FRAMES", "4")),
        "calibrate_every_n": calib_every_n,
        "calib_log": os.environ.get("AVP_CALIB_LOG", ""),
        "debug_log_every_n": args.debug_every_n,
    })
    avp.addNode(calib)

    avp.executeCommandsFromString("group.start in")

    started = time.time()
    if args.max_seconds > 0:
        # hard exit even if the main loop wedges (observed drain stalls where
        # neither the output-finished check nor max_seconds ever fired);
        # os._exit bypasses any stuck interpreter state. Logs are written
        # continuously, so nothing is lost.
        def _watchdog():
            time.sleep(args.max_seconds + 30.0)
            print("tactical_view: watchdog hard-exit", flush=True)
            os._exit(2)
        threading.Thread(target=_watchdog, daemon=True).start()
    next_heartbeat = started + 1.0
    output_node = avp.node("Output") if args.seconds > 0.0 else None
    output_started = False
    try:
        while True:
            time.sleep(0.2)
            now = time.time()
            if now >= next_heartbeat:
                avp.heartbeat()
                next_heartbeat = now + 1.0
            if output_node is not None:
                if output_node.isWorking:
                    output_started = True
                elif output_started:
                    print("tactical_view: output finished, exiting")
                    break
            if args.max_seconds > 0 and now - started > args.max_seconds:
                print("tactical_view: max seconds reached, exiting")
                break
    finally:
        sys.stdout.flush()
        os._exit(0)  # node threads are non-daemon; exit hard


if __name__ == "__main__":
    main()
