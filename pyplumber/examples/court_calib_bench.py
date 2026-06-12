#!/usr/bin/env python3
"""Tactical-view data-path benchmark.

Runs the metadata path needed for a tactical view: player, ball, court
segmentation, player segmentation, torso/feet/team, and Python/CuPy court
calibration. Pose, draw nodes, metadata dump, and encoder stay out of this
benchmark.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pyplumber
from pyplumber.court_calibration import CourtCalibrationNode
from pyplumber.node import (
    BallHandler,
    BallTracker,
    CudaInferYolo,
    DecVideo,
    Demux,
    FilterVideo,
    ForceFPS,
    InputRec,
    InternalNode,
    JoinMetadata,
    NullSink,
    PlayerTracker,
    PythonNode,
    ShotClassifier,
    Split,
)


DEFAULT_MODELS_DIR = os.environ.get("AVP_MODELS_DIR", "/models")


class NativeNode(InternalNode):
    TYPE = ""

    def __init__(self, node_type, args):
        self.TYPE = node_type
        super().__init__(args)


def _ts(seconds):
    ms_total = int(round(seconds * 1000.0))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _infer_every_n(env_name, default="1"):
    return max(1, int(os.environ.get(env_name, default)))


def build_graph(avp, input_url, models_dir, fps, start_seconds, seconds,
                debug_every_n, stage):
    start_ts = _ts(start_seconds)
    stop_ts = _ts(start_seconds + seconds)
    en_players = _infer_every_n("AVP_EVERY_N_PLAYERS")
    en_ball = _infer_every_n("AVP_EVERY_N_BALL")
    en_seg = _infer_every_n("AVP_EVERY_N_SEG")
    en_pseg = _infer_every_n("AVP_EVERY_N_PLAYER_SEG", "2")
    calib_every_n = max(1, int(os.environ.get("AVP_CALIB_EVERY_N", "3")))
    court_cpu_mask_every_n = max(0, int(os.environ.get(
        "AVP_COURT_CPU_MASK_EVERY_N", str(calib_every_n))))
    yolo_scale_graph = os.environ.get(
        "AVP_YOLO_SCALE_GRAPH", "scale_cuda=w=960:h=544")
    yolo_scale_w = int(os.environ.get("AVP_YOLO_SCALE_W", "960"))
    yolo_scale_h = int(os.environ.get("AVP_YOLO_SCALE_H", "544"))
    yolo_preprocess_resize = os.environ.get(
        "AVP_YOLO_PREPROCESS_RESIZE", "1") == "1"
    debug_yolo_metadata = os.environ.get("AVP_DEBUG_YOLO_METADATA", "0") == "1"
    debug_trt_init = os.environ.get("AVP_DEBUG_TRT_INIT", "0") == "1"
    avp.edges.planCapacity("*", 14)
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')

    nodes = [
        InputRec({
            "url": input_url,
            "dst": "in_mux0",
            "group": "in",
            "name": "input",
            "initial_timeout": 20,
            "timeout": 10,
            "loop": False,
            "start_ts": start_ts,
            "stop_ts": stop_ts,
            "on_error": "panic",
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": "in",
            "name": "Demux",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "in",
            "name": "Video_Dec",
            "optional": True,
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }),
        ForceFPS({
            "fps": f"{fps}/1",
            "group": "in",
            "src": "v_dec_cuda",
            "dst": "v_dec_fps",
            "name": "Force_FPS",
        }),
    ]
    if stage == "decode":
        for node in nodes:
            avp.addNode(node)
        return "v_dec_fps"

    if stage == "scale":
        scale_src = "v_dec_fps"
    else:
        nodes.append(Split({
                "src": "v_dec_fps",
                "dst": ["v_dec_1080p_main", "v_dec_for_yolo"],
                "group": "in",
                "name": "Split_Fullres_Yolo",
            }))
        scale_src = "v_dec_for_yolo"

    yolo_head_src = scale_src
    if stage == "scale" or not yolo_preprocess_resize:
        nodes.extend([
            FilterVideo({
                "graph": yolo_scale_graph,
                "src": scale_src,
                "dst": "v_pre_yolo",
                "group": "in",
                "name": "Scale_Yolo",
                "dst_width": yolo_scale_w,
                "dst_height": yolo_scale_h,
                "dst_pixel_format": "cuda",
                "hwaccel": "@gpu",
                "defer_preliminary_init": True,
            }),
        ])
        yolo_head_src = "v_pre_yolo"
    if stage == "scale":
        for node in nodes:
            avp.addNode(node)
        return "v_pre_yolo"

    nodes.extend([
        Split({
            "src": yolo_head_src,
            "dst": [
                "v_for_players",
                "v_for_ball",
                "v_for_seg",
                "v_for_player_seg",
            ],
            "group": "in",
            "name": "Split_Yolo_Heads",
        }),
        CudaInferYolo({
            "src": "v_for_players",
            "dst": "v_post_players",
            "group": "in",
            "name": "Yolo_Players",
            "input_format": "RGB",
            "conf_thresh": 0.25,
            "max_det": 40,
            "infer_every_n": en_players,
            "metadata_key_detection": "yolo_players",
            "debug_log_metadata": debug_yolo_metadata,
            "debug_log_every_n": debug_every_n,
            "debug_init_timing": debug_trt_init,
            "allow_input_resize": yolo_preprocess_resize,
            "mask_gpu_every_n": 0,
            "mask_cpu_every_n": 0,
            "models": [{
                "engine": f"{models_dir}/basketball-players-full_960x544.plan",
                "task_type": "detection",
                "class_names": [
                    "_suppress", "Hoop", "Period", "Player", "Ref",
                    "Shot Clock", "Team Name", "Team Points",
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
            "input_format": "RGB",
            "conf_thresh": 0.04,
            "max_det": 10,
            "infer_every_n": en_ball,
            "metadata_key_detection": "yolo_ball",
            "debug_log_metadata": debug_yolo_metadata,
            "debug_log_every_n": debug_every_n,
            "debug_init_timing": debug_trt_init,
            "allow_input_resize": yolo_preprocess_resize,
            "mask_gpu_every_n": 0,
            "mask_cpu_every_n": 0,
            "models": [{
                "engine": f"{models_dir}/ball_960x544.plan",
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
            "input_format": "RGB",
            "conf_thresh": 0.25,
            "max_det": 10,
            "infer_every_n": en_seg,
            "metadata_key_detection": "yolo_seg_det",
            "metadata_key_segmentation": "yolo_seg",
            "debug_log_metadata": debug_yolo_metadata,
            "debug_log_every_n": debug_every_n,
            "debug_init_timing": debug_trt_init,
            "allow_input_resize": yolo_preprocess_resize,
            "mask_gpu_every_n": 1,
            "mask_cpu_every_n": int(os.environ.get("AVP_COURT_YOLO_MASK_CPU_EVERY_N", "0")),
            "mask_cpu_resolution": 272,
            "models": [{
                "engine": f"{models_dir}/court-segmentation_960x544.plan",
                "task_type": "segmentation",
                "class_names": ["basketball-court", "three point line"],
                "output_box_format": "end2end_xyxy",
                "include_in_detection_metadata": False,
            }],
        }),
        NativeNode("court_seg_evidence_cuda", {
            "src": "v_post_seg",
            "dst": "v_post_seg_evidence",
            "group": "in",
            "name": "Court_Seg_Evidence_Cuda",
            "side_data_slot": 0,
            "output_side_data_slot": 0,
            "mask_resolution": 272,
            "mask_threshold": 0.5,
            "emit_cpu_every_n": court_cpu_mask_every_n,
            "output_metadata_key": "court_seg_evidence",
            "debug_log_every_n": debug_every_n,
        }),
        CudaInferYolo({
            "src": "v_for_player_seg",
            "dst": "v_post_player_seg",
            "group": "in",
            "name": "Yolo_Player_Seg",
            "input_format": "RGB",
            "conf_thresh": 0.18,
            "max_det": 20,
            "infer_every_n": en_pseg,
            "metadata_key_detection": "yolo_players_seg_det",
            "metadata_key_segmentation": "yolo_players_seg",
            "debug_log_metadata": debug_yolo_metadata,
            "debug_log_every_n": debug_every_n,
            "debug_init_timing": debug_trt_init,
            "allow_input_resize": yolo_preprocess_resize,
            "mask_gpu_every_n": 1,
            "mask_cpu_every_n": int(os.environ.get("AVP_PLAYER_SEG_MASK_CPU_EVERY_N", "0")),
            "side_data_slot": 1,
            "models": [{
                "engine": f"{models_dir}/player-seg/player-seg_960x544.plan",
                "task_type": "segmentation",
                "class_names": ["player"],
                "output_box_format": "end2end_xyxy",
                "include_in_detection_metadata": False,
            }],
        }),
        Split({
            "src": "v_post_player_seg",
            "dst": ["v_post_player_seg_torso", "v_post_player_seg_main"],
            "group": "in",
            "name": "Split_Player_Seg",
        }),
        NativeNode("player_torso_seg", {
            "src": "v_post_player_seg_torso",
            "dst": "v_post_torso_seg",
            "group": "in",
            "name": "Player_Torso_Seg",
            "metadata_key": "yolo_players_seg",
            "output_metadata_key": "yolo_players_torso_seg",
            "target_labels": ["player"],
            "input_side_data_slot": 1,
            "output_side_data_slot": 2,
            "mask_threshold": 0.5,
            "torso_x_margin_rel": 0.10,
            "torso_y_start_rel": 0.16,
            "torso_y_end_rel": 0.60,
            "sample_inner_x_margin_rel": 0.18,
            "sample_top_y_exclusion_rel": 0.12,
            "skin_filter": True,
            "skin_neutral_y_min": 0,
            "skin_neutral_u_tol": 18,
            "skin_neutral_v_tol": 18,
            "debug_log_every_n": debug_every_n,
        }),
        NativeNode("jersey_color_extract", {
            "src": "v_post_torso_seg",
            "dst": "v_post_torso_color",
            "group": "in",
            "name": "Torso_Color_Extract",
            "metadata_key": "yolo_players_torso_seg",
            "target_labels": ["torso"],
            "mask_threshold": 0.5,
            "min_pixels": 32,
            "body_region": "full",
            "side_data_slot": 2,
            "debug_log_every_n": debug_every_n,
        }),
        JoinMetadata({
            "src": ["v_post_players", "v_post_ball"],
            "dst": "v_players_ball",
            "group": "in",
            "name": "Join_Players_Ball",
        }),
        JoinMetadata({
            "src": ["v_players_ball", "v_post_seg_evidence"],
            "dst": "v_players_ball_court",
            "group": "in",
            "name": "Join_Court_Seg",
        }),
        JoinMetadata({
            "src": ["v_players_ball_court", "v_post_player_seg_main"],
            "dst": "v_inferred",
            "group": "in",
            "name": "Join_Player_Seg",
        }),
        ShotClassifier({
            "src": "v_inferred",
            "dst": "v_classified",
            "group": "in",
            "name": "Shot_Classifier",
        "seg_metadata_key": "yolo_seg",
        "seg_evidence_metadata_key": "court_seg_evidence",
        "player_metadata_key": "yolo_players",
        "player_labels": ["Player"],
        "court_class_indices": [0],
        "mask_model_w": float(os.environ.get("AVP_MASK_MODEL_W", "960")),
        "mask_model_h": float(os.environ.get("AVP_MASK_MODEL_H", "544")),
        "mask_pad_x": float(os.environ.get("AVP_MASK_PAD_X", "0")),
        "mask_pad_y": float(os.environ.get("AVP_MASK_PAD_Y", "0")),
        "mask_content_w": float(os.environ.get("AVP_MASK_CONTENT_W", "960")),
        "mask_content_h": float(os.environ.get("AVP_MASK_CONTENT_H", "544")),
        "wide_court_threshold": 0.25,
        "closeup_court_threshold": 0.05,
            "ambiguous_min_players": 3,
            "high_player_override": 5,
            "player_height_fraction": 0.25,
            "player_height_tolerance": 0.45,
            "player_min_aspect_ratio": 0.75,
            "min_stable_frames": 3,
            "debug_log_every_n": debug_every_n,
        }),
        PlayerTracker({
            "src": "v_classified",
            "dst": "v_tracked_players",
            "group": "in",
            "name": "Player_Tracker",
            "metadata_key": "yolo_players",
            "target_labels": ["Player"],
            "frame_rate": fps,
            "track_buffer": 90,
            "predict_on_empty": True,
            "track_thresh": 0.2,
            "high_thresh": 0.85,
            "camera_shot_metadata_key": "camera_shot_info",
            "debug_log_every_n": debug_every_n,
        }),
        NativeNode("player_feet_seg", {
            "src": "v_tracked_players",
            "dst": "v_tracked_players_feet",
            "group": "in",
            "name": "Player_Feet_Seg",
            "metadata_key": "yolo_players_seg",
            "player_metadata_key": "yolo_players",
            "output_metadata_key": "player_feet",
            "target_labels": ["player"],
            "input_side_data_slot": 1,
            "output_side_data_slot": 3,
            "mask_threshold": 0.5,
            "foot_y_start_rel": 0.70,
            "foot_x_margin_rel": 0.02,
            "min_pixels": 10,
            "debug_log_every_n": debug_every_n,
        }),
        BallTracker({
            "src": "v_tracked_players_feet",
            "dst": "v_ball_tracked",
            "group": "in",
            "name": "Ball_Tracker",
            "metadata_key": "yolo_ball",
            "target_label": "basketball",
            "coast": True,
            "min_conf": 0.04,
            "camera_shot_metadata_key": "camera_shot_info",
            "coast_edge_jump_veto_enabled": True,
            "coast_edge_zone_rel": 0.14,
            "coast_edge_jump_rel": 0.18,
            "coast_edge_confirm_frames": 9,
            "debug_log_every_n": debug_every_n,
        }),
        BallHandler({
            "src": "v_ball_tracked",
            "dst": "v_tracked",
            "group": "in",
            "name": "Ball_Handler",
            "ball_metadata_key": "yolo_ball",
            "player_metadata_key": "yolo_players",
            "output_metadata_key": "ball_handler",
            "ball_label": "basketball",
            "player_labels": ["Player"],
            "max_distance_px": 35,
            "hysteresis_frames": 12,
            "camera_shot_metadata_key": "camera_shot_info",
            "debug_log_every_n": debug_every_n,
        }),
        JoinMetadata({
            "src": ["v_tracked", "v_post_torso_color"],
            "dst": "v_team_inputs",
            "group": "in",
            "name": "Join_Team_Inputs",
        }),
        NativeNode("torso_team_classifier", {
            "src": "v_team_inputs",
            "dst": "v_teams",
            "group": "in",
            "name": "Torso_Team_Classifier",
            "player_metadata_key": "yolo_players",
            "torso_metadata_key": "yolo_players_torso_seg",
            "player_seg_metadata_key": "yolo_players_seg",
            "output_player_metadata_key": "yolo_players",
            "debug_metadata_key": "team_classifier_debug",
            "camera_shot_metadata_key": "camera_shot_info",
            "player_labels": ["Player"],
            "torso_labels": ["torso"],
            "require_wide_shot": True,
            "require_player_match_for_training": True,
            "require_player_match_for_assignment": True,
            "iou_match_threshold": 0.10,
            "fallback_center_distance_px": 60.0,
            "min_jersey_pixels": 32,
            "min_jersey_confidence": 0.05,
            "bootstrap_frames": 10,
            "bootstrap_min_samples": 12,
            "bootstrap_min_cluster_size": 3,
            "bootstrap_min_prototype_distance": 0.06,
            "assignment_margin": 0.03,
            "prototype_update_margin": 0.06,
            "uv_weight": 1.0,
            "l_weight": 1.2,
            "ema_alpha_centroid": 0.02,
            "rewrite_torso_cls": True,
            "write_back_to_player_seg": True,
            "rewrite_player_seg_cls": False,
            "tracker_fallback_enabled": True,
            "tracker_fallback_max_age_frames": 3,
            "tracker_fallback_min_margin": 0.06,
            "tracker_fallback_min_confidence": 0.02,
            "debug_log_every_n": debug_every_n,
        }),
        JoinMetadata({
            "src": ["v_dec_1080p_main", "v_teams"],
            "dst": "v_1080p_md",
            "group": "in",
            "name": "Join_Fullres_Metadata",
        }),
    ])

    for node in nodes:
        avp.addNode(node)
    return "v_1080p_md"


class FrameCounterNode(PythonNode):
    def __init__(self, args):
        super().__init__(args)
        self.frames = 0
        self.valid_frames = 0
        self.first_frame_time = None
        self.last_frame_time = None
        self.first_valid_time = None
        self.metadata_key = args.get("metadata_key", "")

    def process(self):
        frame = self._src.get()
        if not frame:
            return
        now = time.perf_counter()
        if self.first_frame_time is None:
            self.first_frame_time = now
        self.last_frame_time = now
        self.frames += 1
        if self.metadata_key:
            try:
                md = json.loads(frame.metadata[self.metadata_key])
                if md.get("valid"):
                    self.valid_frames += 1
                    if self.first_valid_time is None:
                        self.first_valid_time = now
            except Exception:
                pass
        self._dst.enqueue(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.mp4")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--start-seconds", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-wall-seconds", type=float, default=60.0)
    ap.add_argument("--debug-every-n", type=int, default=0)
    ap.add_argument("--stage", choices=["decode", "scale", "full"],
                    default="full")
    args = ap.parse_args()

    avp = pyplumber.AVPlumber()
    graph_dst = build_graph(avp, args.input, args.models_dir, args.fps,
                            args.start_seconds, args.seconds,
                            args.debug_every_n, args.stage)

    if args.stage != "full":
        counter = FrameCounterNode({
            "src": graph_dst,
            "dst": "v_counted",
            "group": "in",
            "name": "Bench_Frame_Counter",
            "metadata_key": "",
        })
        avp.addNode(counter)
        avp.addNode(NullSink({
            "src": "v_counted",
            "group": "in",
            "name": "Bench_Sink",
        }))

        sink = avp.node("Bench_Sink")
        expected = int(round(args.seconds * args.fps))
        started = time.perf_counter()
        avp.group("in").startNodes()

        timed_out = False
        ever_working = False
        next_heartbeat = started + 1.0
        try:
            while True:
                time.sleep(0.05)
                now = time.perf_counter()
                if now >= next_heartbeat:
                    avp.heartbeat()
                    next_heartbeat = now + 1.0
                if sink.isWorking:
                    ever_working = True
                if counter.frames >= expected:
                    break
                if ever_working and not sink.isWorking:
                    break
                if now - started > args.max_wall_seconds:
                    timed_out = True
                    break
        finally:
            elapsed = time.perf_counter() - started
            frames = counter.frames
            fps = frames / elapsed if elapsed > 0 else 0.0
            if frames > 1 and counter.first_frame_time is not None:
                output_elapsed = counter.last_frame_time - counter.first_frame_time
                output_fps = (frames - 1) / output_elapsed if output_elapsed > 0 else 0.0
                first_output_delay = counter.first_frame_time - started
            else:
                output_elapsed = 0.0
                output_fps = 0.0
                first_output_delay = float("nan")
            print("court_calib_bench: "
                  f"stage={args.stage} "
                  f"frames={frames} expected={expected} "
                  f"elapsed={elapsed:.3f}s fps={fps:.2f} "
                  f"first_output_delay={first_output_delay:.3f}s "
                  f"output_elapsed={output_elapsed:.3f}s "
                  f"output_fps={output_fps:.2f} "
                  f"timed_out={str(timed_out).lower()}",
                  flush=True)
            sys.stdout.flush()
            os._exit(1 if timed_out else 0)

    avp.addNode(Split({
        "src": "v_1080p_md",
        "dst": ["v_1080p_md_calib", "v_1080p_md_count"],
        "group": "in",
        "name": "Split_Pre_Calib_Count",
    }))
    pre_counter = FrameCounterNode({
        "src": "v_1080p_md_count",
        "dst": "v_pre_calib_counted",
        "group": "in",
        "name": "Bench_Pre_Calib_Counter",
        "metadata_key": "",
    })
    avp.addNode(pre_counter)
    avp.addNode(NullSink({
        "src": "v_pre_calib_counted",
        "group": "in",
        "name": "Bench_Pre_Calib_Sink",
    }))

    calib_every_n = int(os.environ.get("AVP_CALIB_EVERY_N", "3"))
    calib = CourtCalibrationNode({
        "src": "v_1080p_md_calib",
        "dst": "v_calib",
        "group": "in",
        "name": "Court_Calibration",
        "require_wide_shot": False,
        "relative_refine": os.environ.get("AVP_RELATIVE_REFINE", "1") == "1",
        "clean_tpl_mask": True,
        "hold_frames": 10,
        "gate_median_ft": 3.0,
        "min_coverage": 0.25,
        "hoop_gate_ft": 15.0,
        "sideline_gate_ft": 6.0,
        "min_region_area": 0.03,
        "min_arc_coverage_deg": 40.0,
        "calib_log": os.environ.get("AVP_CALIB_LOG", ""),
        "debug_log_every_n": args.debug_every_n,
        "calibrate_every_n": calib_every_n,
        "mask_model_w": float(os.environ.get("AVP_MASK_MODEL_W", "960")),
        "mask_model_h": float(os.environ.get("AVP_MASK_MODEL_H", "544")),
        "mask_pad_x": float(os.environ.get("AVP_MASK_PAD_X", "0")),
        "mask_pad_y": float(os.environ.get("AVP_MASK_PAD_Y", "0")),
        "mask_content_w": float(os.environ.get("AVP_MASK_CONTENT_W", "960")),
        "mask_content_h": float(os.environ.get("AVP_MASK_CONTENT_H", "544")),
    })
    counter = FrameCounterNode({
        "src": "v_calib",
        "dst": "v_counted",
        "group": "in",
        "name": "Bench_Frame_Counter",
        "metadata_key": "court_calib",
    })
    avp.addNode(calib)
    avp.addNode(counter)
    avp.addNode(NullSink({
        "src": "v_counted",
        "group": "in",
        "name": "Bench_Sink",
    }))

    sink = avp.node("Bench_Sink")
    expected = int(round(args.seconds * args.fps))
    started = time.perf_counter()
    avp.group("in").startNodes()

    timed_out = False
    ever_working = False
    next_heartbeat = started + 1.0
    try:
        while True:
            time.sleep(0.05)
            now = time.perf_counter()
            if now >= next_heartbeat:
                avp.heartbeat()
                next_heartbeat = now + 1.0
            if sink.isWorking:
                ever_working = True
            if counter.frames >= expected:
                break
            if ever_working and not sink.isWorking:
                break
            if now - started > args.max_wall_seconds:
                timed_out = True
                break
    finally:
        elapsed = time.perf_counter() - started
        frames = counter.frames
        fps = frames / elapsed if elapsed > 0 else 0.0
        if frames > 1 and counter.first_frame_time is not None:
            output_elapsed = counter.last_frame_time - counter.first_frame_time
            output_fps = (frames - 1) / output_elapsed if output_elapsed > 0 else 0.0
            first_output_delay = counter.first_frame_time - started
        else:
            output_elapsed = 0.0
            output_fps = 0.0
            first_output_delay = float("nan")
        if pre_counter.frames > 1 and pre_counter.first_frame_time is not None:
            pre_elapsed = pre_counter.last_frame_time - pre_counter.first_frame_time
            pre_fps = (pre_counter.frames - 1) / pre_elapsed if pre_elapsed > 0 else 0.0
            first_pre_delay = pre_counter.first_frame_time - started
        else:
            pre_fps = 0.0
            first_pre_delay = float("nan")
        first_valid_delay = counter.first_valid_time - started \
            if counter.first_valid_time is not None else float("nan")
        print("court_calib_bench: "
              f"frames={frames} expected={expected} "
              f"elapsed={elapsed:.3f}s fps={fps:.2f} "
              f"pre_frames={pre_counter.frames} "
              f"first_pre_delay={first_pre_delay:.3f}s "
              f"pre_output_fps={pre_fps:.2f} "
              f"first_output_delay={first_output_delay:.3f}s "
              f"valid_frames={counter.valid_frames} "
              f"first_valid_delay={first_valid_delay:.3f}s "
              f"output_elapsed={output_elapsed:.3f}s "
              f"output_fps={output_fps:.2f} "
              f"timed_out={str(timed_out).lower()}",
              flush=True)
        sys.stdout.flush()
        os._exit(1 if timed_out else 0)


if __name__ == "__main__":
    main()
