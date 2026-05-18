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
        [--instance-name auto-mixer] \\
        [--janus-preview]

For WebRTC-only operation through a local Janus streaming mountpoint, omit
`--output` and pass `--janus-output`.

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
import threading
import time

from pyplumber import AVPlumber
from pyplumber.audio_vad import SileroVadRegistryBridge, Speaker, VisualSpeechRegistryNode
from pyplumber.auto_switcher import AutoSwitcher
from pyplumber.mixer import MixerGraphBuilder
from pyplumber.node import NullSink, Split

from .config import (
    CANVAS_H,
    CANVAS_W,
    FPS_DEN,
    FPS_NUM,
    HWACCEL,
    JANUS_DEFAULT_AUDIO_PORT,
    JANUS_DEFAULT_HOST,
    JANUS_DEFAULT_VIDEO_BITRATE_KBPS,
    JANUS_DEFAULT_VIDEO_PORT,
    MIN_ACTIVE_AUDIO_LEVEL_DBFS,
)
from .inputs import build_input_subgraph, default_face_engine, find_named_input, input_basename
from .outputs import (
    build_audio_output,
    build_janus_rtp_output,
    build_mux_output,
    build_video_output,
    output_format_for_url,
)
from .profiles.talkshow import (
    GENARO_INPUT_NAME,
    RENE_INPUT_NAME,
    RENE_REQUIRED_LEAD_DB,
    SERGIO_INPUT_NAME,
)
from .preheated import (
    PREHEATED_GROUP,
    build_preheated_scene_sources,
    define_preheated_scenes,
)
from .scenes import define_auto_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input URLs (at least 2)",
    )
    parser.add_argument(
        "--output",
        help="Output RTMP URL or file path. Optional when --janus-output or --janus-preview is set.",
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
        "--switch-pts-lead-ms", default=600, type=int,
        help=(
            "Schedule auto-switch cuts this many milliseconds ahead of the latest "
            "VAD event PTS; use a negative value to disable PTS scheduling "
            "(default: 600)"
        ),
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
    parser.add_argument(
        "--auto-switch-layout",
        choices=("videoconf", "full_face"),
        default="videoconf",
        help="Scene family used by the auto switcher (default: videoconf)",
    )
    parser.add_argument(
        "--disable-preheated-scenes", "--disable-preheated-speaker-scenes",
        dest="disable_preheated_scenes",
        action="store_true",
        help="Use dynamic mixer scene loading instead of geometry-preheated scene sources.",
    )
    parser.add_argument(
        "--debug-mouth-roi-bboxes", action="store_true",
        help="Draw mouth ROI boxes plus separate audio/video speaking labels into the output video.",
    )
    parser.add_argument(
        "--static-genaro-face-crop", action="store_true",
        help="Use a centered static 9:16 crop for the Genaro input instead of face tracking.",
    )
    parser.add_argument(
        "--janus-preview", action="store_true",
        help="Send a low-latency RTP copy to Janus in addition to any --output.",
    )
    parser.add_argument(
        "--janus-output", action="store_true",
        help="Send the program output to Janus RTP. May be used without --output.",
    )
    parser.add_argument(
        "--janus-host", default=JANUS_DEFAULT_HOST,
        help=f"Janus RTP ingest host (default: {JANUS_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--janus-video-port", default=JANUS_DEFAULT_VIDEO_PORT, type=int,
        help=f"Janus RTP H.264 video port (default: {JANUS_DEFAULT_VIDEO_PORT})",
    )
    parser.add_argument(
        "--janus-audio-port", default=JANUS_DEFAULT_AUDIO_PORT, type=int,
        help=f"Janus RTP Opus audio port (default: {JANUS_DEFAULT_AUDIO_PORT})",
    )
    parser.add_argument(
        "--janus-video-pt", default=96, type=int,
        help="RTP payload type for Janus H.264 video (default: 96)",
    )
    parser.add_argument(
        "--janus-audio-pt", default=111, type=int,
        help="RTP payload type for Janus Opus audio (default: 111)",
    )
    parser.add_argument(
        "--janus-video-codec", default="h264_nvenc",
        help="Video encoder codec for Janus RTP output (default: h264_nvenc)",
    )
    parser.add_argument(
        "--janus-video-bitrate-kbps", default=JANUS_DEFAULT_VIDEO_BITRATE_KBPS, type=int,
        help=f"Janus H.264 bitrate in kbit/s (default: {JANUS_DEFAULT_VIDEO_BITRATE_KBPS})",
    )
    parser.add_argument(
        "--janus-audio-bitrate", default="100k",
        help="Janus Opus audio bitrate (default: 100k)",
    )
    args = parser.parse_args()

    n = len(args.inputs)
    if n < 2:
        parser.error("At least 2 inputs are required.")
    janus_enabled = args.janus_preview or args.janus_output
    if not args.output and not janus_enabled:
        parser.error("Either --output or --janus-output/--janus-preview is required.")
    if args.janus_preview and not args.output and not args.janus_output:
        parser.error("--janus-preview needs --output; use --janus-output for Janus-only.")
    if args.janus_video_bitrate_kbps <= 0:
        parser.error("--janus-video-bitrate-kbps must be greater than 0.")
    if 0 <= args.switch_pts_lead_ms < 100:
        parser.error("--switch-pts-lead-ms must be at least 100, or negative to disable PTS scheduling.")
    switch_pts_lead_ms = args.switch_pts_lead_ms if args.switch_pts_lead_ms >= 0 else None
    sergio_input_index = find_named_input(args.inputs, SERGIO_INPUT_NAME)
    rene_input_index = find_named_input(args.inputs, RENE_INPUT_NAME)
    genaro_input_index = find_named_input(args.inputs, GENARO_INPUT_NAME)
    speaker_registry = Speaker()

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
            static_face_crop=(args.static_genaro_face_crop and i == genaro_input_index),
            debug_mouth_rois=args.debug_mouth_roi_bboxes,
            speaker_registry=speaker_registry,
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

    preheated_scenes = None
    if not args.disable_preheated_scenes:
        preheated_scenes = build_preheated_scene_sources(
            avp,
            mx,
            face_edges=[sg["face_edge"] for sg in subgraphs],
            orig_edges=[sg["orig_edge"] for sg in subgraphs],
            input_groups=[sg["input_group"] for sg in subgraphs],
        )

    if preheated_scenes:
        define_preheated_scenes(mx, n, preheated_scenes)
    else:
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
    auto_switch_scene = lambda i: f"{args.auto_switch_layout}_{i}"
    mx.set_initial_scene(auto_switch_scene(0), slot="A")
    video_out_edge = mx.build()

    record_enabled = args.output is not None

    # ---- Output routing ----
    if rene_input_index is None:
        parser.error(f'Program audio input "{RENE_INPUT_NAME}" was not found in --inputs.')
    program_audio_edge = subgraphs[rene_input_index]["program_audio_edge"]
    for i, sg in enumerate(subgraphs):
        if i == rene_input_index:
            continue
        avp.addNode(NullSink({
            "name": f"program_audio_sink_{i}",
            "src": sg["program_audio_edge"],
            "group": sg["input_group"],
            "auto_restart": "group",
        }))

    record_video_edge = video_out_edge
    janus_video_edge = video_out_edge
    record_audio_edge = program_audio_edge
    janus_audio_edge = program_audio_edge

    if record_enabled and janus_enabled:
        avp.addNode(Split({
            "name": "split_program_video_output",
            "src": video_out_edge,
            "dst": ["program_video_record", "program_video_janus"],
            "group": "output",
        }))
        record_video_edge = "program_video_record"
        janus_video_edge = "program_video_janus"
        avp.addNode(Split({
            "name": "split_program_audio_output",
            "src": program_audio_edge,
            "dst": ["program_audio_record", "program_audio_janus"],
            "group": "output",
        }))
        record_audio_edge = "program_audio_record"
        janus_audio_edge = "program_audio_janus"

    if record_enabled:
        audio_enc_edge = build_audio_output(avp, record_audio_edge, codec=args.audio_codec, prefix="program")
        video_enc_edge = build_video_output(avp, record_video_edge, args, prefix="program")
        build_mux_output(
            avp,
            [video_enc_edge, audio_enc_edge],
            mux_edge="program_mux_out",
            output_url=args.output,
            output_format=output_format_for_url(args.output),
        )

    if janus_enabled:
        build_janus_rtp_output(avp, janus_video_edge, janus_audio_edge, args)

    # ---- Speech detection: Silero audio VAD + visual lip-motion ----
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
    if preheated_scenes:
        avp.group(PREHEATED_GROUP).startNodes()
    mx.start_groups()
    avp.group("output").startNodes()

    output_targets = []
    if record_enabled:
        output_targets.append(args.output)
    if janus_enabled:
        output_targets.append(
            f"Janus RTP video={args.janus_host}:{args.janus_video_port} audio={args.janus_host}:{args.janus_audio_port}"
        )
    print(f"[auto_mixer] Graph started with {n} input(s) → {', '.join(output_targets)}")
    print(f"[auto_mixer] Canvas: {CANVAS_W}x{CANVAS_H}, face engine: {args.face_engine}")
    if sergio_input_index is not None:
        print(f"[auto_mixer] Sergio input detected: {sergio_input_index} ({input_basename(args.inputs[sergio_input_index])})")
    if rene_input_index is not None:
        print(f"[auto_mixer] Rene input detected: {rene_input_index} ({input_basename(args.inputs[rene_input_index])})")
    if genaro_input_index is not None:
        crop_mode = "static centered 9:16 crop" if args.static_genaro_face_crop else "face-tracked 9:16 crop"
        print(f"[auto_mixer] Genaro input detected: {genaro_input_index} ({input_basename(args.inputs[genaro_input_index])}) - {crop_mode}")
    if args.debug_mouth_roi_bboxes:
        print("[auto_mixer] Debug mouth ROI and speaking status overlays enabled")
    if preheated_scenes:
        print(f"[auto_mixer] Preheated scene geometries enabled: {preheated_scenes.summary()}")
    else:
        print("[auto_mixer] Preheated scene geometries disabled")
    if switch_pts_lead_ms is None:
        print("[auto_mixer] Auto-switch PTS scheduling disabled")
    else:
        print(f"[auto_mixer] Auto-switch cuts scheduled {switch_pts_lead_ms} ms ahead of VAD PTS")
    print(f"[auto_mixer] Scenes: {mx.scenes()}")

    # ---- Auto-switcher ----
    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=n,
        scene_for_input=auto_switch_scene,
        fade_duration_s=args.fade,
        min_dwell_program_s=args.min_dwell,
        min_active_level_db=MIN_ACTIVE_AUDIO_LEVEL_DBFS,
        switch_pts_lead_ms=switch_pts_lead_ms,
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
                    audio_pts_s = (
                        f"  a_pts={e.audio_event_pts_ms / 1000.0:.3f}s"
                        if e.audio_event_pts_ms is not None
                        else ""
                    )
                    print(
                        f"[vad] input {e.index}: {audio_s} {visual_s}  "
                        f"{e.level_db:+.1f} dB  "
                        f"dur={duration_s:.1f}s"
                        f"{audio_pts_s}"
                    )
                print(f"[mixer] PGM: {mx.current_scene}")
    finally:
        if not args.disable_auto_switcher:
            switcher.stop()
        print("[auto_mixer] Done.")


if __name__ == "__main__":
    main()
