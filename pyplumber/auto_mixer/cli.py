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
        [--fade-frames 15] \\
        [--remote-control-port 7777] \\
        [--logfile /tmp/auto_mixer.log] \\
        [--webui-api http://<webui-host>:<port>] \\
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
import json
import os

from pyplumber.audio_vad import SileroVadRegistryBridge, Speaker, VisualSpeechRegistryNode
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
)
from .inputs import build_input_subgraph, default_face_engine
from .native_exceptions import AutoMixerAVPlumber, NativeExceptionRegistry
from .outputs import (
    build_audio_output,
    build_html_overlay_output,
    build_janus_rtp_output,
    build_mux_output,
    build_video_output,
    output_format_for_url,
)
from .profiles.talkshow import RENE_INPUT_NAME
from .run_config import derive_run_config
from .runtime import (
    create_auto_switch_runtime,
    print_startup_summary,
    run_until_interrupted,
    start_graph_groups,
)
from .shot_rules import (
    DEFAULT_MANUAL_SUGGESTION_FAMILIES,
    DEFAULT_MANUAL_SUGGESTION_WINDOW_S,
    ShotRules,
)
from .preheated import (
    build_preheated_scene_sources,
    define_preheated_scenes,
)
from .scenes import define_auto_scenes
from .shot_selector import (
    AutoShotSceneBuilder,
    HistoryAwareShotSelector,
    profile_names,
)


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
        "--wipe", dest="wipe", action="store_true", default=True,
        help="Declare wipe subgraph for mixer.wipe transitions (default: enabled)",
    )
    parser.add_argument(
        "--disable-wipe", dest="wipe", action="store_false",
        help="Do not declare the wipe subgraph.",
    )
    parser.add_argument(
        "--fade", default=None, type=float,
        help="Crossfade duration in seconds when auto-switch transitions use fade; overrides --fade-frames.",
    )
    parser.add_argument(
        "--fade-frames", default=15, type=int,
        help=f"Crossfade duration in frames at {FPS_NUM / FPS_DEN:g} fps (default: 15)",
    )
    parser.add_argument(
        "--auto-switch-transition",
        choices=("cut", "fade", "wipe"),
        default="cut",
        help="Transition used by automatic AI speaker switches (default: cut)",
    )
    parser.add_argument(
        "--auto-switch-wipe-file", default="",
        help="Media wipe file used when automatic AI speaker switches use wipe.",
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
        help="Web UI server API endpoint URL for auto-registration",
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
        help="Scene family used by the fixed auto switcher and as profile fallback (default: videoconf)",
    )
    parser.add_argument(
        "--auto-switch-shot-profile",
        choices=("fixed", *profile_names()),
        default="fixed",
        help="Auto-switch shot selection profile. fixed preserves --auto-switch-layout.",
    )
    parser.add_argument(
        "--manual-suggestion-window",
        default=DEFAULT_MANUAL_SUGGESTION_WINDOW_S,
        type=float,
        help=(
            "Seconds to keep a manually clicked scene geometry as the auto-switch "
            f"suggestion (default: {DEFAULT_MANUAL_SUGGESTION_WINDOW_S:g})"
        ),
    )
    parser.add_argument(
        "--auto-switch-random-seed",
        default=20260521,
        type=int,
        help="Random seed for non-fixed auto-switch shot profiles.",
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
        "--html-overlay-socket",
        default=os.environ.get("AVP_HTML_OVERLAY_SOCKET", ""),
        help="Enable final Electron DMA-BUF overlay from this UNIX socket path (empty = disabled).",
    )
    parser.add_argument(
        "--html-overlay-drm-device",
        default=os.environ.get("AVP_HTML_OVERLAY_DRM_DEVICE", ""),
        help="DRM render device used to attach a DRM hw_frames_ctx to the overlay source (empty = omit).",
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
        "--janus-video-only", action="store_true",
        help="Send only video to Janus RTP.",
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

    run_config = derive_run_config(args, parser)
    n = run_config.n_inputs
    janus_enabled = run_config.janus_enabled
    record_enabled = run_config.record_enabled
    rene_input_index = run_config.rene_input_index
    genaro_input_index = run_config.genaro_input_index
    speaker_registry = Speaker()
    native_exception_registry = NativeExceptionRegistry()

    avp = AutoMixerAVPlumber(native_exception_registry)
    avp.setLogFile(args.logfile)
    if args.remote_control_port:
        avp.enableControlServer(args.remote_control_port)
    if args.webui_api:
        avp.registerWithWebUI(args.webui_api, args.instance_name, args.logfile)
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    overlay_source_hwaccel = None
    if args.html_overlay_socket and args.html_overlay_drm_device:
        avp.executeCommandsFromString(
            'hwaccel.init { "name": "@drm", "type": "drm", '
            f'"device": {json.dumps(args.html_overlay_drm_device)} }}'
        )
        overlay_source_hwaccel = "@drm"
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

    auto_scene_builder = AutoShotSceneBuilder(
        mx,
        n_inputs=n,
        preheated=preheated_scenes,
    )
    auto_scene_builder.register_initial_scenes()
    auto_switch_selector = None
    if args.auto_switch_shot_profile == "fixed":
        fixed_rules = ShotRules(
            name=f"fixed-{args.auto_switch_layout}",
            single_speaker_weights=((args.auto_switch_layout, 100),),
            stack_rules=(),
            conversation_window_s=6.0,
            stack_hold_s=4.0,
            manual_suggestion_window_s=args.manual_suggestion_window,
            manual_suggestion_families=DEFAULT_MANUAL_SUGGESTION_FAMILIES,
            avoid_recent_scenes=3,
        )
        auto_switch_selector = HistoryAwareShotSelector(
            mx.scenes(),
            n_inputs=n,
            fallback_layout=args.auto_switch_layout,
            scene_builder=auto_scene_builder,
            rules=fixed_rules,
            seed=args.auto_switch_random_seed,
        )
        auto_switch_scene = auto_switch_selector
        auto_initial_scene = auto_switch_selector.initial_scene(0)
    else:
        auto_switch_selector = HistoryAwareShotSelector(
            mx.scenes(),
            n_inputs=n,
            profile_name=args.auto_switch_shot_profile,
            fallback_layout=args.auto_switch_layout,
            scene_builder=auto_scene_builder,
            manual_suggestion_window_s=args.manual_suggestion_window,
            seed=args.auto_switch_random_seed,
        )
        auto_switch_scene = auto_switch_selector
        auto_initial_scene = auto_switch_selector.initial_scene(0)

    mx.set_initial_scene(auto_initial_scene, slot="A")
    video_out_edge = mx.build()
    if args.html_overlay_socket:
        video_out_edge = build_html_overlay_output(
            avp,
            video_out_edge,
            socket_path=args.html_overlay_socket,
            source_hwaccel=overlay_source_hwaccel,
        )

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
    janus_audio_edge = None if args.janus_video_only else program_audio_edge

    if record_enabled and janus_enabled:
        avp.addNode(Split({
            "name": "split_program_video_output",
            "src": video_out_edge,
            "dst": ["program_video_record", "program_video_janus"],
            "group": "output",
            "on_error": "panic",
        }))
        record_video_edge = "program_video_record"
        janus_video_edge = "program_video_janus"
        if args.janus_video_only:
            record_audio_edge = program_audio_edge
        else:
            avp.addNode(Split({
                "name": "split_program_audio_output",
                "src": program_audio_edge,
                "dst": ["program_audio_record", "program_audio_janus"],
                "group": "output",
                "on_error": "panic",
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
    start_graph_groups(
        avp,
        n_inputs=n,
        mixer=mx,
        preheated_scenes=preheated_scenes,
    )

    print_startup_summary(args, run_config, mx, preheated_scenes, auto_switch_selector)

    # ---- Auto-switcher ----
    auto_runtime = create_auto_switch_runtime(
        args=args,
        avp=avp,
        mx=mx,
        speaker_registry=speaker_registry,
        run_config=run_config,
        native_exception_registry=native_exception_registry,
        auto_switch_scene=auto_switch_scene,
    )

    # Unlock the control server so remote clients can issue commands.
    avp.setReady()
    auto_switcher_enabled = not args.disable_auto_switcher
    auto_runtime.start_switcher(enabled=auto_switcher_enabled)

    # ---- Run until interrupted ----
    run_until_interrupted(
        runtime=auto_runtime,
        speaker_registry=speaker_registry,
        mx=mx,
        auto_switcher_enabled=auto_switcher_enabled,
    )


if __name__ == "__main__":
    main()
