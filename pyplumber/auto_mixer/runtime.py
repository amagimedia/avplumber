from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable

from pyplumber.auto_switcher import AutoSwitcher

from .auto_switch_control import AutoSwitchControlCommands
from .config import CANVAS_H, CANVAS_W, FPS_DEN, FPS_NUM, MIN_ACTIVE_AUDIO_LEVEL_DBFS
from .control_status import MixerProgramSceneReader
from .inputs import input_basename
from .native_exceptions import NativeExceptionRegistry
from .preheated import PREHEATED_GROUP, PREHEATED_ROUTER_GROUP
from .profiles.talkshow import RENE_REQUIRED_LEAD_DB
from .run_config import AutoMixerRunConfig


@dataclass
class AutoSwitchRuntime:
    switcher: AutoSwitcher
    auto_control_commands: AutoSwitchControlCommands | None
    program_scene_reader: MixerProgramSceneReader | None

    def start_switcher(self, *, enabled: bool) -> None:
        if enabled:
            self.switcher.start()

    def stop(self, *, auto_switcher_enabled: bool) -> None:
        if auto_switcher_enabled:
            self.switcher.stop()
        if self.auto_control_commands is not None:
            self.auto_control_commands.stop()
        if self.program_scene_reader is not None:
            self.program_scene_reader.close()


def start_graph_groups(avp, *, n_inputs: int, mixer, preheated_scenes) -> None:
    for i in range(n_inputs):
        avp.group(f"input_{i}").startNodes()
    # Group startup is asynchronous; give upstream groups a short head start
    # before creating dependent preheat/mixer/output nodes.
    time.sleep(2.0)
    if preheated_scenes:
        avp.group(PREHEATED_ROUTER_GROUP).startNodes()
        time.sleep(2.0)
        avp.group(PREHEATED_GROUP).startNodes()
        time.sleep(2.0)
    mixer.start_groups()
    time.sleep(2.0)
    avp.group("output").startNodes()


def print_startup_summary(
    args,
    config: AutoMixerRunConfig,
    mx,
    preheated_scenes,
    auto_switch_selector,
) -> None:
    print(f"[auto_mixer] Graph started with {config.n_inputs} input(s) -> {', '.join(config.output_targets)}")
    print(f"[auto_mixer] Canvas: {CANVAS_W}x{CANVAS_H}, face engine: {args.face_engine}")
    if config.sergio_input_index is not None:
        print(
            f"[auto_mixer] Sergio input detected: {config.sergio_input_index} "
            f"({input_basename(args.inputs[config.sergio_input_index])})"
        )
    if config.rene_input_index is not None:
        print(
            f"[auto_mixer] Rene input detected: {config.rene_input_index} "
            f"({input_basename(args.inputs[config.rene_input_index])})"
        )
    if config.genaro_input_index is not None:
        crop_mode = "static centered 9:16 crop" if args.static_genaro_face_crop else "face-tracked 9:16 crop"
        print(
            f"[auto_mixer] Genaro input detected: {config.genaro_input_index} "
            f"({input_basename(args.inputs[config.genaro_input_index])}) - {crop_mode}"
        )
    if args.debug_mouth_roi_bboxes:
        print("[auto_mixer] Debug mouth ROI and speaking status overlays enabled")
    if args.html_overlay_socket:
        print("[auto_mixer] HTML overlay graph enabled; starts off")
    if preheated_scenes:
        print(
            f"[auto_mixer] Preheated scene geometries enabled: "
            f"{preheated_scenes.summary()}"
        )
    else:
        print("[auto_mixer] Preheated scene geometries disabled")
    if config.switch_pts_lead_ms is None:
        print("[auto_mixer] Auto-switch PTS scheduling disabled")
    else:
        print(f"[auto_mixer] Auto-switch cuts scheduled {config.switch_pts_lead_ms} ms ahead of VAD PTS")
    if args.auto_switch_shot_profile == "fixed":
        print(f"[auto_mixer] Auto-switch layout: fixed {args.auto_switch_layout}")
    else:
        print(
            f"[auto_mixer] Auto-switch shot profile: {args.auto_switch_shot_profile} "
            f"(seed={args.auto_switch_random_seed})"
        )
    if auto_switch_selector is not None:
        print(
            "[auto_mixer] Manual geometry suggestions: "
            f"{auto_switch_selector.rules.manual_suggestion_window_s:.1f}s "
            f"{auto_switch_selector.rules.manual_suggestion_families}"
        )
    print(f"[auto_mixer] Scenes: {mx.scenes()}")


def create_auto_switch_runtime(
    *,
    args,
    avp,
    mx,
    speaker_registry,
    run_config: AutoMixerRunConfig,
    native_exception_registry: NativeExceptionRegistry,
    auto_switch_scene: Callable[[int], str],
) -> AutoSwitchRuntime:
    program_scene_reader = None
    if args.remote_control_port:
        program_scene_reader = MixerProgramSceneReader(
            host="localhost",
            port=args.remote_control_port,
            mixer_name=mx.name,
        )
    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=run_config.n_inputs,
        scene_for_input=auto_switch_scene,
        fade_duration_s=args.fade,
        wipe_file=args.auto_switch_wipe_file or None,
        transition_mode=args.auto_switch_transition,
        min_dwell_program_s=args.min_dwell,
        min_active_level_db=MIN_ACTIVE_AUDIO_LEVEL_DBFS,
        switch_pts_lead_ms=run_config.switch_pts_lead_ms,
        special_speaker_index=run_config.rene_input_index,
        special_speaker_margin_db=RENE_REQUIRED_LEAD_DB,
        vad_only_priority_speaker_index=run_config.sergio_input_index,
        program_scene_getter=(
            program_scene_reader.program_scene if program_scene_reader is not None else None
        ),
    )
    auto_control_commands = None
    if args.remote_control_port:
        auto_control_commands = AutoSwitchControlCommands(
            avp,
            switcher,
            native_exceptions=native_exception_registry,
        )
        auto_control_commands.start()
        print(
            "[auto_mixer] Auto-switch control: main control port "
            f"{args.remote_control_port}"
        )
    print(
        "[auto_mixer] Auto-switch transition: "
        f"{args.auto_switch_transition}"
        f"{f' ({args.fade:.3f}s)' if args.auto_switch_transition == 'fade' else ''}"
    )
    if args.auto_switch_wipe_file:
        print(f"[auto_mixer] Auto-switch wipe file: {args.auto_switch_wipe_file}")
    print(
        f"[auto_mixer] Fade duration: {args.fade_frames} frame(s) "
        f"at {FPS_NUM / FPS_DEN:g} fps ({args.fade:.3f}s)"
    )
    return AutoSwitchRuntime(
        switcher=switcher,
        auto_control_commands=auto_control_commands,
        program_scene_reader=program_scene_reader,
    )


def run_until_interrupted(
    *,
    runtime: AutoSwitchRuntime,
    speaker_registry,
    mx,
    auto_switcher_enabled: bool,
) -> None:
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
        runtime.stop(auto_switcher_enabled=auto_switcher_enabled)
        print("[auto_mixer] Done.")
