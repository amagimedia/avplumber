#!/usr/bin/env python3
"""Headless five-clip CUDA/Janus playlist acceptance sequence.

Unlike the Textual UI this command intentionally prints one JSON report. It is
for the NVIDIA/Janus test host and fails immediately if the permanent RTP node
stops during any playlist or element action.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from player import (FrameworkStdoutRedirect, demo_clips,
                    stop_application_bounded)
from playlist import Clip, ElementMode, PlaylistMode, TransportState
from playlist_app import (JanusVideoConfig, PlaylistConfig,
                          build_playlist_application)


@dataclass
class Check:
    name: str
    active: int | None
    transport: str
    output_alive: bool
    position_ms: int | None


class LiveRegression:
    def __init__(self, application):
        self.application = application
        self.controller = application.controller
        self.checks: list[Check] = []
        self.timeout = application.config.control_timeout

    def poll(self):
        self.controller.poll(int(time.monotonic() * 1000))

    def wait_for(self, predicate, description: str, timeout: float | None = None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            self.poll()
            if predicate():
                return
            status = self.controller.status()
            if status.error:
                raise RuntimeError(f"{description}: {status.error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(description)
            time.sleep(0.02)

    def assert_output(self, name: str):
        self.poll()
        status = self.controller.status()
        native_alive = bool(
            self.application.avp.node("janus_rtp_output").isWorking)
        if not status.output_alive or not native_alive:
            raise RuntimeError(f"Janus RTP output stopped during {name}")
        self.checks.append(Check(
            name=name,
            active=status.active_index,
            transport=status.transport.value,
            output_alive=True,
            position_ms=self.controller.observed_pts_ms,
        ))

    def assert_quiet_output(self, name: str, settle: float = 0.25,
                            sample: float = 0.25):
        """Require a stopped source PTS and a working RTP node throughout."""
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            self.assert_output(f"{name} settle")
            self.checks.pop()
            time.sleep(0.02)
        stopped_position = self.controller.observed_pts_ms
        if stopped_position is None:
            raise RuntimeError(f"{name}: no decoded source position was observed")
        deadline = time.monotonic() + sample
        while time.monotonic() < deadline:
            self.assert_output(f"{name} sample")
            self.checks.pop()
            if self.controller.observed_pts_ms != stopped_position:
                raise RuntimeError(f"{name}: source position advanced after Stop")
            time.sleep(0.02)
        self.assert_output(name)

    def wait_playing(self, index: int, name: str):
        self.wait_for(
            lambda: (self.controller.status().active_index == index
                     and self.controller.status().transport is TransportState.PLAYING),
            name,
        )
        self.assert_output(name)

    def run(self, natural_eof_timeout: float = 13.0):
        self.assert_output("startup")

        for mode in PlaylistMode:
            self.controller.set_mode(mode)
            if self.controller.status().mode is not mode:
                raise RuntimeError(f"playlist mode did not change to {mode.value}")
            self.assert_output(f"playlist mode {mode.value}")

        self.controller.pause()
        self.wait_for(
            lambda: self.controller.status().transport is TransportState.PAUSED,
            "playlist pause")
        self.assert_output("playlist pause")
        self.controller.play()
        self.wait_playing(self.controller.status().active_index, "playlist resume")

        active = self.controller.status().active_index
        self.controller.stop()
        self.wait_for(
            lambda: self.controller.status().transport is TransportState.STOPPED,
            "playlist stop")
        self.assert_quiet_output("playlist stop quiet interval")
        self.controller.play()
        self.wait_playing(active, "playlist stop-to-play")

        # Every item executes Play, Pause, Stop, and Stop-to-Play.
        for index in range(5):
            self.controller.element_play(index)
            self.wait_playing(index, f"item {index + 1} play")
            self.controller.element_pause(index)
            self.wait_for(
                lambda: self.controller.status().transport is TransportState.PAUSED,
                f"item {index + 1} pause")
            self.assert_output(f"item {index + 1} pause")
            self.controller.element_stop(index)
            self.wait_for(
                lambda: self.controller.status().transport is TransportState.STOPPED,
                f"item {index + 1} stop")
            self.assert_quiet_output(f"item {index + 1} stop")
            self.controller.element_play(index)
            self.wait_playing(index, f"item {index + 1} stop-to-play")

        self.controller.set_mode(PlaylistMode.LOOP_ALL)
        before = self.controller.status().active_index
        self.controller.next()
        expected = (before + 1) % 5
        self.wait_playing(expected, "playlist next")
        self.controller.prev()
        self.wait_playing(before, "playlist previous")

        # Mode, cue, speed, disable, reorder, add and remove operations.
        self.controller.set_element_mode(before, ElementMode.TIMED, 3000)
        self.wait_playing(before, "active element Timed mode")
        self.controller.update_clip(before, play_from_ms=1000, play_to_ms=8000)
        self.wait_playing(before, "active element cue edit")
        self.controller.update_clip(before, speed=1.25)
        self.assert_output("active element speed")
        self.controller.set_disabled(before, True)
        self.wait_for(
            lambda: self.controller.status().transport is TransportState.STOPPED,
            "active element disable")
        self.assert_quiet_output("active element disable")
        self.controller.set_disabled(before, False)
        self.controller.element_play(before)
        self.wait_playing(before, "active element enable/play")
        inactive = (before + 2) % 5
        self.controller.reorder_clip(inactive, 0)
        self.assert_output("element reorder")

        added = Clip(
            url=self.controller.clips[0].url,
            name="temporary-added-fixture",
            item_id="temporary-added-fixture",
        )
        added_index = self.controller.append_clip(added)
        self.controller.element_play(added_index)
        self.wait_playing(added_index, "element add/play")
        self.controller.remove_clip(added_index)
        self.assert_output("element remove")
        self.controller.element_play(0)
        self.wait_playing(0, "play after active element removal")

        # A failed target must not kill the previous source or Janus graph.
        previous_active_id = self.controller.clips[
            self.controller.status().active_index].item_id
        broken = Clip(
            url=str(Path(self.application.config.log_file).with_suffix(".missing.mp4")),
            name="intentional-load-failure",
            item_id="intentional-load-failure",
        )
        broken_index = self.controller.append_clip(broken)
        self.controller.element_play(broken_index)
        self.wait_for(
            lambda: bool(self.controller.status().error),
            "intentional source failure",
            timeout=self.timeout + 1,
        )
        active_index = self.controller.status().active_index
        if active_index is None or self.controller.clips[active_index].item_id != previous_active_id:
            raise RuntimeError("failed source replaced the previous active item")
        self.assert_output("source-load failure retention")
        self.controller.remove_clip(broken_index)
        self.controller.clear_error()

        # Natural EOF must advance using the current policy.
        natural_index = self.controller.status().active_index
        self.controller.set_mode(PlaylistMode.LOOP_ALL)
        self.controller.set_element_mode(natural_index, ElementMode.PLAY_TO_END)
        self.wait_playing(natural_index, "natural EOF source restart")
        expected_id = self.controller.clips[
            (natural_index + 1) % len(self.controller.clips)].item_id
        self.wait_for(
            lambda: (self.controller.status().active_index is not None
                     and self.controller.clips[
                         self.controller.status().active_index].item_id == expected_id),
            "natural EOF advance",
            timeout=natural_eof_timeout,
        )
        self.assert_output("natural EOF advance")
        return self.checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path,
                        default=Path(__file__).with_name("test-media"))
    parser.add_argument("--janus-host", default="127.0.0.1")
    parser.add_argument("--janus-video-port", type=int, default=5004)
    parser.add_argument("--janus-video-pt", type=int, default=96)
    parser.add_argument("--janus-video-ssrc", type=lambda value: int(value, 0),
                        default=0x41565001)
    parser.add_argument("--janus-rtcp-bind", default="0.0.0.0")
    parser.add_argument("--janus-rtcp-port", type=int, default=0)
    parser.add_argument("--log-file", default="playlist-regression.log")
    parser.add_argument("--control-timeout", type=float, default=10.0)
    parser.add_argument("--natural-eof-timeout", type=float, default=13.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    redirect = FrameworkStdoutRedirect(args.log_file)
    redirect.__enter__()
    application = None
    report = None
    exit_code = 1
    try:
        application = build_playlist_application(PlaylistConfig(
            clips=demo_clips(args.media_dir),
            janus=JanusVideoConfig(
                host=args.janus_host,
                video_port=args.janus_video_port,
                payload_type=args.janus_video_pt,
                ssrc=args.janus_video_ssrc,
                rtcp_bind=args.janus_rtcp_bind,
                rtcp_port=args.janus_rtcp_port,
            ),
            control_timeout=args.control_timeout,
            log_file=args.log_file,
        ))
        application.start()
        checks = LiveRegression(application).run(args.natural_eof_timeout)
        report = {
            "result": "PASS",
            "checks": [asdict(check) for check in checks],
        }
        exit_code = 0
    except Exception as exc:
        report = {"result": "FAIL", "error": str(exc)}
    finally:
        if application is not None and not stop_application_bounded(
                application, args.control_timeout):
            report = {"result": "FAIL", "error": "playlist regression shutdown timed out"}
            exit_code = 1
        redirect.__exit__(None, None, None)
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
