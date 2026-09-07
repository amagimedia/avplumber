#!/usr/bin/env python3
"""Playlist backend process: mixer graph, policy loop and control commands.

Run this on the NVIDIA host, then attach ``player.py`` over TCP.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

from avpmixer.janus import JanusVideoConfig
from control import VERBS, apply_verb, now_ms, status_json
from engine import PlaylistConfig, PlaylistEngine, load_avp_api
from playlist import Clip, ElementMode, PlaylistController, PlaylistMode, Transition

FIXTURE_NAMES = ("01-testsrc2.mp4", "02-smpte.mp4", "03-smpte-hd.mp4", "04-rgb.mp4", "05-yuv.mp4")
POLL_SEC = 0.02


def default_clips(media_dir: Path) -> List[Clip]:
    clips = [Clip(url=str(media_dir / name), name=name.rsplit(".", 1)[0], item_id=f"fixture-{i}")
             for i, name in enumerate(FIXTURE_NAMES)]
    clips[2] = Clip(url=clips[2].url, name=clips[2].name, item_id=clips[2].item_id,
                    play_from_ms=2000, play_to_ms=8000)
    clips[3] = Clip(url=clips[3].url, name=clips[3].name, item_id=clips[3].item_id,
                    element_mode=ElementMode.TIMED, duration_ms=4000, speed=2.0)
    return clips


def load_clips(path: Path) -> List[Clip]:
    from control import clip_from_json
    return [clip_from_json(entry) for entry in json.loads(path.read_text())]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--media-dir", type=Path, default=Path(__file__).resolve().parent / "test-media")
    p.add_argument("--playlist", type=Path, help="JSON list of elements instead of the five fixtures")
    p.add_argument("--mode", choices=[m.value for m in PlaylistMode], default=PlaylistMode.LOOP_ALL.value)
    p.add_argument("--transition", choices=[t.value for t in Transition], default=Transition.CUT.value)
    p.add_argument("--transition-ms", type=int, default=500)
    p.add_argument("--wipe-file", help="Alpha media file for Wipe transitions")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--control-port", type=int, default=7778)
    p.add_argument("--janus-host", default="127.0.0.1")
    p.add_argument("--janus-video-port", type=int, default=5004)
    p.add_argument("--janus-video-pt", type=int, default=96)
    p.add_argument("--janus-video-ssrc", type=lambda v: int(v, 0), default=0x41565001)
    p.add_argument("--janus-bitrate-kbps", type=int, default=4500)
    p.add_argument("--janus-rtcp-bind", default="0.0.0.0")
    p.add_argument("--janus-rtcp-port", type=int, default=0)
    p.add_argument("--preroll-ms", type=int, default=50)
    p.add_argument("--switch-margin-ms", type=int, default=100)
    p.add_argument("--control-timeout", type=float, default=10.0)
    p.add_argument("--log-file", default="playlist-demo.log")
    p.add_argument("--record", help="Also write the program to this .mp4/.ts for verification")
    p.add_argument("--run-seconds", type=float, help="Exit after this long (smoke tests)")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PlaylistConfig:
    return PlaylistConfig(
        fps=args.fps, control_port=args.control_port, control_timeout=args.control_timeout,
        log_file=args.log_file, preroll_ms=args.preroll_ms, switch_margin_ms=args.switch_margin_ms,
        wipe_file=args.wipe_file, record=args.record,
        janus=JanusVideoConfig(host=args.janus_host, video_port=args.janus_video_port,
                               payload_type=args.janus_video_pt, ssrc=args.janus_video_ssrc,
                               bitrate_kbps=args.janus_bitrate_kbps, rtcp_bind=args.janus_rtcp_bind,
                               rtcp_port=args.janus_rtcp_port))


class PlaylistServer:
    """Owns the engine and controller; serializes control commands with the poll loop."""

    def __init__(self, avp, api, config: PlaylistConfig, clips: List[Clip], mode: PlaylistMode,
                 transition: Transition, transition_ms: int):
        self.avp = avp
        self.engine = PlaylistEngine(avp, api, config)
        self.controller = PlaylistController(self.engine, clips, mode, transition, transition_ms)
        self.config = config
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def register_commands(self) -> None:
        self.avp.registerControlCommand("playlist.status", self._status, True)
        for verb in VERBS:
            self.avp.registerControlCommand(f"playlist.{verb}", self._verb(verb), True)

    def _status(self, _arg: str) -> str:
        with self._lock:
            now = now_ms()
            self.controller.poll(now)
            # Native commands end their payload with "\n" so the server's own "\n"
            # forms the blank line that terminates a 201 response.  Do the same.
            return json.dumps(status_json(self.controller, now), separators=(",", ":")) + "\n"

    def _verb(self, verb: str):
        def handler(arg: str) -> str:
            payload = json.loads(arg) if arg.strip() else {}
            with self._lock:
                try:
                    apply_verb(self.controller, verb, payload)
                except (ValueError, IndexError, TypeError) as exc:
                    self.controller.set_error(str(exc))
            return ""
        return handler

    def start(self) -> None:
        first = self.controller.clips[self.controller.selected_index]
        self.engine.build(first)
        self.register_commands()
        self.engine.start()
        with self._lock:
            if not self.controller.play():
                raise RuntimeError("first element did not start")
        deadline = time.monotonic() + self.config.control_timeout
        while True:
            with self._lock:
                self.controller.poll(now_ms())
                status = self.controller.status()
            if status.playing and status.output_alive:
                break
            if status.error:
                raise RuntimeError(status.error)
            if time.monotonic() >= deadline:
                raise TimeoutError("playlist startup timed out")
            time.sleep(POLL_SEC)
        self.avp.setReady()

    def run(self, run_seconds=None) -> None:
        end = None if run_seconds is None else time.monotonic() + run_seconds
        while not self._stop.is_set() and (end is None or time.monotonic() < end):
            with self._lock:
                self.controller.poll(now_ms())
            time.sleep(POLL_SEC)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.engine.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    clips = load_clips(args.playlist) if args.playlist else default_clips(args.media_dir)
    api = load_avp_api()
    server = PlaylistServer(api.AVPlumber(), api, build_config(args), clips,
                            PlaylistMode(args.mode), Transition(args.transition), args.transition_ms)
    signal.signal(signal.SIGINT, lambda *_: server.stop())
    signal.signal(signal.SIGTERM, lambda *_: server.stop())
    try:
        server.start()
        print(f"playlist ready: control port {args.control_port}, "
              f"RTP to {args.janus_host}:{args.janus_video_port}", flush=True)
        server.run(args.run_seconds)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
