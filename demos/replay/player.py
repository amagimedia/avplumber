#!/usr/bin/env python3
"""Play one AVPlumber replay recording directly to a Janus RTP mountpoint."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from replay import (JanusVideoConfig, PlaybackOperation as Op, PlayerConfig,
                    ReplaySlotConfig, build_player_application)


@dataclass(frozen=True)
class ExerciseResult:
    name: str
    outcome: str
    detail: str = ""


def _wait(controller, predicate, timeout: float):
    deadline = time.monotonic() + timeout
    before = controller.status()
    while time.monotonic() < deadline:
        status = controller.status()
        if predicate(status):
            return status
        time.sleep(0.02)
    raise TimeoutError(f"before={before}; after={controller.status()}")


def exercise_v2(controller, timeout: float) -> list[ExerciseResult]:
    results: list[ExerciseResult] = []

    def check(name, action, predicate=lambda _status: True):
        try:
            action()
            status = _wait(controller, predicate, timeout)
            results.append(ExerciseResult(name, "PASS", f"frame={status.frame_number} pos={status.position_ms}ms"))
            return status
        except Exception as exc:
            results.append(ExerciseResult(name, "FAIL", str(exc)))
            return None

    artifact = controller.artifact
    frame_ms = 1000 / artifact.fps
    try:
        ready = _wait(controller, lambda status: status.ready, timeout)
    except Exception as exc:
        return [ExerciseResult("source ready", "FAIL", str(exc))]
    results.append(ExerciseResult("source ready", "PASS", f"frame={ready.frame_number}"))

    start = ready.position_ms
    check("play advances", lambda: controller.execute(Op.PLAY), lambda s: s.position_ms > start + frame_ms)
    controller.execute(Op.PAUSE)
    paused = controller.status().position_ms
    time.sleep(min(timeout, max(0.1, 2 / artifact.fps)))
    stable = abs(controller.status().position_ms - paused) <= frame_ms * 1.5
    results.append(ExerciseResult("pause stable", "PASS" if stable else "FAIL", f"{paused}->{controller.status().position_ms}ms"))

    middle = artifact.duration_ms // 2
    check(
        "absolute seek",
        lambda: controller.execute(Op.SEEK_MS, middle),
        lambda s: abs(s.position_ms - middle) <= frame_ms * 2,
    )
    for frames in (-30, -5, -1, 1, 5, 30):
        base_frame = min(max(40, artifact.frame_count // 2), artifact.frame_count - 41)
        if not 0 <= base_frame + frames < artifact.frame_count:
            results.append(ExerciseResult(f"nudge {frames:+d}f", "SKIP", "recording too short"))
            continue
        base_ms = artifact.seek_entries[base_frame].timestamp_ms - artifact.start_ms
        controller.execute(Op.SEEK_MS, base_ms)
        _wait(controller, lambda s: abs((s.frame_number or 0) - base_frame) <= 1, timeout)
        check(
            f"nudge {frames:+d}f",
            lambda frames=frames: controller.execute(Op.SEEK_FRAMES, frames),
            lambda s, expected=base_frame + frames: abs((s.frame_number or 0) - expected) <= 1,
        )

    for seconds in (-30, -5, -1, 1, 5, 30):
        target = middle + seconds * 1000
        if not 0 <= target <= artifact.duration_ms:
            results.append(ExerciseResult(f"nudge {seconds:+d}s", "SKIP", "recording too short"))
            continue
        controller.execute(Op.SEEK_MS, middle)
        _wait(controller, lambda s: abs(s.position_ms - middle) <= frame_ms * 2, timeout)
        check(
            f"nudge {seconds:+d}s",
            lambda seconds=seconds: controller.execute(Op.SEEK_SECONDS, seconds),
            lambda s, target=target: abs(s.position_ms - target) <= frame_ms * 2,
        )

    if artifact.duration_ms < 6_000:
        results.append(ExerciseResult("speed 50/100/200%", "SKIP", "recording too short"))
    else:
        rates = []
        for speed in (50, 100, 200):
            controller.execute(Op.PAUSE)
            controller.execute(Op.SEEK_MS, 1_000)
            _wait(controller, lambda s: abs(s.position_ms - 1_000) <= frame_ms * 2, timeout)
            controller.execute(Op.SPEED, speed)
            before = controller.status().position_ms
            started = time.monotonic()
            controller.execute(Op.PLAY)
            time.sleep(min(0.6, timeout / 2))
            controller.execute(Op.PAUSE)
            rates.append((controller.status().position_ms - before) / (time.monotonic() - started))
        distinct = rates[0] < rates[1] < rates[2]
        results.append(ExerciseResult("speed 50/100/200%", "PASS" if distinct else "FAIL", repr([round(rate) for rate in rates])))

        try:
            controller.execute(Op.SEEK_MS, 1_000)
            _wait(controller, lambda s: abs(s.position_ms - 1_000) <= frame_ms * 2, timeout)
            controller.execute(Op.PLAY)
            _wait(controller, lambda s: s.position_ms > 1_000 + frame_ms * 2, timeout)
            controller.execute(Op.SPEED, 100)
            marker = controller.observation_marker()
            frames = []
            deadline = time.monotonic() + timeout
            while len(frames) < 10 and time.monotonic() < deadline:
                frames = list(controller.observed_frames_since(marker))
                time.sleep(0.005)
            frames = frames[:10]
            deltas = [current - previous for previous, current in zip(frames, frames[1:])]
            continuous = len(frames) == 10 and all(delta == 1 for delta in deltas)
            results.append(ExerciseResult(
                "speed 200->100% frame continuity",
                "PASS" if continuous else "FAIL",
                repr(frames),
            ))
        except Exception as exc:
            results.append(ExerciseResult("speed 200->100% frame continuity", "FAIL", str(exc)))

    controller.execute(Op.SPEED, 100)
    controller.execute(Op.PAUSE)
    controller.execute(Op.SEEK_MS, middle)
    _wait(controller, lambda s: abs(s.position_ms - middle) <= frame_ms * 2, timeout)
    check("reverse play", lambda: controller.execute(Op.REVERSE), lambda s: s.position_ms < middle - frame_ms)
    controller.execute(Op.PAUSE)
    prior = controller.status()
    controller.execute(Op.SCRUB, 200)
    check("scrub forward", lambda: None, lambda s: s.position_ms > prior.position_ms + frame_ms)
    controller.execute(Op.SCRUB, 0)
    restored = controller.status()
    restore_ok = restored.playing == prior.playing and restored.direction == prior.direction
    results.append(ExerciseResult("scrub restore", "PASS" if restore_ok else "FAIL", restored.direction))
    controller.execute(Op.SEEK_MS, middle)
    _wait(controller, lambda s: abs(s.position_ms - middle) <= frame_ms * 2, timeout)
    controller.execute(Op.SCRUB, -200)
    check("scrub reverse", lambda: None, lambda s: s.position_ms < middle - frame_ms)
    controller.execute(Op.SCRUB, 0)

    tail = max(artifact.duration_ms - 3_000, 0)
    check("tail -3s", lambda: controller.execute(Op.TAIL), lambda s: abs(s.position_ms - tail) <= frame_ms * 2)
    utc_ms = artifact.history.media_to_wallclock_ms(artifact.start_ms + middle)
    utc = datetime.fromtimestamp(utc_ms / 1000, timezone.utc)
    check("UTC seek", lambda: controller.execute(Op.SEEK_UTC, utc), lambda s: abs(s.position_ms - middle) <= frame_ms * 2)

    controller.execute(Op.PAUSE)
    started = time.monotonic()
    for index in range(20):
        controller.execute(Op.SEEK_MS, middle + (index % 2) * min(1_000, artifact.duration_ms - middle))
    rapid_ok = time.monotonic() - started < timeout
    results.append(ExerciseResult("rapid paused seeks", "PASS" if rapid_ok else "FAIL"))
    return results


def parse_media_time(value: str) -> int:
    if ":" not in value:
        return int(value)
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("use milliseconds, MM:SS, or HH:MM:SS")
    seconds = float(parts[-1]) + int(parts[-2]) * 60
    if len(parts) == 3:
        seconds += int(parts[0]) * 3600
    return round(seconds * 1000)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC seek requires Z or an explicit offset")
    return parsed.astimezone(timezone.utc)


def stop_bounded(application, timeout: float) -> bool:
    errors = []

    def stop():
        try:
            application.stop()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=stop, name="replay-stop", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False
    if errors:
        raise errors[0]
    return True


try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, Input, Static
except ImportError:
    ReplayTui = None
else:
    class ReplayTui(App):
        TITLE = "AVPlumber Replay → Janus"
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("space", "toggle", "Play/Pause"),
            Binding("left", "frame_back", "-1 frame"),
            Binding("right", "frame_forward", "+1 frame"),
            Binding("down", "second_back", "-1 second"),
            Binding("up", "second_forward", "+1 second"),
        ]
        CSS = """
        Screen { background: $surface-darken-1; }
        #state { height: 6; border: heavy $success; padding: 0 2; }
        #state.paused { border: heavy $warning; }
        #state.reverse { border: heavy $secondary; }
        #state.error { border: heavy $error; color: $error-lighten-1; }
        .row { height: 4; border: solid $primary-darken-2; padding: 0 1; }
        Button { margin-right: 1; min-width: 9; }
        Input { width: 32; margin-right: 1; }
        #exercise_results { height: 8; overflow-y: auto; }
        """

        def __init__(self, application):
            super().__init__()
            self.application = application
            self.controller = application.controller

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id="state")
            with Vertical():
                with Horizontal(classes="row"):
                    for button_id, label in (
                        ("play", "▶ PLAY"), ("pause", "⏸ PAUSE"), ("toggle", "⏯ TOGGLE"),
                        ("reverse", "◀ REVERSE"), ("scrub_-200", "⏪ SCRUB"),
                        ("scrub_200", "SCRUB ⏩"), ("scrub_0", "■ SCRUB"), ("tail", "⏭ TAIL -3s"),
                    ):
                        yield Button(label, id=button_id, classes="control")
                with Horizontal(classes="row"):
                    for value in (-30, -5, -1, 1, 5, 30):
                        yield Button(f"{value:+d}f", id=f"frame_{value}", classes="control")
                with Horizontal(classes="row"):
                    for value in (-30, -5, -1, 1, 5, 30):
                        yield Button(f"{value:+d}s", id=f"second_{value}", classes="control")
                with Horizontal(classes="row"):
                    for value in (0, 25, 50, 100, 200):
                        yield Button(f"{value / 100:g}×", id=f"speed_{value}", classes="control")
                with Horizontal(classes="row"):
                    yield Input(placeholder="milliseconds or HH:MM:SS.mmm", id="media_time")
                    yield Button("⌖ GO TO TIME", id="seek_media", classes="control")
                    yield Input(placeholder="2026-08-10T12:00:00Z", id="utc_time")
                    yield Button("🌐 GO TO UTC", id="seek_utc", classes="control")
                    yield Button("✓ RUN V2", id="exercise", classes="control")
            yield Static(id="exercise_results")
            yield Footer()

        def on_mount(self):
            self.set_interval(0.2, self.refresh_status)
            self.refresh_status()

        def refresh_status(self):
            status = self.controller.status()
            wallclock = (
                datetime.fromtimestamp(status.wallclock_ms / 1000, timezone.utc).isoformat()
                if status.wallclock_ms is not None else "unknown"
            )
            state = self.query_one("#state", Static)
            state.update(
                f"SOURCE  {self.application.artifact.path}\n"
                f"JANUS   {self.application.config.janus.host}:{self.application.config.janus.video_port}  "
                f"PT={self.application.config.janus.payload_type}  SSRC=0x{self.application.config.janus.ssrc:08x}\n"
                f"{'▶ PLAY' if status.playing else '⏸ PAUSE'}  {status.direction.upper()}  "
                f"speed={status.speed_percent:g}% scrub={status.scrubbing_percent:g}%  "
                f"frame={status.frame_number if status.frame_number is not None else '—'}  "
                f"{status.position_ms / 1000:.3f}/{status.duration_ms / 1000:.3f}s  "
                f"UTC={wallclock}  LOOP={'ON' if status.loop else 'OFF'}\n"
                f"{status.error or status.message or status.last_command}"
            )
            state.set_class(bool(status.error), "error")
            state.set_class(not status.playing and not status.error, "paused")
            state.set_class(status.direction == "reverse" and not status.error, "reverse")
            for button in self.query(".control"):
                button.disabled = not status.ready

        def _execute(self, operation, value=None):
            try:
                self.controller.execute(operation, value)
            except Exception as exc:
                self.notify(str(exc), severity="error")
            self.refresh_status()

        def action_toggle(self): self._execute(Op.TOGGLE)
        def action_frame_back(self): self._execute(Op.SEEK_FRAMES, -1)
        def action_frame_forward(self): self._execute(Op.SEEK_FRAMES, 1)
        def action_second_back(self): self._execute(Op.SEEK_SECONDS, -1)
        def action_second_forward(self): self._execute(Op.SEEK_SECONDS, 1)

        @on(Button.Pressed)
        def button_pressed(self, event):
            button_id = event.button.id or ""
            operations = {
                "play": Op.PLAY, "pause": Op.PAUSE, "toggle": Op.TOGGLE,
                "reverse": Op.REVERSE, "tail": Op.TAIL,
            }
            if button_id in operations:
                self._execute(operations[button_id])
            elif button_id.startswith("frame_"):
                self._execute(Op.SEEK_FRAMES, int(button_id[6:]))
            elif button_id.startswith("second_"):
                self._execute(Op.SEEK_SECONDS, int(button_id[7:]))
            elif button_id.startswith("speed_"):
                self._execute(Op.SPEED, int(button_id[6:]))
            elif button_id.startswith("scrub_"):
                self._execute(Op.SCRUB, int(button_id[6:]))
            elif button_id == "seek_media":
                try:
                    self._execute(Op.SEEK_MS, parse_media_time(self.query_one("#media_time", Input).value))
                except Exception as exc:
                    self.notify(str(exc), severity="error")
            elif button_id == "seek_utc":
                try:
                    self._execute(Op.SEEK_UTC, parse_utc(self.query_one("#utc_time", Input).value))
                except Exception as exc:
                    self.notify(str(exc), severity="error")
            elif button_id == "exercise":
                self.run_exercise()

        @work(thread=True, exclusive=True, group="exercise")
        def run_exercise(self):
            results = exercise_v2(self.controller, self.application.config.slot.control_timeout)
            text = "\n".join(f"{result.outcome:4} {result.name}: {result.detail}" for result in results)
            self.call_from_thread(self.query_one("#exercise_results", Static).update, text)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--janus-host", default="127.0.0.1")
    parser.add_argument("--janus-video-port", type=int, default=5004)
    parser.add_argument("--janus-video-pt", type=int, default=96)
    parser.add_argument("--janus-video-ssrc", type=lambda value: int(value, 0), default=0x41565001)
    parser.add_argument("--janus-rtcp-bind", default="0.0.0.0")
    parser.add_argument("--janus-rtcp-port", type=int, default=0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--control-timeout", type=float, default=5.0)
    parser.add_argument("--no-tui", action="store_true")
    parser.add_argument("--exercise-v2", action="store_true")
    args = parser.parse_args(argv)
    try:
        return PlayerConfig(
            ReplaySlotConfig(args.recording.resolve(), not args.no_loop, args.control_timeout),
            JanusVideoConfig(
                args.janus_host, args.janus_video_port, args.janus_video_pt,
                args.janus_video_ssrc, args.janus_rtcp_bind, args.janus_rtcp_port,
            ),
        ), args.no_tui, args.exercise_v2
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    try:
        config, no_tui, run_exercise = parse_args(argv)
        application = build_player_application(config)
        application.start()
        try:
            if run_exercise:
                results = exercise_v2(application.controller, config.slot.control_timeout)
                for result in results:
                    print(f"{result.outcome:4} {result.name}: {result.detail}")
                return 1 if any(result.outcome == "FAIL" for result in results) else 0
            if no_tui:
                while True:
                    time.sleep(1)
            elif ReplayTui is None:
                raise RuntimeError("Textual is not installed; install demos/replay/requirements.txt")
            else:
                ReplayTui(application).run()
        except KeyboardInterrupt:
            return 0
        finally:
            if not stop_bounded(application, config.slot.control_timeout):
                print("FAIL shutdown exceeded control timeout")
                return 1
    except Exception as exc:
        print(f"player failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
