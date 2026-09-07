"""Exercise every generic mixer layout and transition over the TCP protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

try:
    from .control import (
        AvpConnection,
        mixer_command,
        parse_mixer_status,
        parse_scene_list,
    )
except ImportError:
    from control import (  # type: ignore[no-redef]
        AvpConnection,
        mixer_command,
        parse_mixer_status,
        parse_scene_list,
    )


async def _command(
    connection: AvpConnection,
    command: str,
    *,
    timeout: float,
):
    return await asyncio.wait_for(connection.command(command), timeout=timeout)


async def _status(connection: AvpConnection, mixer: str, *, timeout: float):
    content = await _command(
        connection,
        f"mixer.status {mixer}",
        timeout=timeout,
    )
    if content is None:
        raise RuntimeError("mixer.status returned no content")
    return parse_mixer_status(content)


async def _wait_for(
    connection: AvpConnection,
    mixer: str,
    predicate,
    *,
    timeout: float,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await _status(connection, mixer, timeout=timeout)
        if predicate(status):
            return status
        await asyncio.sleep(0.02)
    raise TimeoutError(f"mixer status did not settle within {timeout:.1f}s")


async def _preview(
    connection: AvpConnection,
    mixer: str,
    scene: str,
    *,
    timeout: float,
) -> float:
    started = time.monotonic()
    await _command(
        connection,
        mixer_command("preview", mixer, scene=scene),
        timeout=timeout,
    )
    await _wait_for(
        connection,
        mixer,
        lambda status: status.pvw_scene == scene and status.transition == "idle",
        timeout=timeout,
    )
    return (time.monotonic() - started) * 1_000


async def _transition(
    connection: AvpConnection,
    mixer: str,
    command: str,
    scene: str,
    *,
    timeout: float,
    hold_seconds: float = 0,
    **payload,
) -> dict[str, float]:
    started = time.monotonic()
    await _command(
        connection,
        mixer_command(command, mixer, scene=scene, **payload),
        timeout=timeout,
    )
    command_ms = (time.monotonic() - started) * 1_000
    await _wait_for(
        connection,
        mixer,
        lambda status: status.pgm_scene == scene and status.transition == "idle",
        timeout=timeout,
    )
    timing = {
        "command_ms": round(command_ms, 1),
        "settled_ms": round((time.monotonic() - started) * 1_000, 1),
    }
    if hold_seconds:
        await asyncio.sleep(hold_seconds)
    return timing


async def run(args: argparse.Namespace) -> dict:
    connection = AvpConnection(args.host, args.port)
    await asyncio.wait_for(connection.connect(), timeout=args.timeout)
    try:
        content = await _command(
            connection,
            f"mixer.scenes {args.mixer}",
            timeout=args.timeout,
        )
        if content is None:
            raise RuntimeError("mixer.scenes returned no content")
        scenes = set(parse_scene_list(content))
        required = {
            "fullscreen_0",
            "grid_2_page_0",
            "grid_4_page_0",
            "grid_8_page_0",
            "grid_16_page_0",
        }
        missing = sorted(required - scenes)
        if missing:
            raise RuntimeError(f"missing required scenes: {missing}")

        previews = {}
        for scene in sorted(required):
            previews[scene] = round(
                await _preview(
                    connection,
                    args.mixer,
                    scene,
                    timeout=args.timeout,
                ),
                1,
            )

        transitions = {
            "cut": await _transition(
                connection,
                args.mixer,
                "cut",
                "grid_2_page_0",
                timeout=args.timeout,
                hold_seconds=args.hold_seconds,
            ),
        }
        await _preview(
            connection,
            args.mixer,
            "grid_4_page_0",
            timeout=args.timeout,
        )
        transitions["fade"] = await _transition(
            connection,
            args.mixer,
            "fade",
            "grid_4_page_0",
            timeout=args.timeout,
            duration_sec=args.fade_duration,
            hold_seconds=args.hold_seconds,
        )
        wipe_scenes = (
            "grid_8_page_0",
            "grid_16_page_0",
            "fullscreen_0",
            "grid_2_page_0",
        )
        for scene in wipe_scenes:
            await _preview(
                connection,
                args.mixer,
                scene,
                timeout=args.timeout,
            )
            transitions[f"media_wipe_{scene}"] = await _transition(
                connection,
                args.mixer,
                "wipe",
                scene,
                timeout=args.timeout,
                wipe_file=args.wipe_file,
                hold_seconds=args.hold_seconds,
            )
        result = {
            "scene_count": len(scenes),
            "preview_command_ms": previews,
            "transitions": transitions,
        }
        if args.strict_timing:
            validate_timing(result, args.fade_duration, args.wipe_duration)
        return result
    finally:
        await connection.disconnect()


def validate_timing(result: dict, fade_duration: float, wipe_duration: float) -> None:
    """Local-host acceptance limits; pixels/PTS require the recording test too."""
    for scene, elapsed in result["preview_command_ms"].items():
        if elapsed > 50:
            raise AssertionError(f"preview {scene}: command took {elapsed} ms")
    for name, timing in result["transitions"].items():
        if timing["command_ms"] > 50:
            raise AssertionError(f"{name}: command took {timing['command_ms']} ms")
        duration_ms = 0 if name == "cut" else 1000 * (
            fade_duration if name == "fade" else wipe_duration)
        if timing["settled_ms"] > duration_ms + 120:
            raise AssertionError(f"{name}: remained busy for {timing['settled_ms']} ms")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--fade-duration", type=float, default=0.25)
    parser.add_argument("--wipe-file", required=True,
                        help="transparent clip path on the mixer backend")
    parser.add_argument("--wipe-duration", type=float,
                        help="actual clip duration for strict timing checks; does not override playback")
    parser.add_argument("--strict-timing", action="store_true",
                        help="check command/cleanup latency on a local control connection")
    parser.add_argument("--hold-seconds", type=float, default=0,
                        help="hold each completed scene for frame-continuity recording")
    args = parser.parse_args(argv)
    if args.strict_timing and (args.wipe_duration is None or args.wipe_duration <= 0):
        parser.error("--strict-timing requires the clip's positive --wipe-duration")
    print(json.dumps(asyncio.run(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
