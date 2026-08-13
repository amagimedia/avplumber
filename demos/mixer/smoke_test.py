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
    return {
        "command_ms": round(command_ms, 1),
        "settled_ms": round((time.monotonic() - started) * 1_000, 1),
    }


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
        )
        wipe_scenes = (
            ("wipe_left", "grid_8_page_0"),
            ("wipe_right", "grid_16_page_0"),
            ("wipe_down", "fullscreen_0"),
            ("wipe_up", "grid_2_page_0"),
        )
        for style, scene in wipe_scenes:
            await _preview(
                connection,
                args.mixer,
                scene,
                timeout=args.timeout,
            )
            transitions[style] = await _transition(
                connection,
                args.mixer,
                "cuda_wipe",
                scene,
                timeout=args.timeout,
                style=style,
                duration_sec=args.wipe_duration,
            )
        return {
            "scene_count": len(scenes),
            "preview_command_ms": previews,
            "transitions": transitions,
        }
    finally:
        await connection.disconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--fade-duration", type=float, default=0.25)
    parser.add_argument("--wipe-duration", type=float, default=0.6)
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
