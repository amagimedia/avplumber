import asyncio
import json

import pytest

from control import AvpConnection, mixer_command, parse_mixer_status, parse_scene_list


def test_cancelled_tui_request_cannot_leave_a_reply_for_the_next_command():
    """Exclusive TUI workers can be cancelled while a TCP reply is pending."""
    async def exercise():
        received = asyncio.Event()
        release = asyncio.Event()

        async def serve(reader, writer):
            writer.write(b"100 ready\n")
            await writer.drain()
            await reader.readline()
            received.set()
            await release.wait()
            writer.write(b'201 status\n{"transition":"idle"}\n\n')
            try:
                await writer.drain()
                await reader.readline()
                writer.write(b"200 cut accepted\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        connection = AvpConnection("127.0.0.1", server.sockets[0].getsockname()[1])
        try:
            await connection.connect()
            request = asyncio.create_task(connection.command("mixer.status mixer"))
            await asyncio.wait_for(received.wait(), 1)
            request.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert connection.connected
            assert await connection.command('mixer.cut {"scene":"next"}') is None
        finally:
            release.set()
            await connection.disconnect()
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


def test_mixer_command_uses_generic_json_payload():
    command = mixer_command("fade", "main", scene="grid_4_page_0", duration_sec=0.5)
    prefix, payload = command.split(" ", 1)
    assert prefix == "mixer.fade"
    assert json.loads(payload) == {
        "mixer": "main",
        "scene": "grid_4_page_0",
        "duration_sec": 0.5,
    }


def test_status_and_scene_parsing():
    status = parse_mixer_status(
        '{"pgm_scene":"fullscreen_0","pvw_scene":"grid_2_page_0","transition":"idle"}'
    )
    assert status.pgm_scene == "fullscreen_0"
    assert status.pvw_scene == "grid_2_page_0"
    assert status.transition == "idle"
    assert parse_scene_list('["fullscreen_0", "grid_2_page_0"]') == [
        "fullscreen_0",
        "grid_2_page_0",
    ]


def test_invalid_response_shapes_are_rejected():
    with pytest.raises(ValueError, match="object"):
        parse_mixer_status("[]")
    with pytest.raises(ValueError, match="list"):
        parse_scene_list("{}")
