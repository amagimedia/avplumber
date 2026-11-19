"""Backpressure retries and discarded frames have separate queue counters."""

import asyncio
import importlib.util
import json
import socket
import sys
from pathlib import Path
from time import monotonic

import pytest


def test_queue_retries_and_drops(tmp_path, run_script):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native extension")
    result = run_script(Path(__file__).resolve(), tmp_path, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


async def run(directory):
    from pyplumber import AVPlumber
    from pyplumber import node as api
    from pyplumber.mixer.control import AvpConnection

    frame_count = 60
    capacity = 3
    path = Path(directory) / "frames.gray"
    path.write_bytes(bytes(16 * 16 * frame_count))
    host = "127.0.0.1"
    with socket.socket() as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]

    avp = AVPlumber()
    control = AvpConnection(host, port)

    async def snapshot():
        content = await control.command("queues.json")
        return {queue["name"]: queue for queue in json.loads(content)}

    try:
        avp.edges.planCapacity("*", capacity)
        avp.enableControlServer(port, host)
        avp.setReady()
        await asyncio.wait_for(control.connect(), timeout=5)

        for cls, params in (
            (api.InputRec, dict(name="reader", url=str(path), format="rawvideo", dst="packets",
                                options=dict(pixel_format="gray", video_size="16x16", framerate="30"))),
            (api.Demux, dict(name="demux", src="packets", routing={"v:0": "video"})),
            (api.DecVideo, dict(name="decoder", src="video", dst="decoded", codec="rawvideo",
                                pixel_format="gray")),
            (api.Pause, dict(name="pause", src="decoded", dst="paused")),
            (api.Realtime, dict(name="realtime", src="paused", dst="paced", set_pts=False)),
            (api.Split, dict(name="split", src="paced", dst=["lossy", "delivered"], drop=["lossy"])),
        ):
            avp.addNode(cls(dict(params, group="accounting", auto_restart="off")))
        delivered = avp.getEdge("delivered", "VideoFrame")
        lossy = avp.getEdge("lossy", "VideoFrame")
        avp.group("accounting").startNodes()

        # Hold the blocking output until backpressure reaches both retrying nodes.
        retry_edges = ("paused", "paced")
        deadline = monotonic() + 8
        while True:
            queues = await snapshot()
            if all(name in queues for name in retry_edges):
                assert all("retry_total" in queues[name] for name in retry_edges), queues
                if all(queues[name]["retry_total"] > 0 for name in retry_edges):
                    break
            assert monotonic() < deadline, queues
            await asyncio.sleep(0.02)

        pts = []
        deadline = monotonic() + 12
        while len(pts) < frame_count and monotonic() < deadline:
            frame = delivered.tryGet(100)
            if frame is not None:
                pts.append(frame.pts.timestamp)
        assert pts == list(range(frame_count)), pts

        # The split tries the unread output first, so its counts are settled
        # by the time the last frame arrives on the blocking output.
        queues = await snapshot()
        for name in retry_edges:
            assert queues[name]["retry_total"] > 0, queues[name]
            assert queues[name]["dropped_total"] == 0, queues[name]
        assert queues["delivered"]["dropped_total"] == 0, queues["delivered"]
        assert queues["lossy"]["enqueued_total"] == capacity, queues["lossy"]
        assert queues["lossy"]["dropped_total"] == frame_count - capacity, queues["lossy"]
        assert queues["lossy"]["retry_total"] == 0, queues["lossy"]
        kept = [lossy.get(100) for _ in range(capacity)]
        assert [frame.pts.timestamp for frame in kept] == list(range(capacity))
    finally:
        await control.disconnect()
        avp.shutdown()


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
