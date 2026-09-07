"""Native Demux lifecycle checks; subprocess deadlines bound shutdown failures."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest


@pytest.fixture(scope="module")
def demux_recording(tmp_path_factory):
    if importlib.util.find_spec("_avplumber") is None or not shutil.which("ffmpeg"):
        pytest.skip("requires native avplumber and FFmpeg")
    path = tmp_path_factory.mktemp("demux") / "two-streams.nut"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=16x16:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "0.2", "-c:v", "rawvideo", "-threads:v", "1",
        "-c:a", "pcm_s16le", "-f", "nut", str(path),
    ], check=True, capture_output=True, timeout=15)
    return path


@pytest.mark.parametrize("case", [
    "blocked_read", "stop_eof_interleave", "blocked_eof_output",
    "ordinary_streams", "retained_eof", "repeated_stop",
])
def test_demux_lifecycle(case, demux_recording):
    attempts = 8 if case in {"blocked_read", "stop_eof_interleave"} else 1
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), case, str(demux_recording)],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired as error:
            pytest.fail(f"native Demux {case} exceeded shutdown deadline: {error.stdout!r}")
        assert result.returncode == 0, f"attempt {attempt + 1}: {result.stdout}{result.stderr}"
        assert f"PASS {case}" in result.stdout


def _wait(predicate, description):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, description
        time.sleep(0.001)


def _run(case, recording):
    from pyplumber import AVPlumber
    from pyplumber.node import Demux, Input

    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 64)
    avp.addNode(Input({
        "name": "fixture_reader", "url": recording, "dst": "fixture_packets",
        "eof_mode": "drain", "auto_restart": "off",
    }), early_create=True, start=True)
    _wait(lambda: not avp.node("fixture_reader").isWorking, "fixture read did not finish")
    fixture_edge = avp.getEdge("fixture_packets", "Packet")
    fixture_packets = []
    while fixture_edge.occupied:
        fixture_packets.append(fixture_edge.get(0))
    eof = fixture_packets[-1]
    assert eof.size == 1 and eof.data == b"\xff", "fixture EOF marker missing"
    if case == "blocked_eof_output":
        avp.edges.planCapacity("video", 1)
        avp.edges.planCapacity("audio", 1)
    avp.addNode(Input({
        "name": "input", "url": recording, "dst": "packets", "eof_mode": "drain",
        "auto_restart": "off",
    }), early_create=True)
    params = {"name": "demux", "src": "packets",
              "routing": {"v:0": "video", "a:0": "audio"},
              "wait_for_keyframe": False, "auto_restart": "off"}
    if case != "ordinary_streams":
        params["stop_on_eof"] = False
    avp.addNode(Demux(params), early_create=True)
    node = avp.node("demux")
    incoming = avp.getEdge("packets", "Packet")
    outputs = [avp.getEdge(name, "Packet") for name in ("video", "audio")]
    observed = [[], []]
    for index, edge in enumerate(outputs):
        edge.addWiretapCallback(lambda packet, i=index: observed[i].append(packet))

    entered, release = threading.Event(), threading.Event()
    if case == "stop_eof_interleave":
        # Hold a real EOF at its forwarding boundary. While the worker cannot
        # read its input, drain that queue through the public edge API after
        # stop. Releasing EOF must not resurrect the stopped demux.
        def hold_eof(packet):
            if packet.size == 1 and not entered.is_set():
                entered.set()
                assert release.wait(3), "EOF barrier was not released"
        for edge in outputs:
            edge.addWiretapCallback(hold_eof)

    if case == "blocked_eof_output":
        for edge in outputs:
            edge.enqueue(eof)
    node.start()
    _wait(lambda: node.isWorking, "demux did not start")
    if case == "ordinary_streams":
        original = []
        incoming.addWiretapCallback(lambda packet: original.append(packet))
        avp.node("input").start()
        _wait(lambda: not node.isWorking, "ordinary EOF did not finish demux")
        for stream, packets in enumerate(observed):
            expected = [p.data for p in original if p.size > 1 and p.stream_index == stream]
            assert expected, f"fixture stream {stream} produced no packets"
            assert [p.data for p in packets if p.size > 1] == expected
            assert sum(p.size == 1 for p in packets) == 1
            assert all(p.stream_index == stream for p in packets)
    elif case == "retained_eof":
        for count in (1, 2):
            incoming.enqueue(eof)
            _wait(lambda: all(len(items) == count for items in observed),
                  "EOF was not forwarded to both streams")
            assert node.isWorking, "interactive demux stopped at EOF"
        # Resume real packets after EOF without recreating the demux.
        avp.node("input").start()
        _wait(lambda: all(any(p.size > 1 for p in items) for items in observed),
              "packets after EOF were not forwarded")
        _wait(lambda: all(sum(p.size == 1 for p in items) == 3 for items in observed),
              "resumed input EOF was not forwarded")
        assert node.isWorking
        node.stop(False)
    elif case == "blocked_eof_output":
        incoming.enqueue(eof)
        _wait(lambda: incoming.occupied == 0, "demux did not consume EOF")
        assert all(edge.occupied == 1 for edge in outputs)
        node.stop(False)
    else:
        # Give the native worker time to enter the empty packet read. The EOF
        # callback above then provides a barrier for the dangerous interleaving.
        time.sleep(0.03)
        if case == "stop_eof_interleave":
            incoming.enqueue(eof)
            assert entered.wait(1), "demux did not reach the EOF forwarding barrier"
        node.stop(False)
        if case == "stop_eof_interleave":
            assert incoming.tryGet(0) is None
            release.set()
        if case == "repeated_stop":
            node.stop(False)
    _wait(lambda: not node.isWorking, "demux did not stop after its wake-up")
    node.join()
    node.stop(False)
    assert not errors, errors
    avp.shutdown()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    # A failed child must not hang again while destructing the stuck graph.
    import traceback
    try:
        _run(*sys.argv[1:])
    except BaseException:
        traceback.print_exc()
        os._exit(1)
