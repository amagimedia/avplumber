"""Compare MPEG-TS file EOF with explicit EOF on a still-open TCP publisher.

Run remotely with the native module under an external timeout. Supply a short
H.264/AAC TS fixture: this test does no encoding and needs no GPU. It compares
every elementary packet, including those held in the final PES/parser buffers.
"""

import argparse
import hashlib
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path

from pyplumber import AVPlumber
from pyplumber.node import InputRec


def capture(url, sent=None):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *args: errors.append(args)
    packets = defaultdict(list)
    avp.executeCommandsFromString("queue.plan_capacity packets 2\n")
    avp.addNode(InputRec({"name": "input", "group": "test", "dst": "packets",
                         "url": url, "format": "mpegts", "timeout": 10,
                         "options": {"analyzeduration": "250000", "probesize": "32768"}}))
    eof_count = 0
    requested = False
    before_drain = None
    try:
        avp.group("test").startNodes()
        edge = avp.getEdge("packets", "Packet")
        # Exercise normal bounded-queue backpressure before releasing consumers.
        time.sleep(0.05)
        deadline = time.monotonic() + 15
        last_packet = time.monotonic()
        while avp.node("input").isWorking or edge.occupied:
            assert time.monotonic() < deadline, "Input did not finish bounded drain"
            assert not errors, errors
            packet = edge.tryGet(20)
            if packet is not None:
                last_packet = time.monotonic()
                if packet.size == 1 and packet.data == b"\xff":
                    eof_count += 1
                else:
                    assert eof_count == 0, "Media appeared after EOF"
                    packets[packet.stream_index].append((
                        packet.pts.timestamp, packet.dts.timestamp,
                        packet.size, hashlib.sha256(packet.data).hexdigest()))
            if (sent is not None and sent.is_set() and not requested
                    and time.monotonic() - last_packet > 0.2):
                # The publisher is still connected. The input has consumed all
                # available bytes, but cannot flush its final PES/parser yet.
                before_drain = sum(map(len, packets.values()))
                avp.executeCommandsFromString("node.object.set input request-eof true\n")
                # The command is idempotent while drain remains in progress.
                avp.executeCommandsFromString("node.object.set input request-eof true\n")
                requested = True
        assert not errors, errors
        assert eof_count == 1, f"Expected exactly one EOF, got {eof_count}"
        if sent is not None:
            assert requested
            assert before_drain < sum(map(len, packets.values())), "Fixture did not exercise parser tail"
        return dict(packets)
    finally:
        avp.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    expected = capture(str(args.input.resolve()))
    assert len(expected) >= 2, "Fixture must contain audio and video"
    sent, release = threading.Event(), threading.Event()
    failures = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def publish():
            try:
                with listener.accept()[0] as client:
                    with args.input.open("rb") as source:
                        while block := source.read(64 * 1024):
                            client.sendall(block)
                    sent.set()
                    assert release.wait(30), "Test did not release publisher"
            except Exception as exc:
                failures.append(exc)

        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        try:
            actual = capture(f"tcp://127.0.0.1:{listener.getsockname()[1]}", sent)
            assert actual == expected, {
                "file_packets": {key: len(value) for key, value in expected.items()},
                "live_packets": {key: len(value) for key, value in actual.items()},
            }
        finally:
            release.set()
            worker.join(5)
        assert not failures, failures
    print("Live input parser drain matches file EOF:",
          {key: len(value) for key, value in expected.items()}, flush=True)


if __name__ == "__main__":
    main()
