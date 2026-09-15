"""Native regression: decode a short A/V fixture, reach EOF, then reconnect.

Run in the built avplumber environment:
    python3 tests/smoke_format_eof.py <short-av-file> <video|audio> <live|finite>
"""

import sys
import time

from pyplumber import AVPlumber
from pyplumber import node as n


def run(path, media, live):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    video = media == "video"
    data_type = "VideoFrame" if video else "AudioSamples"
    decoder = n.DecVideo if video else n.DecAudio
    declaration = n.FakeVideoFormat if video else n.FakeAudioMetadata
    avp.edges.planCapacity("*", 4096)
    for cls, params in (
        (n.InputRec, {"name": "input", "url": path, "dst": "packets"}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": {
            "v:0" if video else "a:0": "selected"}}),
        (decoder, {"name": "decode", "src": "selected", "dst": "decoded"}),
    ):
        avp.addNode(cls({"group": "input", **params}))
    params = {"name": "format", "group": "output", "src": "decoded", "dst": "result"}
    if live:
        params["ignore_eof"] = True
    avp.addNode(declaration(params))
    source = avp.getEdge("decoded", data_type)
    result = avp.getEdge("result", data_type)
    passes = []
    try:
        avp.group("output").startNodes()
        for attempt in range(2 if live else 1):
            previous_input = source.enqueued_total
            if attempt:
                avp.group("input").restartNodes()
            else:
                avp.group("input").startNodes()
            deadline = time.monotonic() + 15
            while True:
                assert not errors, errors
                assert time.monotonic() < deadline, "decode/EOF did not complete"
                if (source.enqueued_total > previous_input
                        and not avp.node("decode").isWorking
                        and source.occupied == 0):
                    if live or not avp.node("format").isWorking:
                        break
                time.sleep(0.01)
            rows = []
            eof = 0
            while result.occupied:
                frame = result.wait_dequeue()
                if frame.pts.timestamp == -(1 << 63):
                    eof += 1
                else:
                    rows.append((frame.pts.timestamp, frame.pts.timebase.num,
                                 frame.pts.timebase.den))
            assert rows, "no decoded media reached the output"
            assert eof == (0 if live else 1), (live, eof)
            assert avp.node("format").isWorking == live
            passes.append(rows)
        if live:
            assert passes[0] == passes[1], "reconnect changed or lost decoded frames"
        print(f"PASS {media} {'live reconnect' if live else 'default finite EOF'}: "
              f"{len(passes[0])} frames per pass", flush=True)
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] == "live")
