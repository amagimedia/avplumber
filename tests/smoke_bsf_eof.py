"""Native regression: a bounded input through encoder -> bsf -> mux must end cleanly.

Run in the built avplumber environment:
    python3 tests/smoke_bsf_eof.py <short-video-file> <output.nut>

Before the fix the bitstream-filter node pushed everything it received into
libavcodec: the engine's 1-byte EOF marker, and the empty packet an encoder's flush
can end with. libavcodec takes an empty packet as end of stream and refuses the next
one with EINVAL ("A non-NULL packet sent after an EOF"), which panicked the node at
the end of every bounded input; the marker itself became a bogus packet downstream.
"""

import json
import subprocess
import sys
import time

from pyplumber import AVPlumber
from pyplumber import node as n


def run(path, output):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 4096)
    for cls, params in (
        (n.InputRec, {"group": "input", "name": "input", "url": path, "dst": "packets"}),
        (n.Demux, {"group": "input", "name": "demux", "src": "packets",
                   "routing": {"v:0": "selected"}}),
        (n.DecVideo, {"group": "input", "name": "decode", "src": "selected", "dst": "decoded"}),
        # The declaration gives the encoder and the output their time base.
        (n.FakeVideoFormat, {"group": "output", "name": "format", "src": "decoded",
                             "dst": "declared"}),
        (n.EncVideo, {"group": "output", "name": "encoder", "src": "declared", "dst": "encoded",
                      "codec": "mpeg2video"}),
        (n.Bsf, {"group": "output", "name": "bsf", "src": "encoded", "dst": "filtered",
                 "bsf": "dump_extra=freq=keyframe"}),
        (n.Mux, {"group": "output", "name": "mux", "src": ["filtered"], "dst": "muxed"}),
        (n.Output, {"group": "output", "name": "out", "src": "muxed", "url": output}),
    ):
        avp.addNode(cls(params))
    encoded = avp.getEdge("encoded", "Packet")
    try:
        avp.group("output").startNodes()
        avp.group("input").startNodes()
        deadline = time.monotonic() + 30
        while avp.node("out").isWorking or avp.node("bsf").isWorking:
            assert not errors, errors
            assert time.monotonic() < deadline, "bounded input did not drain"
            time.sleep(0.02)
        assert not errors, errors
        # The marker and any empty flush packet never reach the muxer, so the file
        # holds exactly the encoder's packets (the marker is counted on the edge).
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "json", output],
            capture_output=True, text=True, check=True)
        muxed = int(json.loads(probe.stdout)["streams"][0]["nb_read_packets"])
        produced = encoded.enqueued_total - 1
        assert muxed == produced, f"muxed {muxed} packets, encoder produced {produced}"
        print(f"PASS bsf_eof packets={muxed}", flush=True)
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run(*sys.argv[1:])
