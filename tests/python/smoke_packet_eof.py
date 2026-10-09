"""Run under an external timeout; two input EOFs must finish one mux output."""

import time

from _avplumber import Packet
from pyplumber import AVPlumber
from pyplumber.node import Mux

marker = Packet.eof()
assert marker.size == 1
assert marker.data == b"\xff"
assert marker.pts.timestamp == Packet().pts.timestamp
assert Packet().size == 0

avp = AVPlumber()
avp.addNode(Mux({"name": "mux", "group": "test", "src": ["video", "audio"],
                 "dst": "merged", "allow_no_encoder": True}))
avp.group("test").startNodes()
avp.getEdge("video", "Packet").enqueue(marker)
time.sleep(0.05)
assert avp.node("mux").isWorking, "Video EOF must not discard unfinished audio"
assert avp.getEdge("merged", "Packet").occupied == 0
avp.getEdge("audio", "Packet").enqueue(Packet.eof())
deadline = time.monotonic() + 5
while avp.node("mux").isWorking and time.monotonic() < deadline:
    time.sleep(0.01)
assert not avp.node("mux").isWorking, "Mux did not finish after both EOFs"
merged = avp.getEdge("merged", "Packet")
assert merged.occupied == 1
assert merged.tryGet(0).data == b"\xff"
avp.shutdown()
print("Packet EOF and mux drain passed", flush=True)
