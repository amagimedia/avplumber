"""Check the native AVP input and demux nodes against a live NDI audio/video sender."""
import argparse
import time

from pyplumber import AVPlumber, node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="NDI source name as reported by find_sources")
    parser.add_argument("--extra-ips", help="additional addresses for source discovery")
    args = parser.parse_args()
    options = {"wait_sources": "5"}
    if args.extra_ips:
        options["extra_ips"] = args.extra_ips

    engine = AVPlumber()
    errors = []
    engine.on_exception = lambda *error: errors.append(error)
    engine.edges.planCapacity("*", 128)
    engine.addNode(node.Input({"group": "ndi", "name": "receive", "dst": "packets",
                               "format": "libndi_newtek", "url": args.source,
                               "options": options, "initial_timeout": 12}))
    engine.addNode(node.Demux({"group": "ndi", "name": "split", "src": "packets",
                               "routing": {"v:0": "video", "a:0": "audio"}}))
    edges = {name: engine.getEdge(name, "Packet") for name in ("audio", "video")}
    timestamps = {name: [] for name in edges}
    try:
        engine.group("ndi").startNodes()
        deadline = time.monotonic() + 20
        while any(len(values) < 15 for values in timestamps.values()):
            assert not errors, errors
            assert time.monotonic() < deadline, "timed out waiting for NDI audio and video"
            for name, edge in edges.items():
                packet = edge.tryGet(30)
                if packet is not None:
                    timestamps[name].append(packet.pts.timestamp)
        for name, values in timestamps.items():
            assert all(a < b for a, b in zip(values, values[1:])), (name, values)
            print(f"{name}: {len(values)} packets with increasing PTS")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
