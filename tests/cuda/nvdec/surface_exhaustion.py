"""Surface exhaustion on a running show, on the NVIDIA host.

Two CUarray decoders of the same clip, with little headroom. One decoder's consumer stops
taking frames for --stall-ms and holds what it has (more than the pool), as a stalled
output does; the other's consumer keeps going, on a thread of its own. Expected, and checked:

- the stalled decoder runs out of surfaces and stays alive: no error, no restart, its
  packets wait, its `status` counts the episode (gate pauses, then `exhausted_now`);
- the other decoder keeps delivering above real time (--fps) throughout: nothing paces the
  consumers here, so the baseline rate is the decoder's top speed, and the report gives the
  rate in every phase for comparison;
- when the consumer lets go, the stalled decoder delivers again, from a keyframe, and its
  `status` shows the recovery and no degradation.

    python3 tests/cuda/nvdec/surface_exhaustion.py --input CLIP --fps 30 [--extra-hw-frames 1] [--stall-ms 3000]

The clip must be looped or longer than the baseline, the stall and the recovery together.
Cover HEVC and H.264 clips: their decoders lose a picture differently (HEVC reports the
shortage and the packet is sent again; H.264 goes on without the surface and the picture is
lost), and this check must hold for both.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import EOF, finish, make_avp
from pyplumber import node as api


def chain(nodes, tag, path, extra_hw_frames):
    nodes += [
        api.InputRec({"name": f"input_{tag}", "url": str(path), "dst": f"packets_{tag}", "loop": True}),
        api.Demux({"name": f"demux_{tag}", "src": f"packets_{tag}", "routing": {"v:0": f"video_{tag}"}}),
        api.DecVideo({"name": f"decode_{tag}", "src": f"video_{tag}", "dst": f"decoded_{tag}",
                      "hwaccel": "exhaust_gpu", "pixel_format": "cuarray",
                      "options": {"threads": 1, "hwaccel_flags": "unsafe_output", "extra_hw_frames": extra_hw_frames}}),
    ]
    return f"decoded_{tag}"


def status(avp, name):
    try:
        return avp.node(name).getObject("status")
    except RuntimeError:   # the group is still creating its nodes, or the node is gone
        return None


class Consumer:
    """Takes frames off one edge on its own thread, counting them; while `hold` is set it keeps
    them, as a stalled output keeps what it has."""

    def __init__(self, edge):
        self.edge, self.count, self.hold, self.held = edge, 0, False, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            frame = self.edge.tryGet(20)
            if frame is None:
                continue
            if frame.pts.timestamp != EOF:
                self.count += 1
                if self.hold:
                    self.held.append(frame)
            del frame

    def release(self):
        self.hold = False
        self.held.clear()

    def close(self):
        self.stop.set()
        self.thread.join(5)
        self.held.clear()


def run(args):
    avp, errors = make_avp("exhaust_gpu", capacity=3)
    nodes = []
    report = {"ok": False, "input": str(args.input), "fps": args.fps, "extra_hw_frames": args.extra_hw_frames,
              "stall_ms": args.stall_ms, "phases": {}, "errors": errors}
    consumers = {}
    try:
        edges = {tag: chain(nodes, tag, args.input, args.extra_hw_frames) for tag in ("a", "b")}
        for node in nodes:
            node.parameters.update(group="exhaust", auto_restart="off", on_error="off")
            avp.addNode(node)
        consumers = {tag: Consumer(avp.getEdge(edge, "VideoFrame")) for tag, edge in edges.items()}
        avp.group("exhaust").startNodes()

        def phase(name, seconds):
            before = {tag: consumer.count for tag, consumer in consumers.items()}
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                assert not errors, errors
                time.sleep(0.05)
            delivered = {tag: consumer.count - before[tag] for tag, consumer in consumers.items()}
            report["phases"][name] = {"seconds": seconds, "delivered": delivered,
                                      "fps": {tag: round(n / seconds, 1) for tag, n in delivered.items()},
                                      "held_a": len(consumers["a"].held),
                                      "status_a": status(avp, "decode_a"), "status_b": status(avp, "decode_b")}
            return delivered

        real_time = 0.8 * args.fps

        # Both flow: the rate the other decoder must keep is real time, whatever its top speed.
        baseline = phase("baseline", args.baseline_s)
        assert baseline["a"] >= real_time * args.baseline_s and baseline["b"] >= real_time * args.baseline_s, baseline

        # A holds everything it gets: its pool runs dry; B must not notice.
        consumers["a"].hold = True
        stalled = phase("stall", args.stall_ms / 1000)
        status_a = report["phases"]["stall"]["status_a"]
        surfaces = status_a["surfaces"]
        assert surfaces["pauses"] >= 1 or surfaces["exhaustions"] >= 1, status_a
        assert surfaces["exhausted_now"] or surfaces["exhaustions"] >= 1 or len(consumers["a"].held) >= surfaces["pool"], \
            (len(consumers["a"].held), status_a)
        assert stalled["b"] >= real_time * args.stall_ms / 1000, (stalled, report["phases"]["stall"]["fps"])
        assert not errors, errors

        # A lets go: the decoder must come back on its own, from a keyframe, and not be degraded.
        consumers["a"].release()
        recovered = phase("recovery", args.recovery_s)
        status_a = report["phases"]["recovery"]["status_a"]
        assert recovered["a"] >= 10, (recovered, status_a)
        assert recovered["b"] >= real_time * args.recovery_s, (recovered, report["phases"]["recovery"]["fps"])
        assert not status_a["degraded"] and not status_a["surfaces"]["exhausted_now"], status_a
        if status_a["surfaces"]["exhaustions"]:
            assert status_a["surfaces"]["recoveries"] >= 1, status_a
        assert not status_a["surfaces"]["hiding"], status_a
        assert not errors, errors
        report["ok"] = True
    except AssertionError as error:
        report["failure"] = str(error)
    finally:
        for consumer in consumers.values():
            consumer.close()
        finish(avp, nodes, timeout=20)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "phases"}, indent=1))
    for name, data in report["phases"].items():
        print(name, json.dumps(data["fps"]), "held_a", data["held_a"], json.dumps(data["status_a"]))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="an H.264 or HEVC clip the decoder loops")
    parser.add_argument("--fps", type=float, default=30.0, help="the clip's rate: real time for the neighbour")
    parser.add_argument("--extra-hw-frames", type=int, default=1)
    parser.add_argument("--baseline-s", type=float, default=3.0)
    parser.add_argument("--stall-ms", type=int, default=3000)
    parser.add_argument("--recovery-s", type=float, default=5.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.extra_hw_frames < 0 or args.stall_ms < 1500 or args.baseline_s <= 0 or args.recovery_s <= 0 or args.fps <= 0:
        parser.error("extra frames nonnegative, stall at least 1500 ms (the gate gives up after one second), positive phases and fps")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
