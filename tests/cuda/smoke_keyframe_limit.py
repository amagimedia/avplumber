"""Isolated 16-input, 60-fps mixer/NVENC keyframe-spam regression.

Uses the demo's actual Janus encoder graph but terminates at an encoded-packet
reader, never a live Janus mountpoint. CPU decoding is only the post-encode
pixel oracle. No CPU/GPU round trip is added to the mixer processing graph.
Run separate processes with --minimum-ms 0 and 150 for the A/B comparison.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos" / "mixer"))
from mixer import GraphOptions, build_application


def run(args):
    stop = threading.Event()
    app = build_application(GraphOptions(
        inputs=tuple(args.inputs[i % 2] for i in range(args.source_count)),
        fps=60, loop_inputs=True, janus_output=True,
        janus_video_bitrate_kbps=2700, keyframe_min_interval_ms=args.minimum_ms,
        remote_control_port=args.port, wipe_cache_mb=0,
        prewarm_cut_scenes=("*",),
    ))
    # The owned test ends at the encoder; leave RTP/BSF lifecycle to its own
    # tests, and never publish this graph to the demo's streaming mountpoint.
    app.avp.executeCommandsFromString("node.delete janus_rtp_output\n"
                                     "node.delete janus_mux\nnode.delete janus_repeat_headers")
    app.rtcp_feedback_listener = None
    errors, frames, cuts = [], [], []
    app.avp.on_exception = lambda *error: errors.append(error)

    def capture(packet):
        if packet.size <= 0 or packet.pts.timestamp == -(1 << 63):
            return
        tb = packet.pts.timebase
        frames.append({"at": time.monotonic(), "pts": round(packet.pts.timestamp * tb.num / tb.den * 60),
                       "key": bool(packet.flags & 1), "data": packet.data})

    # A queue reader avoids Python callbacks from a native encoder thread
    # during shutdown (shutdown holds the GIL while joining native workers).
    encoded = app.avp.getEdge("janus_encoded", "Packet")

    def read_encoded():
        while not stop.is_set():
            packet = encoded.tryGet(100)
            if packet is not None:
                capture(packet)

    packet_reader = threading.Thread(target=read_encoded, daemon=True)
    packet_reader.start()

    def wait_for(predicate, label, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert not errors, errors
            assert time.monotonic() < deadline, label
            time.sleep(.005)

    try:
        app.start()
        app.avp.registerWithWebUI(args.webui, "keyframe-limit-smoke", "")
        wait_for(lambda: len(frames) >= 60, "encoder startup")
        state = app.avp.node("janus_force_keyframe").getObject("status")
        assert state["min_interval_ms"] == args.minimum_ms, state
        begin = time.monotonic()
        for index in range(args.cuts):
            target = 1 - index % 2
            cuts.append({"at": time.monotonic(), "target": target})
            app.mixer.cut(f"fullscreen_{target}")
            time.sleep(.05)
        # Let the final cut/keyframe request finish after spam stops.
        time.sleep(.4)
        assert not app.avp.node("janus_force_keyframe").getObject("pending")
        cut_end = time.monotonic()
        # Exercise interrupted crossfades, then a fully completed fade.
        for index in range(12):
            app.mixer.fade(f"fullscreen_{index % 2}", duration_sec=.25)
            time.sleep(.075)
        app.mixer.fade("fullscreen_0", duration_sec=.25)
        time.sleep(.6)
        end = time.monotonic()
        state = app.avp.node("janus_force_keyframe").getObject("status")
        assert not state["pending"], state
        assert not errors, errors
        captured = tuple(frames)
        measured = [f for f in captured if begin <= f["at"] <= end]
        pts = [f["pts"] for f in measured]
        assert all(b - a == 1 for a, b in zip(pts, pts[1:])), "output frame gap or duplicate PTS"
        keys = [f["pts"] for f in captured if f["key"]]
        key_spacing = [b - a for a, b in zip(keys, keys[1:])]
        assert len(key_spacing) >= 3, "insufficient keyframes"
        assert all(g * 1000 >= args.minimum_ms * 60 for g in key_spacing), key_spacing
        gaps = [(b["at"] - a["at"]) * 1000 for a, b in zip(measured, measured[1:])]
        assert max(gaps) < 150, ("encoded output stalled", max(gaps))

        # Red/blue solid input fixtures make each cut's actual first picture
        # distinguishable. This also proves that cuts need not wait for an IDR.
        decoded = subprocess.run([
            args.ffmpeg, "-v", "error", "-threads", "1", "-f", "h264", "-i", "pipe:0",
            "-vf", "crop=2:2:540:960", "-pix_fmt", "gray", "-fps_mode", "passthrough",
            "-f", "rawvideo", "pipe:1",
        ], input=b"".join(f["data"] for f in captured), capture_output=True, check=True, timeout=30).stdout
        assert len(decoded) == len(captured) * 4, (len(decoded), len(captured))
        for i, frame in enumerate(captured):
            y = decoded[i * 4]
            frame["source"] = 0 if 60 <= y <= 95 else 1 if 15 <= y <= 45 else None
        latencies, non_key_cuts = [], 0
        for i, cut in enumerate(cuts):
            deadline = cuts[i + 1]["at"] if i + 1 < len(cuts) else cut_end
            matches = [f for f in captured if cut["at"] <= f["at"] < deadline
                       and f["source"] == cut["target"]]
            assert matches, ("cut picture did not arrive before the next cut", i, cut)
            latencies.append((matches[0]["at"] - cut["at"]) * 1000)
            non_key_cuts += not matches[0]["key"]
        if args.minimum_ms:
            assert non_key_cuts > len(cuts) // 2, "cuts appear to wait for keyframes"
        print("KEYFRAME_LIMIT_RESULT " + json.dumps({
            "minimum_ms": args.minimum_ms, "inputs": args.source_count,
            "cuts": len(cuts), "cut_pictures_without_keyframe": non_key_cuts,
            "cut_latency_ms": {"median": statistics.median(latencies), "max": max(latencies)},
            "frames": len(measured), "fps": (len(measured) - 1) / (measured[-1]["at"] - measured[0]["at"]),
            "encoded_gap_ms": {"p99": sorted(gaps)[int(.99 * (len(gaps) - 1))], "max": max(gaps)},
            "keyframes": len(keys), "min_keyframe_spacing_frames": min(key_spacing),
            "status": state,
        }), flush=True)
    finally:
        app.stop()
        stop.set()
        packet_reader.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, required=True, metavar=("RED_H264", "BLUE_H264"))
    parser.add_argument("--port", type=int, required=True, help="unused control port for this isolated graph")
    parser.add_argument("--webui", required=True, help="existing WebUI backend")
    parser.add_argument("--source-count", type=int, choices=(2, 16), default=16)
    parser.add_argument("--minimum-ms", type=int, choices=(0, 100, 150, 200), default=150)
    parser.add_argument("--cuts", type=int, default=80)
    parser.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    run(parser.parse_args())
