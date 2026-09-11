"""CUDA/NVENC integration test; use two H.264 solid red/blue input fixtures.

The CPU decode is solely the pixel assertion boundary AFTER NVENC. No download
or upload is inserted into the CUDA decode → mixer → encoder processing path.
Run only in an isolated instance, never against a production mixer's control port.
"""
import argparse
import json
import socket
import statistics
import subprocess
import threading
import time

from avpmixer import MixerGraphBuilder
from pyplumber import AVPlumber
from pyplumber.node import AssumeVideoFormat, DecVideo, Demux, EncVideo, ForceFPS, InputRec, Realtime


def run(args):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.enableControlServer(args.port)
    avp.executeCommandsFromString('hwaccel.init {"name":"cut_test_gpu","type":"cuda"}')
    avp.edges.planCapacity("*", 3)
    for i in range(args.source_count):
        path = args.inputs[i % len(args.inputs)]
        stages = (
            (InputRec, {"url": path, "loop": True, "dst": f"p{i}"}),
            (Demux, {"src": f"p{i}", "routing": {"v:0": f"v{i}"}}),
            (DecVideo, {"src": f"v{i}", "dst": f"d{i}", "pixel_format": "?cuda", "hwaccel": "cut_test_gpu"}),
            (Realtime, {"src": f"d{i}", "dst": f"r{i}", "set_pts": True}),
            (ForceFPS, {"src": f"r{i}", "dst": f"f{i}", "fps": "30/1"}),
        )
        for j, (node, params) in enumerate(stages):
            avp.addNode(node({"name": f"input_{i}_{j}", "group": "inputs", **params}))
    mixer = MixerGraphBuilder(avp, name="cut_test", canvas=(1080, 1920), fps=(30, 1),
                              hwaccel="cut_test_gpu", defer_output=True, enable_wipe=False)
    for i in range(args.source_count):
        mixer.add_source(f"camera{i}", f"f{i}", "inputs", default_graph="")
        mixer.add_scene(f"scene{i}", {f"camera{i}": {
            "dst_x": 0, "dst_y": 0, "dst_w": 1080, "dst_h": 1920, "fit": "contain"}})
    mixer.set_initial_scene("scene0")
    if args.layout_test:
        assert args.source_count == 2
        for i in range(2):
            mixer.define_scene(f"scene{i}", {f"camera{j}": {
                "dst_x": 0 if i == j else 540, "dst_y": 0,
                "dst_w": 540, "dst_h": 1920, "fit": "contain"} for j in range(2)})
    if args.source_count > 2:
        # A larger catalogue shares decoded inputs; definitions are not
        # independently running renderers. Keep two layouts per input.
        for i in range(args.source_count):
            mixer.add_scene(f"grid{i}", {f"camera{j}": {
                "dst_x": (j % 4) * 270, "dst_y": (j // 4) * 480,
                "dst_w": 270, "dst_h": 480, "fit": "contain"}
                for j in range(args.source_count)})
    output = mixer.build()
    for node, params in (
        (ForceFPS, {"name": "test_fps", "src": output, "dst": "test_fps", "fps": "30/1"}),
        (AssumeVideoFormat, {"name": "test_format", "src": "test_fps", "dst": "test_video",
                            "width": 1080, "height": 1920, "pixel_format": "cuda", "real_pixel_format": "nv12"}),
        (EncVideo, {"name": "test_encoder", "src": "test_video", "dst": "test_encoded", "codec": "h264_nvenc",
                    "hwaccel": "cut_test_gpu", "options": {"b": "3000k", "g": 30, "bf": 0,
                    "preset": "p7", "tune": "ull", "rc-lookahead": 0, "zerolatency": 1, "delay": 0}}),
    ):
        avp.addNode(node({"group": "output", **params}))

    frames = []
    stop = threading.Event()
    stream = avp.getEdge("test_encoded", "Packet")

    def read_frames():
        while not stop.is_set():
            try:
                packet = stream.get(100)
            except ValueError:
                continue
            if packet.size <= 0:
                continue
            tb = packet.pts.timebase
            if not tb.num or not tb.den:
                continue  # The shutdown EOF marker is not an encoded picture.
            frames.append({"at": time.monotonic(), "data": packet.data,
                           "pts": round(packet.pts.timestamp * tb.num / tb.den * 30)})

    def wait_for(predicate, label, timeout=15):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert not errors, errors
            assert time.monotonic() < deadline, label
            time.sleep(.005)

    reader = threading.Thread(target=read_frames, daemon=True)
    control = None
    channel = None
    try:
        avp.group("inputs").startNodes()
        wait_for(lambda: all(avp.getEdge(f"f{i}").enqueued_total for i in range(args.source_count)), "input startup")
        mixer.start_groups()
        wait_for(lambda: all(avp.node(name).isWorking for name in (
            "cut_test_comp_a", "cut_test_comp_b", "cut_test_otm_scene_a",
            "cut_test_otm_scene_b", "cut_test_out_sel_transition")), "mixer startup")
        mixer.begin_transition_preheat()
        wait_for(lambda: avp.getEdge("cut_test_trans_out").enqueued_total > 0, "preheat")
        mixer.finish_transition_preheat()
        reader.start()
        avp.group("output").startNodes()
        mixer.start_output()
        wait_for(lambda: len(frames) >= 15, "encoded frames")
        avp.setReady()
        avp.registerWithWebUI(args.webui, "cut-latency-smoke", "")
        control = socket.create_connection(("127.0.0.1", args.port), timeout=5)
        channel = control.makefile("rwb", buffering=0)
        assert channel.readline().startswith(b"100 ")

        def command(line):
            channel.write(line.encode() + b"\n")
            status = channel.readline()
            assert status.startswith((b"200 ", b"201 ")), status
            if status.startswith(b"200 "):
                return None
            body = []
            while True:
                part = channel.readline()
                if part in (b"\n", b"\r\n"):
                    return json.loads(b"".join(body))
                assert part, "control closed"
                body.append(part)

        command('mixer.measurements {"mixer":"cut_test","encoder":"test_encoder"}')
        if args.prewarm:
            command('mixer.prewarm ' + json.dumps({"mixer": "cut_test", "scenes": mixer.scenes()}))
            time.sleep(.5)
        print("CUT_LOAD_BEGIN", flush=True)
        cpu_before = time.process_time()
        load_before = time.monotonic()
        time.sleep(args.hold_seconds)
        load_elapsed = time.monotonic() - load_before
        cpu_percent = 100 * (time.process_time() - cpu_before) / load_elapsed
        print("CUT_LOAD_END " + json.dumps({"prewarm": args.prewarm, "sources": args.source_count,
                                           "scenes": len(mixer.scenes()), "cpu_percent": cpu_percent,
                                           "seconds": load_elapsed}), flush=True)
        results = []
        previous = 0
        for mode in ("direct", "previewed"):
            for _ in range(args.trials):
                target = 1 - previous
                if mode == "previewed":
                    command('mixer.preview ' + json.dumps({"mixer": "cut_test", "scene": f"scene{target}"}))
                    time.sleep(.5)
                # Vary command phase within the 30fps output cycle.
                time.sleep(.15 + (_ % 5) * .007)
                before = time.monotonic()
                command('mixer.cut ' + json.dumps({"mixer": "cut_test", "scene": f"scene{target}"}))
                deadline = time.monotonic() + 5
                while True:
                    sample = command("mixer.status cut_test")["cut_latency"][mode]
                    if sample["state"] == "measured":
                        break
                    assert sample["state"] == "pending", sample
                    assert time.monotonic() < deadline, sample
                    time.sleep(.005)
                wait_for(lambda: any(f["pts"] == sample["encoded_pts"] for f in frames), "matching encoded packet")
                picture = next(f for f in frames if f["pts"] == sample["encoded_pts"])
                observed_ms = (picture["at"] - before) * 1000
                assert sample["scene"] == f"scene{target}", sample
                assert len(sample["recent"]) == min(_ + 1, 3), sample
                assert sample["recent"][-1]["id"] == sample["id"], sample
                assert -5 <= observed_ms - sample["ms"] < 30, (sample, observed_ms)
                results.append({"mode": mode, "avp_ms": sample["ms"], "observed_encoded_ms": observed_ms,
                                "before": before, "target": target, "pts": sample["encoded_pts"]})
                previous = target
        status = command("mixer.status cut_test")["cut_latency"]
        for mode in ("direct", "previewed"):
            assert [s["encoded_pts"] for s in status[mode]["recent"]] == [
                s["pts"] for s in results if s["mode"] == mode][-3:], status
        assert not errors, errors
        captured = tuple(frames)
        # Decode the captured access units independently. A command ACK or an
        # old output packet with a new timestamp cannot satisfy the pixel check.
        decoded = subprocess.run(["/usr/local/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
                                  "-threads", "1", "-f", "h264", "-i", "pipe:0", "-vf",
                                  "crop=2:2:270:960", "-pix_fmt", "gray", "-fps_mode", "passthrough",
                                  "-f", "rawvideo", "pipe:1"], input=b"".join(f["data"] for f in captured),
                                 capture_output=True, check=True, timeout=30).stdout
        assert len(decoded) == len(captured) * 4, (len(decoded), len(captured))
        for i, frame in enumerate(captured):
            y = decoded[i * 4]
            frame["source"] = 0 if 60 <= y <= 95 else 1 if 15 <= y <= 45 else None
        for sample in results:
            first = next(f for f in captured if f["at"] >= sample["before"] and f["source"] == sample["target"])
            assert first["pts"] == sample["pts"], (sample, {k: v for k, v in first.items() if k != "data"})
        if args.prewarm:
            # Geometry edits remain warm. New scene-control semantics instead
            # fall back to normal preparation without rejecting the edit.
            layer = {"dst_x": 0, "dst_y": 0, "dst_w": 1080, "dst_h": 1920, "fit": "contain"}
            mixer.define_scene("scene0", {"camera0": layer})
            assert "scene0" in command("mixer.status cut_test")["prewarm_cut_scenes"]
            mixer.define_scene("scene0", {"camera0": layer}, controls=[
                {"node": "test_encoder", "key": "test_only_not_executed", "value": True}])
            assert "scene0" not in command("mixer.status cut_test")["prewarm_cut_scenes"]
            mixer.define_scene("scene0", {"camera0": layer})
            command('mixer.prewarm {"mixer":"cut_test","scenes":[]}')
            assert command("mixer.status cut_test")["prewarm_source_mask"] == 0
            command('mixer.prewarm ' + json.dumps({"mixer": "cut_test", "scenes": mixer.scenes()}))
            assert len(command("mixer.status cut_test")["prewarm_cut_scenes"]) == len(mixer.scenes())
        print("CUT_LATENCY_RESULT " + json.dumps({"samples": results, "summary": {
            mode: {"median_ms": statistics.median(r["avp_ms"] for r in results if r["mode"] == mode),
                   "count": sum(r["mode"] == mode for r in results)} for mode in ("direct", "previewed")}}), flush=True)
    finally:
        if channel:
            channel.close()
        if control:
            control.close()
        try:
            # Keep draining while the encoder flushes during shutdown.
            avp.shutdown()
        finally:
            stop.set()
            if reader.is_alive():
                reader.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, required=True, metavar=("RED_H264", "BLUE_H264"))
    parser.add_argument("--port", type=int, required=True, help="unused control port for this isolated instance")
    parser.add_argument("--webui", required=True, help="existing WebUI backend; register the isolated graph")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--prewarm", action="store_true", help="retain source queues before direct cuts")
    parser.add_argument("--source-count", type=int, default=2, choices=(2, 16))
    parser.add_argument("--layout-test", action="store_true", help="swap two layers while source set stays fixed")
    parser.add_argument("--hold-seconds", type=float, default=1, help="steady-state load sampling interval")
    run(parser.parse_args())
