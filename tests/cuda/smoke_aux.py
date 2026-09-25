"""Remote GPU smoke: independent aux output, assignments, prewarm and stalled consumer.

Run from the repository root with its CUDA Python module on PYTHONPATH.
Generates two small clips; uses private UDP ports and never changes a live show.
"""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos/mixer"))
from mixer import GraphOptions, build_application, load_avp_api
from pyplumber.node import PythonNode
from pyplumber.mixer.control import AvpConnection


class ConsumerGate(PythonNode):
    blocked = False

    def process(self):
        if self.blocked:
            time.sleep(.01)
        else:
            frame = self._src.get()
            if frame:
                self._dst.enqueue(frame)


def wait_for(predicate, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("aux condition timed out")


def control_json(command):
    async def read():
        connection = AvpConnection("127.0.0.1", 18777)
        await connection.connect()
        try:
            return json.loads(await connection.command(command))
        finally:
            await connection.disconnect()
    return asyncio.run(read())


def subscription_flags():
    queues = control_json("queues.json")
    assert all("subscription_active" not in q for q in queues if q["name"] == "mixer_final_out")
    return {q["name"]: q["subscription_active"] for q in queues if "subscription_active" in q}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, choices=(2, 64, 96), default=2,
                        help="Use tiny independent decoders to exercise the wide aux mask")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="avp-aux-") as directory:
        root = Path(directory)
        sources = []
        for i, color in enumerate(("red", "blue")):
            path = root / f"{color}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=320x180:r=60",
                            "-t", "2", "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p",
                            "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709", str(path)], check=True)
            sources.append({"id": color, "kind": "video", "path": str(path), "color": "sdr"})
        scenes = [{"id": s["id"], "items": [{"source": s["id"], "dst": {"x": 0, "y": 0, "w": 320, "h": 480}}]} for s in sources]
        scenes.append({"id": "repeat", "items": [
            {"source": "red", "dst": {"x": -40, "y": 0, "w": 200, "h": 240}, "fit": "stretch"},
            {"source": "red", "dst": {"x": 160, "y": 240, "w": 160, "h": 240}, "fit": "stretch"}]})
        sources[1:1] = [{**sources[0], "id": f"unused{i}"} for i in range(args.sources - 2)]
        for source in sources:
            source["independent"] = True
        config = {"canvas": {"width": 320, "height": 480, "fps": 60}, "sources": sources, "scenes": scenes,
                  "initial_scene": "red", "renditions": [{"id": "pgm", "target": "janus", "port": 15004}],
                  "aux_buses": [{"id": "mv", "scenes": ["red", "blue", "repeat", None, None, None, None, None],
                                 "renditions": [{"id": "monitor", "port": 15008}]}]}
        path = root / "mixer.json"
        path.write_text(json.dumps(config))
        api = load_avp_api()
        filter_video = api.FilterVideo
        api.FilterVideo = lambda params: filter_video({**params, "src": "aux_test_gated"}
            if params["name"] == "aux_mv_sdr" else params)
        with patch("mixer.load_avp_api", return_value=api):
            app = build_application(GraphOptions(config=str(path), janus_output=True, remote_control_port=18777,
                                                 prewarm_cut_scenes=("*",)))
        app.avp.edges.planCapacity("aux_test_gated", 1)
        gate = ConsumerGate({"name": "aux_test_gate", "src": "aux_mv_out", "dst": "aux_test_gated", "group": "aux_mv"})
        app.avp.addNode(gate)
        errors = []
        app.avp.on_exception = lambda *args: errors.append(args)
        app.avp.registerWithWebUI("http://127.0.0.1:22222", "aux-smoke", "")
        try:
            app.start()
            bus = app.aux_buses[0]
            edge = app.avp.getEdge(bus.output_edge)
            main_edge = app.avp.getEdge("mixer_final_out")
            wait_for(lambda: edge.enqueued_total >= 30)
            expected = {name: name in (bus.edges[0], bus.edges[-1], bus.pgm_edge)
                        for name in [*bus.edges, bus.pgm_edge]}
            assert subscription_flags() == expected
            # Hold a direct cut pending long enough for AUX to poll repeatedly.
            # Its hidden target must not appear as a user-selected preview.
            app.mixer.cut("red")
            wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
            wait_for(lambda: bus.preview == "")
            for preview in ("", "red"):
                if preview:
                    app.mixer.preview(preview)
                    wait_for(lambda: bus.preview == preview)
                app.mixer.cut("blue", start_pts_ms=int(time.monotonic() * 1000) + 500)
                until = time.monotonic() + .25
                while time.monotonic() < until:
                    status = control_json("mixer.status mixer")
                    assert status["pvw_scene"] == preview, status
                    assert bus.state()["pvw_scene"] == preview
                    assert bus.preview == preview
                    time.sleep(.01)
                wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
                assert control_json("mixer.status mixer")["pgm_scene"] == "blue"
                wait_for(lambda: bus.preview == "")
            for scene in ("red", "blue") * 10:
                app.mixer.cut(scene)
                assert bus.state()["pvw_scene"] == ""
                assert bus.preview == ""
                time.sleep(.01)
            wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
            # An unavailable new input must leave the old AUX running, then
            # release the abandoned subscription. A newer request cancels it.
            app.mixer.preview("red")
            wait_for(lambda: bus.state()["pvw_scene"] == "red")
            def assign(scenes):
                bus.assign({"expected_revision": bus.state()["revision"], "scenes": scenes})
            assign(["red"] * 8)
            wait_for(lambda: not subscription_flags()[bus.edges[-1]])
            blue_fps = app.avp.node(f"fps_{args.sources - 1}")
            blue_fps.stopAndWait()
            time.sleep(.3)
            before = edge.enqueued_total
            assign(["blue"] * 8)
            wait_for(lambda: bool(bus.state().get("composition_error")))
            assert not bus.state()["suspended"]
            assert edge.enqueued_total - before >= 3
            assert not subscription_flags()[bus.edges[-1]]
            assign(["blue"] * 8)
            assign(["red"] * 8)
            wait_for(lambda: not bus.state()["composition_pending"])
            assert not bus.state()["composition_error"]
            assert not subscription_flags()[bus.edges[-1]]
            blue_fps.start()
            assign(["blue"] * 8)
            wait_for(lambda: not bus.state()["composition_pending"])
            assert not bus.state()["composition_error"]
            assert subscription_flags()[bus.edges[-1]]
            assign(list(config["aux_buses"][0]["scenes"]))
            wait_for(lambda: not bus.state()["composition_pending"])
            app.mixer.preview("blue")
            wait_for(lambda: bus.state()["pvw_scene"] == "blue")
            initial = bus.state()
            bus.assign({"expected_revision": initial["revision"], "scenes": ["repeat"] * 8})
            assert bus.assign({"expected_revision": initial["revision"], "scenes": [None] * 8})["conflict"]
            app.mixer.cut("blue")
            app.mixer.preview("red")
            expected[bus.edges[-1]] = False
            wait_for(lambda: subscription_flags() == expected)
            time.sleep(1)
            # Stall the first consumer: the compositor must suspend, unsubscribe
            # every input, and let the main program keep advancing.
            gate.blocked = True
            wait_for(lambda: bus.state()["suspended"])
            assert subscription_flags() == {name: False for name in [*bus.edges, bus.pgm_edge]}
            before = main_edge.enqueued_total
            time.sleep(1)
            assert main_edge.enqueued_total - before >= 40
            assert all(app.avp.getEdge(e).occupied == 0 for e in [*bus.edges, bus.pgm_edge])
            gate.blocked = False
            current = bus.state()
            before = edge.enqueued_total
            bus.assign({"expected_revision": current["revision"], "scenes": current["scenes"]})
            wait_for(lambda: edge.enqueued_total >= before + 15)
            assert subscription_flags() == expected
            # Restart only the aux compositor. The main
            # program and the aux encoder must survive this independently.
            app.avp.node(bus.node_name).stopAndWait()
            app.avp.node(bus.node_name).start()
            before = edge.enqueued_total
            current = bus.state()
            bus.assign({"expected_revision": current["revision"], "scenes": current["scenes"]})
            wait_for(lambda: edge.enqueued_total >= before + 30)
            assert not errors, errors
            print("PASS: aux staging, timeout, cancellation, cadence, repeated pads, preview, revision conflict, stalled consumer and PGM continuity", flush=True)
        finally:
            gate.blocked = False
            app.stop()


if __name__ == "__main__":
    main()
