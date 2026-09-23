"""Remote GPU smoke: independent aux output, assignments, prewarm and stalled consumer.

Run from the repository root with its CUDA Python module on PYTHONPATH.
Generates two small clips; uses private UDP ports and never changes a live show.
"""
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


def main():
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
            app.mixer.preview("blue")
            wait_for(lambda: bus.state()["pvw_scene"] == "blue")
            initial = bus.state()
            bus.assign({"expected_revision": initial["revision"], "scenes": ["repeat"] * 8})
            assert bus.assign({"expected_revision": initial["revision"], "scenes": [None] * 8})["conflict"]
            app.mixer.cut("blue")
            time.sleep(1)
            # Stall the first consumer: the compositor must suspend, unsubscribe
            # every input, and let the main program keep advancing.
            gate.blocked = True
            wait_for(lambda: bus.state()["suspended"])
            before = main_edge.enqueued_total
            time.sleep(1)
            assert main_edge.enqueued_total - before >= 40
            assert all(app.avp.getEdge(e).occupied == 0 for e in [*bus.edges, bus.pgm_edge])
            gate.blocked = False
            current = bus.state()
            before = edge.enqueued_total
            bus.assign({"expected_revision": current["revision"], "scenes": current["scenes"]})
            wait_for(lambda: edge.enqueued_total >= before + 15)
            # Restart only the aux compositor. The main
            # program and the aux encoder must survive this independently.
            app.avp.node(bus.node_name).stopAndWait()
            app.avp.node(bus.node_name).start()
            before = edge.enqueued_total
            current = bus.state()
            bus.assign({"expected_revision": current["revision"], "scenes": current["scenes"]})
            wait_for(lambda: edge.enqueued_total >= before + 30)
            assert not errors, errors
            print("PASS: aux cadence, repeated pads, preview, revision conflict, stalled consumer, empty queues and PGM continuity", flush=True)
        finally:
            gate.blocked = False
            app.stop()


if __name__ == "__main__":
    main()
