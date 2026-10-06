"""Remote GPU smoke: independent aux output, assignments, layout switches, prewarm and stalled consumer.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _harness import wait_for
from pyplumber.mixer.cli import GraphOptions, build_application, load_avp_api
from pyplumber.node import PythonNode
from pyplumber.mixer.config import ConfigError
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
    parser.add_argument("--webui", default="http://127.0.0.1:22222", help="existing WebUI backend")
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
        with patch("pyplumber.mixer.cli.load_avp_api", return_value=api):
            app = build_application(GraphOptions(config=str(path), janus_output=True, remote_control_port=18777,
                                                 prewarm_cut_scenes=("*",)))
        app.avp.edges.planCapacity("aux_test_gated", 1)
        gate = ConsumerGate({"name": "aux_test_gate", "src": "aux_mv_out", "dst": "aux_test_gated", "group": "aux_mv"})
        app.avp.addNode(gate)
        errors = []
        app.avp.on_exception = lambda *args: errors.append(args)
        app.avp.registerWithWebUI(args.webui, "aux-smoke", "")
        try:
            app.start()
            bus, = app.aux_buses
            pvw_scene = lambda: bus.state()["pvw_scene"]   # what the bus's follower last drew
            edge = app.avp.getEdge(bus.output_edge)
            main_edge = app.avp.getEdge("mixer_final_out")
            wait_for(lambda: edge.enqueued_total >= 30)
            expected = {name: name in (bus.edges[0], bus.edges[-1], bus.pgm_edge)
                        for name in [*bus.edges, bus.pgm_edge]}
            assert subscription_flags() == expected
            # Immediate cuts publish their preview swap on the committed frame.
            app.mixer.cut("red")
            wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
            wait_for(lambda: pvw_scene() == "")
            for preview in ("", "red"):
                if preview:
                    app.mixer.preview(preview)
                    wait_for(lambda: pvw_scene() == preview)
                app.mixer.cut("blue")
                wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
                assert control_json("mixer.status mixer")["pgm_scene"] == "blue"
                # Swap Preview/Program: the bus then shows the scene that left program, as mixer.status does.
                wait_for(lambda: pvw_scene() == control_json("mixer.status mixer")["pvw_scene"])
            # End on red: the program scene's sources stay warm on the bus (for the swap), so the
            # subscription checks below need blue off program.
            for scene in ("blue", "red") * 10:
                app.mixer.cut(scene)
                time.sleep(.01)
            wait_for(lambda: control_json("mixer.status mixer")["transition"] == "idle")
            # Swap Preview/Program: the bus shows what mixer.status shows, the scene that left
            # program at the last switch ("blue"), or none when that take replaced a pending one.
            # The follower draws the last take a few frames after it, so wait for it to catch up.
            wait_for(lambda: pvw_scene() == control_json("mixer.status mixer")["pvw_scene"])
            assert pvw_scene() in ("", "blue"), pvw_scene()
            # An unavailable new input must leave the old AUX running, then
            # release the abandoned subscription. A newer request cancels it.
            app.mixer.preview("red")
            wait_for(lambda: pvw_scene() == "red")
            def assign(scenes):
                bus.assign({"expected_revision": bus.state()["revision"], "scenes": scenes})
            assign(["red"] * 8)
            wait_for(lambda: not subscription_flags()[bus.edges[-1]])
            blue_pacer = app.avp.node(f"realtime_{args.sources - 1}")
            blue_pacer.stopAndWait()
            time.sleep(.3)
            before = edge.enqueued_total
            assign(["blue"] * 8)
            wait_for(lambda: bool(bus.state().get("composition_error")))
            assert edge.enqueued_total - before >= 3
            assert not subscription_flags()[bus.edges[-1]]
            assign(["blue"] * 8)
            assign(["red"] * 8)
            wait_for(lambda: not bus.state()["composition_pending"])
            assert not bus.state()["composition_error"]
            assert not subscription_flags()[bus.edges[-1]]
            blue_pacer.start()
            assign(["blue"] * 8)
            wait_for(lambda: not bus.state()["composition_pending"])
            assert not bus.state()["composition_error"]
            assert subscription_flags()[bus.edges[-1]], (bus.state(), subscription_flags())
            assign(list(config["aux_buses"][0]["scenes"]))
            wait_for(lambda: not bus.state()["composition_pending"])
            app.mixer.preview("blue")
            wait_for(lambda: pvw_scene() == "blue")
            initial = bus.state()
            bus.assign({"expected_revision": initial["revision"], "scenes": ["repeat"] * 8})
            assert bus.assign({"expected_revision": initial["revision"], "scenes": [None] * 8})["conflict"]
            app.mixer.cut("blue")
            app.mixer.preview("red")
            # Blue is program now: its source stays subscribed, warm for the swap.
            wait_for(lambda: subscription_flags() == expected)
            time.sleep(1)
            # Stall the first consumer: the bus keeps playing its inputs, dropping each frame its
            # encoder has no room for, and the main program keeps advancing.
            gate.blocked = True
            drops, before = bus.state()["output_drops"], main_edge.enqueued_total
            time.sleep(1)
            assert bus.state()["output_drops"] - drops >= 20   # of the second's 30 aux ticks
            assert main_edge.enqueued_total - before >= 40
            assert subscription_flags() == expected
            # Draining it brings the bus's frames back with nothing reapplied.
            gate.blocked = False
            before = edge.enqueued_total
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
            # Switch layouts at runtime: each draws once the compositor reports its revision, and
            # the bus then subscribes to exactly what the layout draws.
            def switch(layout, drawn):
                bus.set_layout({"layout": layout})
                wait_for(lambda: not bus.state()["composition_pending"])
                state = bus.state()
                assert state["composition_revision"] == state["revision"] and not state["composition_error"], state
                wait_for(lambda: subscription_flags() == {name: name in drawn for name in [*bus.edges, bus.pgm_edge]})
                before = edge.enqueued_total
                wait_for(lambda: edge.enqueued_total >= before + 15)
            switch({"preset": "source_pages"}, bus.edges[:12])
            last = len(bus.edges) - 1
            switch({"cells": [{"role": "pgm", "x": 0, "y": 0, "w": 320, "h": 240},
                              {"role": "source", "source": last, "x": 0, "y": 240, "w": 320, "h": 240}]},
                   (bus.pgm_edge, bus.edges[last]))
            # Each PVW cell reserves the largest scene (two items); exceed this bus's budget.
            try:
                bus.set_layout({"layout": {"cells": [{"role": "pvw", "x": 0, "y": 0, "w": 32, "h": 48}]
                                          * (bus.bus.max_layers // 2 + 1)}})
                raise AssertionError("a layout over max_layers was accepted")
            except ConfigError as exc:
                assert "max_layers" in str(exc), exc
            app.mixer.preview("red")
            wait_for(lambda: pvw_scene() == "red")
            switch({"preset": "pgm_pvw_grid"}, (bus.edges[0], bus.edges[-1], bus.pgm_edge))   # the slots came back
            assert bus.state()["scenes"] == current["scenes"]
            assert not errors, errors
            print("PASS: aux staging, timeout, cancellation, cadence, repeated pads, preview, revision conflict, stalled consumer, "
                  "PGM continuity and runtime layout switches", flush=True)
        finally:
            gate.blocked = False
            app.stop()


if __name__ == "__main__":
    main()
