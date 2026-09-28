"""Backend selection must cover every bus without changing native node APIs."""
import json
import subprocess
import sys

import pytest

from mixer import GraphOptions, build_application
from pyplumber.mixer.backend import mixer_backend
from pyplumber.mixer.backends.cuda import CudaMixerBackend
from test_graph import FakeAvp, fake_api
from test_prewarm import native_boundary


def test_selection_and_pure_python_imports():
    assert mixer_backend().name == "cuda"
    with pytest.raises(ValueError, match="Unsupported"):
        mixer_backend("vulkan")
    with pytest.raises(TypeError):
        mixer_backend(object())
    subprocess.run([sys.executable, "-c", "from pyplumber.mixer.backend import mixer_backend; "
                    "from pyplumber.mixer.color import conversion_graph; import sys; "
                    "mixer_backend().conversion('sdr', 'nv12'); "
                    "assert '_avplumber' not in sys.modules; assert 'pyplumber.node' not in sys.modules"], check=True)


def test_program_wipe_aux_and_renditions_use_same_backend(native_boundary, tmp_path):
    nodes, builder = native_boundary
    calls = []
    class RecordingBackend(CudaMixerBackend):
        def compositor(self, params, **kwargs):
            calls.append(("compositor", params["name"]))
            return super().compositor(params, **kwargs)
        def transition(self, params):
            calls.append(("transition", params["name"]))
            return super().transition(params)
        def conversion(self, target, pixel_format, **options):
            calls.append(("conversion", pixel_format))
            return super().conversion(target, pixel_format, **options)
    class Engine(FakeAvp):
        def addNode(self, node, **kwargs):
            super().addNode(node)
    backend = RecordingBackend()
    api = fake_api()
    api.AVPlumber = Engine
    api.CudaRectOverlay = nodes.CudaRectOverlay
    api.MixerGraphBuilder = lambda *a, **kw: builder(*a, backend=backend, **kw)
    path = tmp_path / "show.json"
    path.write_text(json.dumps({
        "canvas": {"width": 1080, "height": 1920, "fps": 60},
        "sources": [{"id": "video", "kind": "video", "path": "clip.mp4", "width": 1920, "height": 1080, "color": "sdr"}],
        "scenes": [{"id": "full", "items": [{"source": "video", "dst": {"x": 0, "y": 0, "w": 1080, "h": 1920}}]}],
        "renditions": [{"id": "program", "port": 5004}],
        "aux_buses": [{"id": "mv", "scenes": ["full"] * 8, "renditions": [{"id": "monitor", "port": 5008}]}],
    }))
    app = build_application(GraphOptions(config=str(path), janus_output=True), api=api)
    assert app.mixer.backend is backend
    assert [name for kind, name in calls if kind == "compositor"] == [
        "mixer_comp_a", "mixer_comp_b", "mixer_wipe_overlay", "aux_mv_comp"]
    assert ("transition", "mixer_out_sel_transition") in calls
    assert len([c for c in calls if c[0] == "conversion"]) == 3  # source, aux and PGM rendition


@pytest.mark.parametrize("scene_view", [True, False])
def test_source_pages_bus_skips_the_program_tap_and_labels_outputs(native_boundary, tmp_path, scene_view):
    nodes, builder = native_boundary
    class Engine(FakeAvp):
        def addNode(self, node, **kwargs):
            super().addNode(node)
    api = fake_api()
    api.AVPlumber = Engine
    api.CudaRectOverlay = nodes.CudaRectOverlay
    api.MixerGraphBuilder = builder
    buses = [{"id": "mv2", "layout": {"preset": "source_pages"}, "renditions": [{"id": "monitor", "port": 5012}]}]
    if scene_view:
        buses.insert(0, {"id": "mv", "scenes": ["full"] * 8, "renditions": [{"id": "monitor", "port": 5008}]})
    path = tmp_path / "show.json"
    path.write_text(json.dumps({
        "canvas": {"width": 1080, "height": 1920, "fps": 25},
        "sources": [{"id": f"video{i}", "kind": "video", "path": f"clip{i}.mp4", "width": 1920, "height": 1080,
                     "color": "sdr"} for i in range(13)],
        "scenes": [{"id": "full", "items": [{"source": "video0", "dst": {"x": 0, "y": 0, "w": 1080, "h": 1920}}]}],
        "renditions": [{"id": "program", "port": 5004}],
        "aux_buses": buses,
    }))
    app = build_application(GraphOptions(config=str(path), janus_output=True), api=api)
    graph = {n.parameters["name"]: n.parameters for n in app.avp.nodes if "name" in n.parameters}
    pages = graph["aux_mv2_comp"]
    # Page 1 of 2: twelve tiles, no PGM pad.
    assert pages["src"] == pages["subscriptions"] == [f"aux_mv2_source_{i}" for i in range(13)]
    assert [layer["input"] for layer in pages["layers"]] == list(range(12))
    assert pages["latency_ms"] == 80
    assert pages["max_layers"] == 12 and graph["mixer_wipe_overlay"]["max_layers"] == 2
    # Monitors keep one NVENC reference frame; the program output leaves it to NVENC.
    assert graph["aux_mv2_encoder"]["options"]["dpb_size"] == 1
    assert "dpb_size" not in graph["janus_encoder"]["options"]
    if scene_view:
        assert graph["program_aux_tap"]["dst"][1:] == ["aux_mv_pgm"]
        # Same playout latency; the PGM pad (after the 13 sources) is shown one tick later instead.
        assert graph["aux_mv_comp"]["latency_ms"] == 80
        assert graph["aux_mv_comp"]["pgm_delay_frames"] == 1
        assert graph["aux_mv_comp"]["src"][-1] == "aux_mv_pgm"   # the delayed input is the last one
        assert "pgm_delay_frames" not in pages
        assert graph["aux_mv_encoder"]["options"]["dpb_size"] == 1
    else:
        assert "program_aux_tap" not in graph
    settings = json.loads(app.avp.commands_registered["mixer.settings"](""))
    assert [(o["bus"], o["label"], o["layout"]) for o in settings["preview_outputs"]] == [
        *([("mv", "Program preview", "pgm_pvw_grid")] if scene_view else []), ("mv2", "Multiviewer", "source_pages")]
