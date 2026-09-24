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
