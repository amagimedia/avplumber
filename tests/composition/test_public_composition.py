"""Public composition works without the mixer application or private AI nodes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyplumber.composition import (
    SourceCompositionConfig,
    SourceCompositionGraph,
    compositor,
)
from pyplumber.composition.geometry import layers_from_plan
from pyplumber.composition.nodes import source_mask_param


class Api:
    def __getattr__(self, kind):
        return lambda params: {"type": kind, **params}


def test_imports_and_construction_do_not_load_native_mixer_or_private_packages():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("pyplumber.mixer", "amagi_avplumber_nodes", "_avplumber")):
            raise AssertionError("unexpected dependency: " + fullname)
sys.meta_path.insert(0, Block())
from pyplumber.composition import SourceCompositionConfig, SourceCompositionGraph, SourceMove, compositor
from pyplumber.composition.geometry import layers_from_plan
class Api:
    def __getattr__(self, name):
        return lambda params: dict(type=name, **params)
nodes=[]
SourceCompositionGraph(SourceCompositionConfig(1080,1920,background="blur"), Api()).build(nodes,"video")
assert compositor({}, timing="clock", api=Api())["type"] == "MixerCompositor"
assert SourceMove([],[],0,1,curve="ease-in-out").at(.5)["progress"] == .5
assert layers_from_plan([],src_w=1920,src_h=1080,dst_w=1080,dst_h=1920)[0]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


@pytest.mark.parametrize(
    "timing,keyer,expected",
    [
        ("source", False, "CudaRectOverlay"),
        ("clock", False, "MixerCompositor"),
        ("clock", True, "MixerKeyer"),
    ],
)
def test_timing_modes_keep_the_original_native_contract(timing, keyer, expected):
    params = {"src": ["a", "b"], "dst": "result", "hwaccel": "device", "layers": []}
    if timing == "clock":
        params.update(fps="30000/1001", mixer="show", latency_ms=80)
    before = json.dumps(params, sort_keys=True)
    node = compositor(params, timing=timing, keyer=keyer, api=Api())
    assert node == {"type": expected, **params}
    assert json.dumps(params, sort_keys=True) == before
    if timing == "source":
        assert "fps" not in node and "clock_input" not in node


@pytest.mark.parametrize(
    "kwargs", [{"timing": "guess"}, {"timing": "source", "keyer": True}]
)
def test_unsupported_timing_fails_before_native_import(kwargs):
    with pytest.raises(ValueError):
        compositor({}, **kwargs)


def test_public_graphs_can_coexist_and_all_branches_use_the_selected_device():
    nodes = []
    for tag in ("portrait", "square"):
        cfg = SourceCompositionConfig(
            1080,
            1920 if tag == "portrait" else 1080,
            name=tag,
            edge_prefix=tag,
            dst=tag + "_out",
            max_layers=5,
            background="blur",
            hwaccel="gpu1",
            metadata_key="layers",
            move_metadata_key="move",
        )
        assert (
            SourceCompositionGraph(cfg, Api()).build(nodes, tag + "_in") == tag + "_out"
        )
    assert len({n["name"] for n in nodes}) == len(nodes)
    assert all(n["hwaccel"] == "gpu1" for n in nodes if "hwaccel" in n)
    for n in nodes:
        if n["type"] == "CudaRectOverlay":
            assert len(n["src"]) == len(n["layers"]) == 6
            assert n["active_inputs"] == 63
            assert n["layers"][-1]["input"] == 5 and n["layers"][-1]["z"] == -1
            assert n["metadata_key"] == "layers" and n["move_metadata_key"] == "move"
        assert "hwdownload" not in n.get("graph", "")
        assert "hwupload" not in n.get("graph", "")


def test_large_compositions_use_the_existing_native_mask_encoding():
    nodes = []
    SourceCompositionGraph(
        SourceCompositionConfig(1080, 1920, max_layers=70), Api()
    ).build(nodes, "in")
    assert nodes[-1]["active_inputs"] == "1" * 70
    assert source_mask_param(1 << 70) == "0" * 70 + "1"


def test_geometry_keeps_unused_pads_hidden_without_a_three_layer_limit():
    plan = [{"src": [0, 0, 1, 1], "tgt": [0, 0, 1, 1]}] * 4
    layers, dsts = layers_from_plan(
        plan, src_w=1920, src_h=1080, dst_w=1080, dst_h=1920, max_layers=5
    )
    assert len(layers) == len(dsts) == 5
    assert all(layers[str(i)]["dst_x"] == 0 for i in range(4))
    assert layers["4"]["dst_x"] == -1080


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dest_width": 1079},
        {"source_height": 0},
        {"max_layers": 0},
        {"max_layers": 256, "background": "blur"},
        {"max_layers": 257},
        {"background": "invalid"},
        {"fps": "0/1"},
        {"blur_step": 1},
        {"blur_fraction": float("nan")},
        {"wash": 2},
    ],
)
def test_invalid_graph_configuration_fails_before_start(kwargs):
    with pytest.raises(ValueError):
        SourceCompositionConfig(**{"dest_width": 1080, "dest_height": 1920, **kwargs})
