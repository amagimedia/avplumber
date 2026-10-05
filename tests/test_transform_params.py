"""Parameters of the cuda_transform node as the shared Python builder emits them.

The node parses these keys in src/nodes/hwaccel/cuda_transform.cpp (outputs, fps, drop) and
src/mixer/primitives/compositor_layers.hpp (layers); the expected dictionaries below are that
contract written out, so a change of either side fails here.
"""
import subprocess
import sys

import pytest

from pyplumber.transform import transform_output, transform_params


def test_fit_with_bars_is_one_contained_layer_over_the_whole_canvas():
    params = transform_params(
        "input_3_fps", [transform_output("input_3_normalized", 1920, 1080, fit="contain")],
        hwaccel="mixer_gpu", name="normalize_3", group="input_3", auto_restart="group")
    assert params == {
        "src": "input_3_fps", "hwaccel": "mixer_gpu",
        "outputs": [{"dst": "input_3_normalized", "width": 1920, "height": 1080, "sw_format": "nv12",
                     "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": 1920, "dst_h": 1080, "fit": "contain"}]}],
        "name": "normalize_3", "group": "input_3", "auto_restart": "group"}


def test_plain_scale_serves_several_edges_from_one_canvas():
    params = transform_params(
        "program_transform",
        [transform_output(["program_sized_sdr", "program_sized_low"], 1280, 720, sw_format="p210le")],
        hwaccel="mixer_gpu", name="transform_renditions", group="output", on_error="panic")
    assert params == {
        "src": "program_transform", "hwaccel": "mixer_gpu",
        "outputs": [{"dst": ["program_sized_sdr", "program_sized_low"], "width": 1280, "height": 720,
                     "sw_format": "p210le",
                     "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": 1280, "dst_h": 720}]}],
        "name": "transform_renditions", "group": "output", "on_error": "panic"}


def test_box_and_crop_place_a_part_of_the_frame():
    output = transform_output("lane", 640, 384, box=(0, 12, 640, 360), crop=(32, 20, 1280, 720))
    assert output == {
        "dst": "lane", "width": 640, "height": 384, "sw_format": "nv12",
        "layers": [{"dst_x": 0, "dst_y": 12, "dst_w": 640, "dst_h": 360,
                    "crop": {"x": 32, "y": 20, "w": 1280, "h": 720}}]}


@pytest.mark.parametrize("fps,emitted", [(30, "30/1"), ("30/1", "30/1"), ((30000, 1001), "30000/1001"),
                                         ("30000/1001", "30000/1001"), ("15", "15/1")])
def test_fps_is_always_a_ratio_string(fps, emitted):
    # The node reads it with get<std::string>(): a JSON number would throw at graph build.
    assert transform_output("half", 96, 64, fps=fps)["fps"] == emitted


def test_optional_keys_are_emitted_only_when_set():
    plain = transform_output("out", 96, 64)
    assert set(plain) == {"dst", "width", "height", "sw_format", "layers"}
    assert set(plain["layers"][0]) == {"dst_x", "dst_y", "dst_w", "dst_h"}
    assert transform_output("out", 96, 64, drop=True)["drop"] is True
    assert transform_output("out", 96, 64, fit="stretch") == plain


def test_a_tuple_of_edges_is_emitted_as_a_list():
    assert transform_output(("a", "b"), 96, 64)["dst"] == ["a", "b"]


@pytest.mark.parametrize("arguments", [
    {"fit": "cover"},                # the node knows stretch and contain only
    {"fps": 0}, {"fps": "-30/1"}, {"fps": (30, 0)},
    {"box": (0, 0, 96)}, {"crop": (0, 0, 96, 64, 1)},
    {"dst": []},
])
def test_invalid_arguments_are_rejected_before_the_graph_is_built(arguments):
    arguments = {"dst": "out", **arguments}
    with pytest.raises(ValueError):
        transform_output(arguments.pop("dst"), 96, 64, **arguments)


def test_a_node_needs_an_output_and_a_device():
    with pytest.raises(ValueError, match="output"):
        transform_params("in", [], hwaccel="gpu")
    with pytest.raises(TypeError):
        transform_params("in", [transform_output("out", 96, 64)])   # hwaccel is required by the node


def test_outputs_are_not_shared_with_the_caller():
    outputs = [transform_output("out", 96, 64)]
    params = transform_params("in", outputs, hwaccel="gpu")
    outputs.append(transform_output("late", 96, 64))
    assert [o["dst"] for o in params["outputs"]] == ["out"]


def test_builder_imports_without_the_native_module():
    # The reframer and a recorder build parameters before (or without) loading the engine.
    subprocess.run([sys.executable, "-c", "import sys, pyplumber.transform; "
                    "assert '_avplumber' not in sys.modules; assert 'pyplumber.node' not in sys.modules"],
                   check=True)
