"""Color contracts and real Python builder topology, without a GPU runtime."""
import importlib
import itertools
import sys
import types
from pathlib import Path

import pytest

from pyplumber.mixer.color import Color, conversion_graph, declared_color, rendition_color
from pyplumber.mixer.config import ConfigError, parse


@pytest.mark.parametrize("source,target", list(itertools.product(("sdr", "hlg", "pq"), repeat=2)))
def test_declared_sources_convert_only_when_the_contract_differs(source, target):
    fmt = "nv12" if target == "sdr" else "p010le"
    graph = conversion_graph(target, fmt, source=source)
    assert graph.startswith(Color(source).setparams + ",")
    if source == target:
        # Identity: stamp the tags, no tone-map pass (it would force 4:2:0).
        assert graph == Color(source).setparams + f",scale_cuda=format={fmt}"
        assert conversion_graph(target, "p210le", source=source, source_format="p210le") == Color(source).setparams
    else:
        assert f"transfer_in=auto:transfer_out={target}:format={fmt}" in graph
        assert ":tonemap=clip:sdr_white=203:hdr_peak=1000:desat=0" in graph
        assert "scale_cuda" not in graph


def test_untagged_source_is_never_inferred_from_storage():
    assert declared_color({}) is None
    for fmt in ("nv12", "p010le"):
        graph = conversion_graph("sdr", "nv12", source_format=fmt)
        assert "setparams" not in graph
        assert "transfer_in=auto" in graph
    with pytest.raises(ValueError, match="incomplete"):
        declared_color({"color_trc": "arib-std-b67"})
    with pytest.raises(ValueError, match="contradict"):
        declared_color({"color": "hlg", "color_trc": "bt709"})


@pytest.mark.parametrize("tags", [
    {**Color("hlg").tags, "color_primaries": "bt709"},
    {**Color().tags, "color_range": "pc"},
    {**Color().tags, "colorspace": "smpte170m"},
    {**Color().tags, "color_trc": "unknown"},
])
def test_unsupported_contracts_fail_instead_of_retagging(tags):
    with pytest.raises(ValueError, match="unsupported|contradictory"):
        Color.parse(tags)


def test_tonemap_converts_storage_in_the_same_pass():
    # 4:2:2 stays inside tonemap_cuda: P210 in, NV12/P210 out, no scale_cuda round trip.
    assert conversion_graph("sdr", "nv12", source_format="p210le") == \
        "tonemap_cuda=transfer_in=auto:transfer_out=sdr:format=nv12:tonemap=clip:sdr_white=203:hdr_peak=1000:desat=0"
    assert conversion_graph("hlg", "p210le").endswith(":format=p210le:tonemap=clip:sdr_white=203:hdr_peak=1000:desat=0")
    assert "scale_cuda" not in conversion_graph("hlg", "p210le", source="sdr", source_format="nv12")
    # Planar CUDA storage is re-laid out once before the tone mapper.
    assert conversion_graph("sdr", "nv12", source_format="yuv422p10le").startswith("scale_cuda=format=p210le,tonemap_cuda=")
    with pytest.raises(ValueError, match="10-bit"):
        conversion_graph("hlg", "nv12")
    with pytest.raises(ValueError, match="source pixel format"):
        conversion_graph("hlg", "p010le", source_format="rgba")


@pytest.mark.parametrize("canvas", ("sdr", "hlg", "pq"))
def test_output_codec_defaults_have_correct_color(canvas):
    assert rendition_color(canvas, "h264_nvenc") == Color("sdr")
    assert rendition_color(canvas, "hevc_nvenc") == Color(canvas)
    assert rendition_color(canvas, "hevc_nvenc", "pq") == Color("pq")
    with pytest.raises(ValueError, match="H.264"):
        rendition_color(canvas, "h264_nvenc", "hlg")


def test_raw_inputs_require_a_declaration_and_video_retains_auto():
    doc = {"canvas": {"width": 1920, "height": 1080},
           "sources": [{"id": "x", "kind": "video", "path": "<path>"}],
           "scenes": [{"id": "full", "items": [{"source": "x", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}}]}]}
    assert parse(doc).sources[0].color is None
    for kind in ("browser", "v210"):
        source = {"id": "x", "kind": kind, "path": "<path>", "url": "https://example.org", "width": 1920, "height": 1080}
        with pytest.raises(ConfigError, match="explicit color setting"):
            parse({**doc, "sources": [source]})
        assert parse({**doc, "sources": [{**source, "color": "sdr"}]}).sources[0].color == Color()


@pytest.fixture
def builder(monkeypatch):
    graph = importlib.import_module("pyplumber.mixer.graph")
    class Avp:
        def __init__(self):
            self.nodes = []
        def addNode(self, node):
            self.nodes.append(node.parameters)
        def executeCommandsFromString(self, command):
            pass
    avp = Avp()
    return graph.MixerGraphBuilder(avp, canvas=(1920, 1080), fps=(60, 1),
                                   working_format="p010le", color="hlg", enable_wipe=False)


def test_aliases_share_one_color_conversion_before_all_scene_slots(builder):
    for name in ("cam", "cam#2"):
        builder.add_source(name, "decoded", "input", default_graph="", color="sdr")
    builder.add_scene("full", {"cam": {}, "cam#2": {}})
    builder.set_initial_scene("full")
    builder.build()
    nodes = {n["name"]: n for n in builder.avp.nodes}
    conversions = [n for n in nodes.values() if "tonemap_cuda" in n.get("graph", "")]
    assert len(conversions) == 1
    assert conversions[0]["src"] == "decoded"
    fanout = nodes["mixer_color_alias_cam"]
    assert fanout["src"] == conversions[0]["dst"]
    assert len(fanout["dst"]) == 2
    for name, edge in zip(("cam", "cam#2"), fanout["dst"]):
        assert nodes[f"mixer_otm_{name}"]["src"] == edge
    assert nodes["mixer_comp_a"]["color"] == "hlg"


@pytest.mark.parametrize("color", (None, "sdr", "hlg", "pq"))
def test_routed_color_contract_reaches_both_slot_conversions(builder, color):
    builder.add_routed_source("cam", "route_a", "route_b", "input", "router", "a", "b",
                              default_graph="", color=color)
    builder.add_scene("full", {"cam": {}}, routes={"cam": 0})
    builder.set_initial_scene("full")
    builder.build()
    nodes = {n["name"]: n for n in builder.avp.nodes}
    expected = conversion_graph("hlg", "p010le", source=color)
    for slot in ("a", "b"):
        conversion = nodes[f"mixer_color_cam_{slot}"]
        assert conversion["src"] == f"route_{slot}"
        assert conversion["graph"] == expected
        assert nodes[f"mixer_comp_{slot}"]["src"] == [conversion["dst"]]


def test_shared_edge_cannot_have_conflicting_contracts(builder):
    builder.add_source("a", "decoded", "input", color="sdr")
    builder.add_source("b", "decoded", "input", color="pq")
    builder.add_scene("full", {"a": {}, "b": {}})
    builder.set_initial_scene("full")
    with pytest.raises(ValueError, match="Conflicting color"):
        builder.build()


def test_rgb_keeps_alpha_and_uses_hdr_compositor(builder):
    builder.add_source("page", "rgba", "input", default_graph="", packed_rgb=True, color="sdr")
    builder.add_scene("full", {"page": {"blend": True}})
    builder.set_initial_scene("full")
    builder.build()
    nodes = {n["name"]: n for n in builder.avp.nodes}
    assert nodes["mixer_color_page"]["graph"] == Color().setparams
    assert nodes["mixer_comp_a"]["color"] == "hlg"
    assert nodes["mixer_comp_a"]["layers"][0]["blend"]


@pytest.mark.parametrize("wipe_color", (None, "sdr"))
def test_media_wipe_blends_onto_an_hdr_canvas(monkeypatch, wipe_color):
    """Auto wipes retain tags for compositor validation; an explicit SDR override
    tags the clip before upload. Both retain the RGBA alpha-blend path."""
    graph = importlib.import_module("pyplumber.mixer.graph")
    class Avp:
        def __init__(self):
            self.nodes = []
        def addNode(self, node):
            self.nodes.append(node.parameters)
        def executeCommandsFromString(self, command):
            pass
    b = graph.MixerGraphBuilder(Avp(), canvas=(1920, 1080), fps=(60, 1), working_format="p210le",
                                color="hlg", wipe_color=wipe_color, enable_wipe=True, cache_wipes_mb=512)
    b.add_source("cam", "decoded", "input", default_graph="", color="sdr")
    b.add_scene("full", {"cam": {}})
    b.set_initial_scene("full")
    b.build()
    nodes = {n["name"]: n for n in b.avp.nodes}
    assert nodes["mixer_wipe_fmt"]["graph"] == (Color().setparams + "," if wipe_color else "") + "format=rgba,hwupload"
    overlay = nodes["mixer_wipe_overlay"]
    assert overlay["sw_format"] == "p210le" and overlay["color"] == "hlg"
    assert overlay["layers"][1]["blend"] is True and overlay["active_inputs"] == 3
    assert nodes["mixer_wipe_cache"]["src"] == "mixer_wipe_rt_out"
    with pytest.raises(ValueError, match="require SDR"):
        graph.MixerGraphBuilder(Avp(), canvas=(1920, 1080), fps=(60, 1), working_format="p210le",
                                color="hlg", wipe_color="hlg")
