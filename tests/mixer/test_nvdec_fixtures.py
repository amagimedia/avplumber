"""The pure parts of the live-feed decode check (tests/cuda/nvdec/live_feeds.py): the H.264 SPS
rewrite that cuts off the VUI, and the pool it expects from a stream's headers."""
import importlib.util
from pathlib import Path
import re

import pytest

spec = importlib.util.spec_from_file_location(
    "live_feeds", Path(__file__).resolve().parents[1] / "cuda" / "nvdec" / "live_feeds.py")
live_feeds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_feeds)


class Writer:
    """Builds an RBSP bit by bit: fixed fields and Exp-Golomb codes, the inverse of live_feeds.Bits."""

    def __init__(self):
        self.bits = ""

    def u(self, n, value):
        self.bits += format(value, f"0{n}b")

    def ue(self, value):
        code = format(value + 1, "b")
        self.bits += "0" * (len(code) - 1) + code

    def se(self, value):
        self.ue(2 * value - 1 if value > 0 else -2 * value)

    def trailing(self):
        self.bits += "1" + "0" * (-(len(self.bits) + 1) % 8)
        return int(self.bits, 2).to_bytes(len(self.bits) // 8, "big")


def high_profile_sps(vui_bits, *, refs=16, level=51, poc_type=0, scaling=False, cropping=False):
    """An SPS RBSP of the High profile at 1920x1080, as NVENC codes one, with *vui_bits* after
    vui_parameters_present_flag = 1; returns (rbsp, bit position of that flag)."""
    w = Writer()
    w.u(8, 100), w.u(8, 0), w.u(8, level), w.ue(0)
    w.ue(1), w.ue(0), w.ue(0), w.u(1, 0)        # 4:2:0, 8-bit, no transform bypass
    w.u(1, int(scaling))
    if scaling:
        for i in range(8):
            w.u(1, 1)
            for _ in range(16 if i < 6 else 64):   # every delta coded: the scale never reaches zero
                w.se(1)
    w.ue(0), w.ue(poc_type)
    if poc_type == 0:
        w.ue(2)
    elif poc_type == 1:
        w.u(1, 0), w.se(0), w.se(0), w.ue(2), w.se(1), w.se(-1)
    w.ue(refs), w.u(1, 0), w.ue(119), w.ue(67), w.u(1, 1), w.u(1, 1)
    w.u(1, int(cropping))
    if cropping:
        w.ue(0), w.ue(0), w.ue(0), w.ue(4)
    position = len(w.bits)
    w.u(1, 1)
    w.bits += vui_bits
    return w.trailing(), position


@pytest.mark.parametrize("options", [{}, {"poc_type": 1}, {"scaling": True}, {"cropping": True}, {"refs": 0, "level": 40}])
def test_vui_flag_position_follows_the_sps_syntax(options):
    rbsp, position = high_profile_sps("1" * 37 + "0" * 11, **options)
    assert live_feeds.vui_flag_position(rbsp) == position


def test_strip_vui_leaves_an_sps_without_vui_and_other_nal_units_as_they_are():
    # A VUI with three zero bytes in it however it aligns: the payload needs an emulation
    # prevention byte, which the rewrite takes out and puts back.
    rbsp, position = high_profile_sps("0" * 32 + "1")
    payload = live_feeds.payload_of(rbsp)
    assert payload != rbsp and live_feeds.rbsp_of(payload) == rbsp
    sps = b"\x67" + payload
    pps, slice_ = b"\x68\xce\x38\x80", b"\x65\x88\x84\x00\x00\x03\x00\x01"
    container = b"\x47\x40\x00\x10"   # bytes around the stream stay, so a transport stream stays one
    stream = container + b"\x00\x00\x00\x01" + sps + b"\x00\x00\x01" + pps + b"\x00\x00\x00\x01" + slice_
    stripped = live_feeds.strip_vui(stream)
    assert len(stripped) == len(stream)
    units = re.split(rb"\x00\x00\x00?\x01", stripped)[1:]
    assert units[1] == pps and units[2] == slice_
    assert stripped.startswith(container + b"\x00\x00\x00\x01\x67") and b"\x00\x00\x00\x01" + slice_ in stripped   # start codes kept
    new_rbsp = live_feeds.rbsp_of(units[0][1:])
    assert live_feeds.vui_flag_position(new_rbsp) == position
    bits = format(int.from_bytes(new_rbsp, "big"), f"0{8 * len(new_rbsp)}b")
    assert bits[:position] == format(int.from_bytes(rbsp, "big"), f"0{8 * len(rbsp)}b")[:position]
    assert bits[position] == "0" and bits[position + 1:].rstrip("0") == "1"


@pytest.mark.parametrize("codec,info,pool", [
    ("h264", {"max_num_ref_frames": 16, "bitstream_restriction_flag": 1, "max_num_reorder_frames": 4}, 37),
    ("h264", {"max_num_ref_frames": 16, "bitstream_restriction_flag": 0, "level_idc": 51,
              "pic_width_in_mbs_minus1": 119, "pic_height_in_map_units_minus1": 67, "frame_mbs_only_flag": 1}, 48),
    ("h264", {"max_num_ref_frames": 2, "level_idc": 41, "pic_width_in_mbs_minus1": 119,
              "pic_height_in_map_units_minus1": 33, "frame_mbs_only_flag": 0}, 23),    # 1080i: 34 map units of two rows
    ("h264", {"max_num_ref_frames": 0, "level_idc": 51, "pic_width_in_mbs_minus1": 119,
              "pic_height_in_map_units_minus1": 67, "frame_mbs_only_flag": 1}, 17),    # intra only: no reorder
    ("hevc", {"sps_max_dec_pic_buffering_minus1": 2}, 21),
    ("hevc", {"sps_max_dec_pic_buffering_minus1": 15}, 34),
    ("hevc", {}, None),
])
def test_expected_pool_reads_the_headers_the_way_libavcodec_does(codec, info, pool):
    assert live_feeds.expected_pool(codec, info, 12) == pool
