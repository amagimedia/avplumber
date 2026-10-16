"""Verify a browser page's channel order survives CUDA import and composition.

The page is red on the left and blue on the right; the frames are composited
onto a P210 canvas and the chroma of each half is compared with BT.709. A
swapped red and blue (an RGBA buffer read as BGRA, or the reverse) moves the
two halves' Cb and Cr far apart. The download is solely a verification boundary.
"""

import argparse
import base64
import json
import uuid

import numpy as np

from _harness import drain, finish, make_avp, start
from pyplumber.mixer.dmabuf_inputs import (
    dmabuf_cuda_input_nodes, open_windows, refresh_windows, rest_request, wait_for_sockets,
)

PAGE = ('<body style="margin:0;background:transparent">'
        '<div style="position:fixed;left:0;top:0;width:50%;height:100%;background:#ff0000"></div>'
        '<div style="position:fixed;left:50%;top:0;width:50%;height:100%;background:#0000ff"></div>')


def bt709_limited_10bit(r, g, b):
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (round(64 + 876 * y), round(512 + 896 * (b - y) / 1.8556), round(512 + 896 * (r - y) / 1.5748))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest", default="http://127.0.0.1:9009")
    parser.add_argument("--socket-dir", default="/tmp/dma-page")
    args = parser.parse_args()
    from pyplumber import node as api

    name = "colour_probe_" + uuid.uuid4().hex[:8]
    width, height = 1280, 720
    url = "data:text/html;base64," + base64.b64encode(PAGE.encode()).decode("ascii")
    avp, errors = make_avp("colour_gpu", capacity=2)
    nodes = []
    try:
        open_windows(args.rest, [{"id": name, "url": url, "width": width, "height": height, "fps": 30}])
        socket = f"{args.socket_dir}/{name}.sock"
        wait_for_sockets([socket], 20)
        nodes, edge = dmabuf_cuda_input_nodes(
            api, prefix="probe", socket=socket, fps=30,
            drm_hwaccel=None, cuda_hwaccel="colour_gpu", source_group="probe", processing_group="probe",
            preserve_alpha=True)
        nodes.append(api.MixerCompositor({
            "name": "blend", "src": [edge], "dst": "blended", "hwaccel": "colour_gpu",
            "width": width, "height": height, "sw_format": "p210le", "color": "sdr", "fps": "30/1",
            "active_inputs": 1,
            "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": width, "dst_h": height, "blend": True}]}))
        nodes.append(api.FilterVideo({"name": "verify", "src": "blended", "dst": "result", "hwaccel": "colour_gpu",
                                      "graph": "hwdownload,format=p210le"}))
        output = start(avp, nodes, "probe", "result")
        refresh_windows(args.rest, [name])
        want = {"red": bt709_limited_10bit(1, 0, 0), "blue": bt709_limited_10bit(0, 0, 1)}
        got = None
        for frame in drain(output, errors, 15, 60):
            y = np.frombuffer(frame.data[0], "<u2").reshape(height, frame.linesize[0] // 2)[:, :width] >> 6
            uv = np.frombuffer(frame.data[1], "<u2").reshape(height, frame.linesize[1] // 2)[:, :width] >> 6
            row = height // 2
            got = {}
            for label, x in (("red", width // 4), ("blue", 3 * width // 4)):
                x -= x % 2
                got[label] = (int(y[row, x]), int(uv[row, x]), int(uv[row, x + 1]))
            if all(max(abs(a - b) for a, b in zip(got[k], want[k])) <= 6 for k in want):
                stats = avp.node("probe_receive").getObject("frame_stats")
                print(f"PASS red and blue kept their channels: YCbCr={got} expected={want} "
                      f"frames received={stats['received']}", flush=True)
                break
        else:
            raise AssertionError(f"channel order or colour wrong: YCbCr={got} expected={want}; errors={errors}")
        assert not errors, errors
    finally:
        rest_request(args.rest, "POST", "/window/close", {"id": name})
        finish(avp, nodes)


if __name__ == "__main__":
    main()
