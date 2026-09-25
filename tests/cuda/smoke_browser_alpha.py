"""Verify browser DMA-BUF alpha survives CUDA import on an NVIDIA host.

Requires the DMA-BUF browser service and the mixer browser_alpha.html page.
The download is solely a verification boundary; the live mixer stays on GPU.
"""

import argparse
import base64
import json
from pathlib import Path
import uuid

import numpy as np

from _harness import drain, finish, make_avp, start
from pyplumber.mixer.dmabuf_inputs import (
    dmabuf_cuda_input_nodes, open_windows, refresh_windows, rest_request, wait_for_sockets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest", default="http://127.0.0.1:9009")
    parser.add_argument("--socket-dir", default="/tmp/dma-page")
    parser.add_argument("--format", choices=("rgba", "bgra"), default="bgra",
                        help="Browser buffer byte order (NVIDIA normally uses BGRA)")
    parser.add_argument("--composite", action="store_true", help="Verify opacity over a black P210 canvas")
    parser.add_argument("--unclocked", action="store_true", help="With --composite, test per-frame layout metadata and upstream pacing")
    args = parser.parse_args()
    if args.unclocked and not args.composite:
        parser.error("--unclocked requires --composite")
    from pyplumber import node as api

    name = "alpha_probe_" + uuid.uuid4().hex[:8]
    width, height = 1280, 720
    page = Path(__file__).resolve().parents[2] / "demos/mixer/browser_alpha.html"
    url = "data:text/html;base64," + base64.b64encode(page.read_bytes()).decode("ascii")
    avp, errors = make_avp("alpha_gpu", capacity=2)
    nodes = []
    try:
        open_windows(args.rest, [{"id": name, "url": url, "width": width, "height": height, "fps": 30}])
        socket = f"{args.socket_dir}/{name}.sock"
        wait_for_sockets([socket], 20)
        nodes, edge = dmabuf_cuda_input_nodes(
            api, prefix="probe", socket=socket, width=width, height=height, fps=30,
            drm_hwaccel=None, cuda_hwaccel="alpha_gpu", source_group="probe", processing_group="probe",
            preserve_alpha=True)
        nodes[1].parameters["real_pixel_format"] = args.format
        if args.unclocked:
            class Layout(api.PythonNode):
                count = 0

                def process(self):
                    frame = self._src.get()
                    if frame:
                        shift = 40 if self.count % 2 else 0
                        self.count += 1
                        frame.metadata["probe_layout"] = json.dumps({"0": {
                            "dst_x": shift, "dst_y": 0, "dst_w": width, "dst_h": height,
                            "fit": "contain", "blend": True}})
                        frame.metadata["probe_shift"] = str(shift)
                        self._dst.enqueue(frame)

            nodes.append(Layout({"name": "layout", "src": edge, "dst": "marked"}))
            edge = "marked"
        if args.composite:
            nodes.append(api.CudaRectOverlay({
                "name": "blend", "src": [edge], "dst": "blended", "hwaccel": "alpha_gpu",
                "width": width, "height": height, "sw_format": "p210le", "color": "sdr",
                **({"metadata_key": "probe_layout", "scale": True} if args.unclocked else {"fps": "30/1"}),
                "active_inputs": 1,
                "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": width, "dst_h": height, "blend": True}]}))
            edge = "blended"
        fmt = "p210le" if args.composite else args.format
        nodes.append(api.FilterVideo({"name": "verify", "src": edge, "dst": "result", "hwaccel": "alpha_gpu",
                                      "graph": f"hwdownload,format={fmt}"}))
        output = start(avp, nodes, "probe", "result")
        refresh_windows(args.rest, [name])
        expected = np.array([0, 64, 128, 191, 255])
        # Centres of the five white swatches; CSS uses 5% margins and 2% gaps.
        xs = [round(width * (0.05 + i * 0.1836 + 0.0828)) for i in range(5)]
        samples = None
        shifts = set()
        for frame in drain(output, errors, 15, 60):
            if args.composite:
                luma = np.frombuffer(frame.data[0], "<u2").reshape(height, frame.linesize[0] // 2) >> 6
                shift = int(frame.metadata["probe_shift"]) if args.unclocked else 0
                samples = luma[round(height * 0.33), [x + shift for x in xs]].astype(int)
                reference = np.rint(64 + 876 * expected / 255)
                if np.max(np.abs(samples - reference)) <= 2:
                    shifts.add(shift)
                    if args.unclocked:
                        if shift:
                            assert np.all(luma[:, :40] == 64)
                        if shifts != {0, 40}:
                            continue
                    print(f"PASS premultiplied browser over P210 black: Y={samples.tolist()}", flush=True)
                    break
                continue
            pixels = np.frombuffer(frame.data[0], np.uint8).reshape(height, frame.linesize[0])[:, :width * 4]
            pixels = pixels.reshape(height, width, 4)
            samples = pixels[round(height * 0.33), xs].copy()
            if np.max(np.abs(samples[:, 3].astype(int) - expected)) <= 2:
                assert any(f", {fmt}," in repr(frame) for fmt in ("rgba", "bgra")), repr(frame)
                assert pixels[0, 0, 3] == 0, "uncovered page background is opaque"
                assert np.max(np.abs(samples[:, :3].astype(int) - expected[:, None])) <= 2, "expected premultiplied white"
                print(f"PASS {frame}: white swatches RGBA/BGRA={samples.tolist()}", flush=True)
                break
        else:
            raise AssertionError(f"alpha did not survive browser import: {samples}; errors={errors}")
        assert not errors, errors
    finally:
        rest_request(args.rest, "POST", "/window/close", {"id": name})
        finish(avp, nodes)


if __name__ == "__main__":
    main()
