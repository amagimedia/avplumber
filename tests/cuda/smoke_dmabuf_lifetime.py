"""Exercise zero-copy composition and disconnects against an isolated browser worker.

The final case deliberately kills its consumer and quarantines a source. Restart
the test worker afterwards. Never point this test at a shared production worker.
"""
import argparse
import base64
import gc
import subprocess
import sys
import time
import uuid

from _harness import drain, finish, make_avp, start
from pyplumber.mixer.dmabuf_inputs import dmabuf_cuda_input_nodes, rest_request


def consume(args):
    from pyplumber import node as api
    avp, errors = make_avp("probe", capacity=2)
    nodes, edge = dmabuf_cuda_input_nodes(
        api, prefix="probe", socket=f"{args.socket_dir}/{args.consumer}.sock",
        width=1280, height=720, fps=30, drm_hwaccel=None, cuda_hwaccel="probe",
        source_group="probe", processing_group="probe", preserve_alpha=True)
    nodes.append(api.CudaRectOverlay({
        "name": "compose", "src": [edge], "dst": "out", "hwaccel": "probe",
        "width": 1280, "height": 720, "fps": "30/1", "sw_format": args.format,
        "color": "sdr", "active_inputs": 1,
        "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": 1280, "dst_h": 720, "blend": True}]}))
    output = start(avp, nodes, "probe", "out")
    try:
        count = 0
        for frame in drain(output, errors, 15, 30):
            count += 1
            del frame
        if args.format == "bgra":
            assert any("texture-backed inputs require RGB-to-YUV" in e[-1] for e in errors), errors
        else:
            assert count == 30 and not errors, (count, errors)
        print("READY", flush=True)
        if args.crash:
            time.sleep(60)
    finally:
        finish(avp, nodes)
        del output, nodes, avp
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest", default="http://127.0.0.1:9009")
    parser.add_argument("--socket-dir", default="/tmp/dma-page")
    parser.add_argument("--consumer")
    parser.add_argument("--format", default="nv12")
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args()
    if args.consumer:
        return consume(args)

    page = '<body style="background:#236"><div style="width:80px;height:80px;background:white;animation:a 1s linear infinite"></div><style>@keyframes a{to{transform:translateX(500px)}}</style>'
    url = "data:text/html;base64," + base64.b64encode(page.encode()).decode()
    for fmt, crash in [("nv12", False), ("p210le", False), ("bgra", False), ("nv12", True)]:
        name = "lifetime_" + uuid.uuid4().hex[:8]
        rest_request(args.rest, "POST", "/window/open", {
            "id": name, "url": url, "width": 1280, "height": 720, "fps": 30, "audio": False})
        child = subprocess.Popen([sys.executable, "-u", __file__, "--consumer", name,
            "--socket-dir", args.socket_dir, "--format", fmt, *(["--crash"] if crash else [])],
            stdout=subprocess.PIPE, text=True)
        try:
            for line in child.stdout:
                if line.strip() == "READY":
                    if crash:
                        child.kill()
                    break
            assert child.wait(timeout=20) == (-9 if crash else 0)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                windows = rest_request(args.rest, "GET", "/status")["windows"]
                stats = next(w["stats"] for w in windows if w["id"] == name)
                if (stats["quarantinedFrameCount"] > 0 if crash else stats["retainedFrameCount"] == 0):
                    break
                time.sleep(.05)
            assert bool(stats["quarantinedFrameCount"]) == crash, stats
            assert stats["retainedFrameCount"] == 0, stats
            print(f"PASS {fmt} {'crash quarantined' if crash else 'clean shutdown'}: {stats}", flush=True)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            rest_request(args.rest, "POST", "/window/close", {"id": name})


if __name__ == "__main__":
    main()
