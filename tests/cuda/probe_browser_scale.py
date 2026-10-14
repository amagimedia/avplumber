"""Many browser windows into one compositor: the compositor's tick timing and each chain's import churn.

Opens --windows animated pages on an isolated browser worker, feeds each through the mixer's browser
input chain into one MixerCompositor (NV12 1080p30, every window a tile), runs --seconds, and prints
the compositor's draw / issue / gpu_wait times and, per chain, fresh imports.
A diagnosis tool, not a pass/fail smoke. Never point it at a shared production worker.
"""
import argparse
import json
import math
import time
import uuid

from _harness import drain, finish, make_avp, start
from pyplumber.mixer.dmabuf_inputs import (
    dmabuf_cuda_input_nodes, open_windows, refresh_windows, rest_request, wait_for_sockets,
)

PAGE = ('<body style="margin:0;background:#123">'
        '<div style="position:absolute;left:0;top:40%;width:200px;height:200px;background:#e83;'
        'animation:a 2s linear infinite"></div><div id="t" style="font:64px sans-serif;color:#fff"></div>'
        '<script>let n=0;setInterval(()=>{document.getElementById("t").textContent=n++},33)</script>'
        '<style>@keyframes a{to{transform:translateX(1700px) rotate(360deg)}}</style>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest", default="http://127.0.0.1:19009")
    parser.add_argument("--socket-dir", default="/tmp/dma-test")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--queue", type=int, default=None, help="browser_frame_queue (mixer default when omitted)")
    args = parser.parse_args()
    from pyplumber import node as api
    import base64

    url = "data:text/html;base64," + base64.b64encode(PAGE.encode()).decode("ascii")
    tag = uuid.uuid4().hex[:6]
    names = [f"scale_{tag}_{i}" for i in range(args.windows)]
    avp, errors = make_avp("scale_gpu", capacity=2)
    nodes = []
    try:
        open_windows(args.rest, [{"id": n, "url": url, "width": 1920, "height": 1080, "fps": 30} for n in names])
        wait_for_sockets([f"{args.socket_dir}/{n}.sock" for n in names], 60)
        edges = []
        for i, n in enumerate(names):
            chain, edge = dmabuf_cuda_input_nodes(
                api, prefix=f"w{i}", socket=f"{args.socket_dir}/{n}.sock", fps=30, drm_hwaccel=None,
                cuda_hwaccel="scale_gpu", source_group="scale", processing_group="scale",
                preserve_alpha=True, browser_frame_queue=args.queue)
            nodes += chain
            edges.append(edge)
        cols = math.ceil(math.sqrt(args.windows)); rows = math.ceil(args.windows / cols)
        w, h = 1920 // cols // 2 * 2, 1080 // rows // 2 * 2
        layers = [{"dst_x": (i % cols) * w, "dst_y": (i // cols) * h, "dst_w": w, "dst_h": h, "blend": True}
                  for i in range(args.windows)]
        nodes.append(api.MixerCompositor({
            "name": "compose", "src": edges, "dst": "out", "hwaccel": "scale_gpu", "width": 1920, "height": 1080,
            "sw_format": "nv12", "color": "sdr", "fps": "30/1", "active_inputs": args.windows, "layers": layers}))
        output = start(avp, nodes, "scale", "out")
        refresh_windows(args.rest, names)
        for _ in drain(output, errors, 10):
            pass
        avp.executeCommandsFromString("node.object.set compose reset_counters true")
        before = {i: avp.node(f"w{i}_to_cuda").getObject("import_stats") for i in range(args.windows)}
        count = sum(1 for _ in drain(output, errors, args.seconds))
        status = avp.node("compose").getObject("status")
        after = {i: avp.node(f"w{i}_to_cuda").getObject("import_stats") for i in range(args.windows)}
        timing = status["playout"]["timing"]
        out = {"frames": count, "errors": errors[:3], "output_drops": status.get("output_drops"),
               "repeats": status["playout"].get("repeats"), "missed": status["playout"].get("missed_deadlines")}
        for part in ("draw", "issue", "gpu_wait"):
            out[part] = {"mean_ms": round(timing[part]["mean_ms"], 2), "max_ms": round(timing[part]["max_ms"], 1)}
        imports = [after[i]["imports"] - before[i]["imports"] for i in range(args.windows)]
        out["fresh_imports"] = {"sum": sum(imports), "max_per_window": max(imports), "per_s": round(sum(imports) / args.seconds, 2),
                                "expired": sum(after[i]["expired"] - before[i]["expired"] for i in range(args.windows)),
                                "evicted": sum(after[i]["evicted"] - before[i]["evicted"] for i in range(args.windows))}
        print("RESULT " + json.dumps(out), flush=True)
        status = rest_request(args.rest, "GET", "/status")
        reasons = {}
        for wdw in status["windows"]:
            for k, v in (wdw["stats"].get("droppedReasons") or {}).items():
                reasons[k] = reasons.get(k, 0) + v
        print("BROWSER " + json.dumps(reasons), flush=True)
    finally:
        for n in names:
            try:
                rest_request(args.rest, "POST", "/window/close", {"id": n})
            except Exception:
                pass
        finish(avp, nodes)


if __name__ == "__main__":
    main()
