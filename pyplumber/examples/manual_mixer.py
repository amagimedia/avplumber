"""Minimal manual video mixer example using MixerGraphBuilder.
Demonstrates MixerGraphBuilder reusability for non-automatic applications.
Two looped MP4 inputs are decoded, fed through the mixer, encoded, and
written to an RTMP or file output.  A tiny stdin REPL accepts commands:
    cut <scene>               -- hard cut
    fade <scene> [secs]       -- crossfade (default 1.0 s)
    wipe <scene> <file>       -- media wipe
    scenes                    -- list registered scenes
    status                    -- print current PGM scene
    quit                      -- exit
Usage
-----
    python manual_mixer.py \\
        --input1 /path/to/a.mp4 \\
        --input2 /path/to/b.mp4 \\
        --output rtmp://host/app/stream  \\
        [--wipe-file /path/to/wipe.mov]
This example mirrors examples/mixer.avplumber with a 1920x1080 canvas so
existing wipe animations and tooling work without modification.
"""
import argparse
import sys
import threading
# Ensure pyplumber package is importable (adjust path as needed).
sys.path.insert(0, ".")
import pyplumber
from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeVideoFormat,
    DecVideo,
    Demux,
    EncVideo,
    ForceFPS,
    InputRec,
    Mux,
    Output,
    Realtime,
)
from pyplumber.mixer import MixerGraphBuilder

CANVAS_W = 1920
CANVAS_H = 1080
FPS_NUM = 30
FPS_DEN = 1
HWACCEL = "@gpu"
def build_input(avp: AVPlumber, idx: int, url: str) -> str:
    """Build the decode chain for one looped MP4 input.
    Returns the name of the force_fps output edge.
    """
    g = f"input_{idx}"
    avp.addNode(InputRec({
        "name": f"input_{idx}",
        "url": url,
        "dst": f"in{idx}_pkt",
        "group": g,
        "loop": True,
        "initial_timeout": 20,
        "timeout": 3_942_000_000,
    }))
    avp.addNode(Demux({
        "src": f"in{idx}_pkt",
        "routing": {"?v:0": f"cam{idx}_pkt"},
        "wait_for_keyframe": False,
        "group": g,
        "auto_restart": "group",
    }))
    avp.addNode(DecVideo({
        "src": f"cam{idx}_pkt",
        "dst": f"cam{idx}_dec",
        "pixel_format": "?cuda",
        "hwaccel": HWACCEL,
        "group": g,
        "auto_restart": "group",
    }))
    avp.addNode(Realtime({
        "src": f"cam{idx}_dec",
        "dst": f"cam{idx}_rt",
        "set_pts": True,
        "group": g,
    }))
    avp.addNode(ForceFPS({
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "src": f"cam{idx}_rt",
        "dst": f"cam{idx}_fps",
        "group": g,
    }))
    return f"cam{idx}_fps"

def build_output(avp: AVPlumber, video_edge: str, output_url: str) -> None:
    """Add fps-normalizer, encoder, mux, and output nodes."""
    g = "output"
    avp.addNode(ForceFPS({
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "src": video_edge,
        "dst": "mixer_norm_fps",
        "group": g,
    }))
    avp.addNode(AssumeVideoFormat({
        "src": "mixer_norm_fps",
        "dst": "mixer_norm",
        "width": CANVAS_W,
        "height": CANVAS_H,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "group": g,
        "auto_restart": "panic",
    }))
    avp.addNode(EncVideo({
        "src": "mixer_norm",
        "dst": "v_enc",
        "name": "enc",
        "codec": "h264_nvenc",
        "hwaccel": HWACCEL,
        "group": g,
        "options": {
            "b": "8000k",
            "maxrate": "14000k",
            "bufsize": "20000k",
            "g": 60,
            "bf": 0,
            "preset": "p3",
            "profile": "high",
        },
    }))
    avp.addNode(Mux({
        "src": ["v_enc"],
        "dst": "mux_out",
        "group": g,
        "ts_sort_wait": 0,
    }))
    fmt = "flv" if output_url.startswith("rtmp://") else "mp4"
    avp.addNode(Output({
        "format": fmt,
        "url": output_url,
        "src": "mux_out",
        "group": g,
        "auto_restart": "panic",
    }))

def define_scenes(mx: MixerGraphBuilder, n: int) -> None:
    """Register a practical set of scenes for *n* inputs on a 1920x1080 canvas."""
    W, H = CANVAS_W, CANVAS_H
    for i in range(n):
        mx.add_scene(f"full_{i}", {
            f"cam{i}": {
                "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                "dst_x": 0,
                "dst_y": 0,
            }
        })
    if n >= 2:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pip_w, pip_h = W // 3, H // 3
                mx.add_scene(f"pip_{i}_main_{j}_small", {
                    f"cam{i}": {
                        "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                        "dst_x": 0,
                        "dst_y": 0,
                    },
                    f"cam{j}": {
                        "graph": f"scale_cuda=w={pip_w}:h={pip_h}:interp_algo=lanczos",
                        "dst_x": W - pip_w - 16,
                        "dst_y": 16,
                    },
                })
    if n >= 2:
        # Tiled 2-column layout; camera[j] at row j//2, col j%2
        tile_w, tile_h = W // 2, H // 2
        sources = {}
        for j in range(min(n, 4)):
            row, col = j // 2, j % 2
            sources[f"cam{j}"] = {
                "graph": f"scale_cuda=w={tile_w}:h={tile_h}:interp_algo=lanczos",
                "dst_x": col * tile_w,
                "dst_y": row * tile_h,
            }
        mx.add_scene("multiviewer", sources)

def repl(mx: MixerGraphBuilder) -> None:
    """Blocking stdin REPL.  Runs in the main thread after graph start."""
    print("Manual mixer REPL ready.  Commands:")
    print("  cut <scene>               hard cut")
    print("  fade <scene> [secs]       crossfade (default 1.0 s)")
    print("  wipe <scene> <file>       media wipe")
    print("  scenes                    list scenes")
    print("  status                    current PGM scene")
    print("  quit                      exit")
    while True:
        try:
            line = input("mixer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd == "cut" and len(parts) >= 2:
                mx.cut(parts[1])
                print(f"Cut → {parts[1]}")
            elif cmd == "fade" and len(parts) >= 2:
                secs = float(parts[2]) if len(parts) >= 3 else 1.0
                mx.fade(parts[1], duration_sec=secs)
                print(f"Fade → {parts[1]} ({secs:.1f} s)")
            elif cmd == "wipe" and len(parts) >= 3:
                mx.wipe(parts[1], wipe_file=parts[2])
                print(f"Wipe → {parts[1]} using {parts[2]}")
            elif cmd == "scenes":
                print("  " + "\n  ".join(mx.scenes()))
            elif cmd == "status":
                print(f"PGM: {mx.current_scene}")
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print(f"Unknown command: {line}")
        except Exception as exc:
            print(f"Error: {exc}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input1", required=True, help="Path or URL for camera 1")
    parser.add_argument("--input2", required=True, help="Path or URL for camera 2")
    parser.add_argument("--output", required=True, help="RTMP URL or file path for output")
    parser.add_argument("--wipe-file", default="", help="Default wipe animation file (optional)")
    args = parser.parse_args()
    avp = AVPlumber()
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    avp.edges.planCapacity("*", 3)
    inputs = [args.input1, args.input2]
    # Build input chains.
    fps_edges = [build_input(avp, i, url) for i, url in enumerate(inputs)]
    # Build mixer.
    mx = MixerGraphBuilder(
        avp,
        name="mixer",
        canvas=(CANVAS_W, CANVAS_H),
        fps=(FPS_NUM, FPS_DEN),
        hwaccel=HWACCEL,
        enable_wipe=bool(args.wipe_file),
    )
    for i, fps_edge in enumerate(fps_edges):
        mx.add_source(f"cam{i}", pre_otm_edge=fps_edge, input_group=f"input_{i}")
    define_scenes(mx, len(inputs))
    mx.set_initial_scene("full_0", slot="A")
    out_edge = mx.build()
    # Build output chain.
    build_output(avp, out_edge, args.output)
    # Start groups.
    for i in range(len(inputs)):
        avp.group(f"input_{i}").startNodes()
    mx.start_groups()
    avp.group("output").startNodes()
    print("Graph started.  Streaming to", args.output)
    repl(mx)
if __name__ == "__main__":
    main()
