"""Render the 10-bit 4:2:2 mixer to viewable files on the NVIDIA host.

Sixteen full-resolution v210 sources (gradient/ramp/HLG/SDR-promoted families)
compose as a 4x4 P210 grid; a second pass runs the static gradient fullscreen
through the copy path as the banding proof. Outputs land in --out:
  grid_h264_8bit.mp4       - plays anywhere (final 8-bit quantization)
  grid_hevc_main10.mp4     - true 10-bit 4:2:0 for mpv/VLC on a 10-bit display
  gradient_proof_h264.mp4  - fullscreen gradient, 4x contrast stretch BEFORE
                             the 8-bit encode: the 8-bit-quantized top half
                             must band ~4x wider than the true-10-bit bottom.
The unclocked composition rate is printed as a rough ingress+composite probe;
it is not the paced latency measurement.
"""

import argparse
from pathlib import Path
import subprocess
import time

import numpy as np

from v210_fixture import COLOR, frame_stride, write_fixture

FAMILY_RING = ("gradient", "ramp", "hlg", "sdr8")
LOOP_FRAMES = 30   # short files loop through InputRec; 16 x 240 full-HD v210 files would be 21 GB


def ensure_fixtures(root, width, height, sources, file_frames):
    paths = []
    for s in range(sources):
        family = FAMILY_RING[s % len(FAMILY_RING)]
        path = Path(root) / f"grid{s:02d}_{family}_{width}x{height}_{file_frames}f.v210"
        if not path.exists():
            print(f"generating {path.name}", flush=True)
            write_fixture(path, width, height, file_frames, family=family, source=s)
        paths.append((path, family))
    return paths


def compose(root, sink_path, width, height, frames, timeout, layout, loop):
    """Run one composition, appending raw contiguous P210 frames to sink_path.

    loop=True reads short files through InputRec; NOTE: stopping a looping
    InputRec -> v210_to_cuda chain mid-stream currently wedges shutdown
    (recorded follow-up), so loop passes rely on the container exiting.
    loop=False uses finite files sized to the target and shuts down cleanly.
    """
    from pyplumber import AVPlumber
    from pyplumber.node import CudaRectOverlay, Demux, FilterVideo, Input, InputRec, V210ToCuda

    sources = ensure_fixtures(root, width, height, layout, LOOP_FRAMES if loop else frames)
    stride = frame_stride(width)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"rndr_gpu","type":"cuda"}')
    nodes = []
    for s, (path, family) in enumerate(sources):
        input_cls, input_extra = (InputRec, {"loop": True}) if loop else (Input, {})
        nodes += [
            input_cls({"name": f"in{s}", "url": str(path), "format": "rawvideo", "dst": f"pkt{s}",
                       **input_extra,
                       "options": {"pixel_format": "gray", "video_size": f"{stride}x{height}",
                                   "framerate": "60"}}),
            Demux({"name": f"dx{s}", "src": f"pkt{s}", "routing": {"v:0": f"packed{s}"}}),
            V210ToCuda({"name": f"up{s}", "src": f"packed{s}", "dst": f"gpu{s}",
                        "hwaccel": "rndr_gpu", "width": width, "height": height,
                        "stride": stride, "fps": "60/1", "timebase": "1/90000",
                        "format": "p210le", **COLOR[family]}),
        ]
    cols = 4 if layout > 1 else 1
    tw, th = width // cols, height // cols
    nodes += [
        CudaRectOverlay({"name": "comp", "src": [f"gpu{s}" for s in range(layout)],
                         "dst": "scene", "hwaccel": "rndr_gpu", "width": width, "height": height,
                         "sw_format": "p210le", "scale": True, "active_inputs": (1 << layout) - 1,
                         "layers": [{"dst_x": (s % cols) * tw, "dst_y": (s // cols) * th,
                                     "dst_w": tw, "dst_h": th} for s in range(layout)]}),
        FilterVideo({"name": "down", "src": "scene", "dst": "raw", "hwaccel": "rndr_gpu",
                     "graph": "hwdownload,format=p210le"}),
    ]
    count = 0
    try:
        for node in nodes:
            node.parameters.update({"group": "render", "auto_restart": "off"})
            avp.addNode(node)
        del node
        out = avp.getEdge("raw", "VideoFrame")
        avp.group("render").startNodes()
        started = time.monotonic()
        deadline = started + timeout
        with Path(sink_path).open("wb") as sink:
            while time.monotonic() < deadline and not errors and count < frames:
                frame = out.tryGet(100)
                if frame is None:
                    continue
                if frame.pts.timestamp == -(1 << 63):
                    break
                for data, pitch in zip(frame.data[:2], frame.linesize[:2]):
                    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, pitch)
                    sink.write(rows[:, :width * 2].tobytes())
                count += 1
        elapsed = time.monotonic() - started
        assert not errors, errors
        assert count >= frames - 2, f"only {count}/{frames} frames"
        print(f"composed {count} frames of {layout} x {width}x{height} in "
              f"{elapsed:.1f}s ({count / elapsed:.1f} fps unclocked)", flush=True)
    finally:
        nodes.clear()
        avp.shutdown()


def encode(raw_path, width, height, filters, codec_args, out_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "p210le",
                    "-video_size", f"{width}x{height}", "-framerate", "60", "-i", str(raw_path),
                    "-vf", filters, *codec_args, str(out_path)], check=True, timeout=600)
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = args.out / "fixtures"
    fixtures.mkdir(exist_ok=True)

    for tag, layout, frames, loop in (("gradient", 1, args.frames // 2, False),
                                      ("grid", 16, args.frames, True)):
        raw = args.out / f"{tag}_p210.raw"
        if raw.exists() and raw.stat().st_size == frames * args.width * args.height * 4:
            print(f"reusing existing {raw.name}", flush=True)
        else:
            compose(fixtures, raw, args.width, args.height, frames, args.timeout, layout, loop)
        if tag == "grid":
            encode(raw, args.width, args.height, "format=nv12",
                   ["-c:v", "h264_nvenc", "-b:v", "12M"], args.out / "grid_h264_8bit.mp4")
            encode(raw, args.width, args.height, "format=p010le",
                   ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-b:v", "20M"],
                   args.out / "grid_hevc_main10.mp4")
        else:
            encode(raw, args.width, args.height,
                   "lutyuv=y='clip((val-64)*4,0,1023)',format=nv12",
                   ["-c:v", "h264_nvenc", "-b:v", "12M"], args.out / "gradient_proof_h264.mp4")
        raw.unlink()


if __name__ == "__main__":
    main()
