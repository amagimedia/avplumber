"""Decoder crop + layer crop on linear CUDA and CUarray, through both rect consumers.

Run on an NVIDIA build host: python3 tests/cuda/nvdec/compositor_crop.py --ffmpeg /path/to/ffmpeg
The FFmpeg build needs h264_nvenc, h264_metadata and CUarray decode. CPU download is only
the pixel-verification boundary. The fixture's crop is inserted without re-encoding.

The draw_bbox consumer puts a NEURAL_NET draw node before the compositor: it reads a CUarray
through its own linear copy, which must keep the whole surface and the decoder crop so that
the compositor applies the crop once. --no-draw skips it for builds without NEURAL_NET.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import drain, make_avp
from compositor_decode import pixels, shutdown
from pyplumber import node as api

FRAMES = 6
VISIBLE = (240, 184)
ORIGIN = (8, 4)


def fixture(ffmpeg, directory):
    base, cropped = directory / "base.mp4", directory / "cropped.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run([*command, "-f", "lavfi", "-i", "testsrc2=size=256x192:rate=6",
                    "-frames:v", str(FRAMES), "-c:v", "h264_nvenc", "-preset", "p1",
                    "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", str(base)], check=True)
    subprocess.run([*command, "-i", str(base), "-c:v", "copy", "-bsf:v",
                    "h264_metadata=crop_left=8:crop_right=8:crop_top=4:crop_bottom=4",
                    str(cropped)], check=True)
    # Disable decoder cropping so software pointer-alignment rounding cannot weaken the oracle.
    raw = subprocess.check_output([*command, "-apply_cropping", "0", "-i", str(cropped),
                                   "-vf", "crop=240:184:8:4,format=nv12", "-f", "rawvideo", "-"])
    size = VISIBLE[0] * VISIBLE[1] * 3 // 2
    assert len(raw) == size * FRAMES
    hashes = [hashlib.sha256(raw[i:i + size]).hexdigest() for i in range(0, len(raw), size)]
    return base, cropped, hashes


def render(path, storage, consumer, layer, size):
    avp, errors = make_avp("crop_gpu")
    options = {"threads": 1}
    if storage == "cuarray":
        options.update(hwaccel_flags="unsafe_output", extra_hw_frames=3)
    nodes = [
        api.Input({"name": "input", "url": str(path), "dst": "packets"}),
        api.Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        api.DecVideo({"name": "decode", "src": "video_packets", "dst": "decoded",
                      "hwaccel": "crop_gpu", "pixel_format": storage, "options": options}),
    ]
    if consumer == "draw_bbox":
        # No detections: the node only passes its picture of the frame on. It reads the
        # decoder's format when created, which a deferred filter cannot give yet.
        nodes.append(api.DrawBBox({"name": "overlay", "src": "decoded", "dst": "source"}))
    else:
        # Identity scale_cuda forwards the original crop metadata, including on CUarray.
        nodes.append(api.FilterVideo({"name": "normalize", "src": "decoded", "dst": "source",
                                      "hwaccel": "crop_gpu", "threads": 1, "defer_preliminary_init": True,
                                      "graph": "scale_cuda=format=nv12"}))
    width, height = size
    canvas = {"width": width, "height": height, "sw_format": "nv12"}
    if consumer == "transform":
        nodes.append(api.CudaTransform({
            "name": "first", "src": "source", "dst": ["drawn"], "hwaccel": "crop_gpu",
            "outputs": [{**canvas, "dst": "drawn", "layers": [layer], "pass_arrays": True}]}))
    else:
        nodes.append(api.CudaRectOverlay({
            **canvas, "name": "first", "src": ["source"], "dst": "drawn",
            "hwaccel": "crop_gpu", "layers": [layer]}))
    # Rendering again at 1:1 catches inherited crop fields: those would crop the canvas twice.
    nodes += [
        api.CudaRectOverlay({**canvas, "name": "second", "src": ["drawn"], "dst": "canvas",
                             "hwaccel": "crop_gpu", "layers": [{"dst_w": width, "dst_h": height}]}),
        api.FilterVideo({"name": "download", "src": "canvas", "dst": "result",
                         "hwaccel": "crop_gpu", "threads": 1, "defer_preliminary_init": True,
                         "graph": "hwdownload,format=nv12"}),
    ]
    output = frame = None
    try:
        for node in nodes:
            node.parameters.update(group="check", auto_restart="off")
            avp.addNode(node)
        del node
        output = avp.getEdge("result", "VideoFrame")
        avp.group("check").startNodes()
        result, state = [], {}
        for frame in drain(output, errors, 20, state=state):
            assert (frame.width, frame.height) == size
            result.append(pixels(frame))
        assert not errors, errors
        assert state["eof"] and state["count"] == FRAMES, state
        return result
    finally:
        output = frame = None
        shutdown(avp, nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--no-draw", action="store_true", help="skip draw_bbox (no NEURAL_NET build)")
    args = parser.parse_args()
    consumers = ("compositor", "transform") + (() if args.no_draw else ("draw_bbox",))
    with tempfile.TemporaryDirectory(prefix="compositor-crop-") as temporary:
        base, cropped, software = fixture(args.ffmpeg, Path(temporary))
        cases = [
            ("visible", VISIBLE, None, False),
            ("layer_crop", (80, 64), (12, 8, 80, 64), False),
            ("scaled_crop", (40, 32), (12, 8, 80, 64), False),
            ("contain", (128, 128), None, True),
        ]
        for name, size, crop, contain in cases:
            x, y, w, h = crop or (0, 0, *VISIBLE)
            layer = {"dst_w": size[0], "dst_h": size[1], "filter": "bilinear",
                     "fit": "contain" if contain else "stretch"}
            if crop:
                layer["crop"] = dict(zip("xywh", crop))
            # The original bitstream + one explicit crop is an independent coordinate oracle.
            reference_layer = {**layer, "crop": dict(zip("xywh", (x + ORIGIN[0], y + ORIGIN[1], w, h)))}
            expected = render(base, "cuda", "compositor", reference_layer, size)
            if name == "visible":
                assert [row[-1] for row in expected] == software, "GPU/reference software pixels differ"
            for storage in ("cuda", "cuarray"):
                for consumer in consumers:
                    actual = render(cropped, storage, consumer, layer, size)
                    assert actual == expected, (name, storage, consumer, "pixels/PTS differ")
                    print(f"PASS {name}: {storage} -> {consumer} -> compositor", flush=True)


if __name__ == "__main__":
    main()
