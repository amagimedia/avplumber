"""Bounded NVDEC -> metadata crop -> pad/scale/format conversion -> NVENC smoke.

Run on an isolated NVIDIA host with the patched FFmpeg libraries and a video
fixture at least 640x360. Frames stay on CUDA throughout the processing graph;
the independent software decode below only validates the encoded output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from pyplumber import AVPlumber
from pyplumber.node import (
    AssumeVideoFormat, CropMetadataCuda, DecVideo, Demux, EncVideo,
    FilterVideo, ForceFPS, InputRec,
)


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = args.output.with_suffix(".input.mkv")
    # Stream-copy a bounded encoded fixture; no pixel conversion is involved.
    subprocess.run([args.ffmpeg, "-v", "error", "-y", "-i", args.input,
                    "-map", "0:v:0", "-c", "copy", "-frames:v", str(args.frames + 16),
                    str(fixture)], check=True, timeout=30)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.executeCommandsFromString('hwaccel.init {"name":"filter_chain_gpu","type":"cuda"}')
    avp.edges.planCapacity("*", 4)
    graph = (
        f"pad_cuda=672:384:16:12:color=black,{args.scaler}=640:360,"
        "convert_cuda=format=yuv420p,convert_cuda=format=nv12"
    )
    if args.band_blur:
        graph += ",band_blur_cuda"
    nodes = [
        InputRec({"name": "input", "url": str(fixture), "loop": False, "dst": "packets"}),
        Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        DecVideo({"name": "decode", "src": "video_packets", "dst": "decoded",
                  "hwaccel": "filter_chain_gpu", "pixel_format": "cuda"}),
        # Empty metadata selects the crop node's documented center fallback.
        # The FFmpeg metadata filter edits properties, not CUDA pixel data.
        FilterVideo({"name": "metadata", "src": "decoded", "dst": "marked",
                     "graph": "metadata=mode=add:key=reframer_bbox:value='{}'"}),
        CropMetadataCuda({"name": "crop", "src": "marked", "dst": "cropped",
                          "dst_width": 640, "dst_height": 360,
                          "offset_log_path": str(args.output.with_suffix(".crop.log"))}),
        FilterVideo({"name": "chain", "src": "cropped", "dst": "filtered", "graph": graph,
                     "hwaccel": "filter_chain_gpu", "dst_width": 640, "dst_height": 360,
                     "dst_pixel_format": "cuda", "dst_frame_rate": "25/1",
                     "defer_preliminary_init": True}),
        # Match the recorder's static timebase boundary before NVENC. The
        # deferred CUDA filter has no output timebase until its first frame.
        ForceFPS({"name": "fps", "src": "filtered", "dst": "paced", "fps": "25/1"}),
        AssumeVideoFormat({"name": "format", "src": "paced", "dst": "nv12",
                           "width": 640, "height": 360, "pixel_format": "cuda",
                           "real_pixel_format": "nv12"}),
        EncVideo({"name": "encode", "src": "nv12", "dst": "encoded",
                  "hwaccel": "filter_chain_gpu", "codec": "h264_nvenc",
                  "options": {"preset": "p1", "tune": "ull", "bf": 0,
                              "rc-lookahead": 0, "delay": 0, "g": 25}}),
    ]
    for node in nodes:
        node.parameters.update({"group": "test", "auto_restart": "off"})
        avp.addNode(node)
    del node
    encoded = avp.getEdge("encoded", "Packet")
    packets = []
    try:
        avp.group("test").startNodes()
        deadline = time.monotonic() + args.timeout
        finished = False
        while time.monotonic() < deadline and not errors:
            packet = encoded.tryGet(100)
            if packet is None:
                continue
            # AVPlumber's packet EOF marker has a one-byte payload and NOPTS.
            if packet.pts.timestamp == -(1 << 63):
                finished = True
                break
            if packet.size > 0:
                if len(packets) < args.frames:
                    packets.append(packet.data)
        assert not errors, errors
        assert finished, "encoded stream did not reach EOF"
        assert len(packets) == args.frames, f"only {len(packets)}/{args.frames} encoded frames"
        # The encoder can enqueue EOF before its worker (or an upstream
        # decoder) has unwound. Await those EOF-driven workers before shutdown;
        # they deliberately have no interface for an abrupt mid-frame stop.
        eof_workers = [node.parameters["name"] for node in nodes
                       if node.parameters["name"] != "fps"]
        while any(avp.node(name).isWorking for name in eof_workers):
            assert time.monotonic() < deadline, "EOF workers did not finish"
            time.sleep(0.01)
    finally:
        # The crop node is EOF-driven, so use a finite fixture and drain the
        # stream instead of forcing a stop midway through its filter graph.
        nodes.clear()
        avp.shutdown()
    args.output.write_bytes(b"".join(packets))
    decoded = subprocess.run([
        args.ffmpeg, "-v", "error", "-threads", "1", "-f", "h264", "-i", str(args.output),
        "-vf", "scale=2:2", "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], check=True, capture_output=True, timeout=30).stdout
    assert len(decoded) == args.frames * 4, "independent decode frame count differs"
    print(json.dumps({"status": "passed", "frames": args.frames, "scaler": args.scaler,
                      "graph": graph, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--scaler", choices=("scale_cuda", "scale_npp"), default="scale_cuda")
    parser.add_argument("--band-blur", action="store_true", help="also check the reframer's optional band_blur_cuda patch")
    parser.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    run(parser.parse_args())
