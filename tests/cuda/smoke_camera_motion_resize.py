"""GPU regression: NVOF must reseed after 720p -> 1080p -> 720p transitions.

The finite test patterns are uploaded once into the same CUDA device. This is
fixture setup; the camera-motion path consumes and forwards CUDA frames only.
Run in an NVIDIA container with HAVE_NVOF and HAVE_CCM_GPU_IRLS enabled.
"""

import argparse
import json
import time

from _avplumber import VideoFrame
from pyplumber import AVPlumber
from pyplumber.node import CudaCameraMotion, DecVideo, Demux, FilterVideo, InputRec


def run(timeout):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.executeCommandsFromString('hwaccel.init {"name":"flow_gpu","type":"cuda"}')
    avp.edges.planCapacity("*", 8)
    nodes = []
    for label, width, height in [("hd", 1280, 720), ("fhd", 1920, 1080)]:
        nodes.extend([
            InputRec({"name": f"{label}_input", "format": "lavfi",
                      "url": f"testsrc2=size={width}x{height}:rate=25,trim=end_frame=2",
                      "dst": f"{label}_packets"}),
            Demux({"name": f"{label}_demux", "src": f"{label}_packets",
                   "routing": {"v:0": f"{label}_video"}}),
            DecVideo({"name": f"{label}_decode", "src": f"{label}_video",
                      "dst": f"{label}_software"}),
            FilterVideo({"name": f"{label}_upload", "src": f"{label}_software",
                         "dst": f"{label}_cuda", "graph": "format=nv12,hwupload",
                         "hwaccel": "flow_gpu", "dst_pixel_format": "cuda",
                         "dst_width": width, "dst_height": height}),
        ])
    for node in nodes:
        node.parameters.update({"group": "fixtures", "auto_restart": "off"})
        avp.addNode(node)
    avp.getEdge("flow_input", "VideoFrame")
    flow = CudaCameraMotion({"name": "flow", "group": "flow", "auto_restart": "off",
                             "src": "flow_input", "dst": "flow_output",
                             "affine_backend": "gpu_irls", "strict_cuda": True})
    avp.addNode(flow)
    source = avp.getEdge("flow_input", "VideoFrame")
    output = avp.getEdge("flow_output", "VideoFrame")
    deadline = time.monotonic() + timeout

    def get(edge):
        while time.monotonic() < deadline:
            assert not errors, errors
            frame = edge.tryGet(100)
            if frame is not None:
                return frame
        raise AssertionError("timed out waiting for a CUDA frame")

    results = []
    frames = {}
    try:
        avp.group("fixtures").startNodes()
        for label in ("hd", "fhd"):
            edge = avp.getEdge(f"{label}_cuda", "VideoFrame")
            frames[label] = [get(edge), get(edge)]
            assert get(edge).pts.timestamp == -(1 << 63), "fixture did not reach EOF"
        avp.group("flow").startNodes()
        for label in ("hd", "fhd", "hd", "fhd"):
            for index, frame in enumerate(frames[label]):
                assert source.enqueue(frame)
                processed = get(output)
                motion = json.loads(processed.metadata["camera_motion"])
                assert (processed.width, processed.height) == (frame.width, frame.height)
                assert motion["status"] == "ok", motion
                assert motion["has_prev"] == (index != 0), motion
                if index:
                    assert motion["affine_valid"], motion
                results.append({"width": frame.width, "height": frame.height,
                                "has_prev": motion["has_prev"],
                                "affine_valid": motion["affine_valid"]})
        source.enqueue(VideoFrame())
        assert get(output).pts.timestamp == -(1 << 63)
        names = [node.parameters["name"] for node in nodes] + ["flow"]
        while any(avp.node(name).isWorking for name in names):
            assert time.monotonic() < deadline, "workers did not finish after EOF"
            time.sleep(0.01)
        assert not errors, errors
    finally:
        frames.clear()
        nodes.clear()
        avp.shutdown()
    print(json.dumps({"status": "passed", "transitions": 3, "frames": results}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30)
    run(parser.parse_args().timeout)
