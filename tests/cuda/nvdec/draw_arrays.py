"""Detection draw nodes on NVDEC CUarray frames against the same drawing on linear CUDA decode.

One clip is decoded three times: to linear CUDA frames without drawing, to linear CUDA frames
through draw_bbox -> draw_bbox_labels -> draw_trail, and to CUarray frames through the same chain.
A Python node attaches the same synthetic detections (boxes, track ids, velocities, a trail) to
every frame of both drawn runs. The downloads are the verification boundary.

Checked:
- the chain's output on CUarray input is a linear CUDA frame, byte-identical to the chain's output
  on linear decode, with the same timestamps and the detection metadata still attached;
- every drawn frame differs from the undrawn decode, so the comparison is not of two empty draws;
- on CUarray input the first node made every picture (`from_array`) and the nodes below drew on
  their input frame at least once (`in_place`); the split between `in_place` and `copied` is
  printed, because a node copies whenever its input is still referenced elsewhere;
- on linear input every node copied every frame, as before;
- the decoder's arrays were reused, so the first node gave its surfaces back.
Identical pixels over inter-predicted frames also mean no node wrote into a decoder surface: a
drawn-on reference picture would show in the frames predicted from it.

The counters are read from `node.object.get <node> pictures`. A finished node is destroyed, so
the Python node holds the end of the stream back until they have been read. No wiretap sits
between two draw nodes: it would reference the frame and force the copy it was meant to observe.

Run on the NVIDIA host with its built module (NEURAL_NET=1, HAVE_NVCC=1) on PYTHONPATH and
--input CLIP, an 8-bit 4:2:0 clip of exactly --frames frames with inter-predicted frames.
"""
import argparse
import gc
import json
from pathlib import Path
import threading
import time

from compositor_decode import EOF, api, make_avp, pixels, shutdown

KEY = "smoke_detections"
CHAIN = ("draw_bbox", "draw_bbox_labels", "draw_trail")


def detections(index, width, height):
    """Three boxes, a label for each and a trail, all moving with the frame number."""
    step = height // 64 * (index % 16)
    boxes = [(width // 16 + step, height // 12, width // 4 + step, height // 2),
             (width // 2, height // 4 + step, width // 2 + width // 8, height // 4 + step + height // 3),
             (width - width // 5, height - height // 4 - step, width - 16, height - 16)]
    return json.dumps({
        "coord_space": "frame", "model_width": width, "model_height": height,
        "detections": [{"xyxy": list(box), "conf": 0.9, "cls": 0, "label": "PLAYER", "track_id": number + 1,
                        "velocity": [1.5 * number, -0.5]} for number, box in enumerate(boxes)],
        "trail": [[width // 20 * (point + 1) + step, height // 2 + (height // 10 if point % 2 else -height // 10)]
                  for point in range(12)],
    })


class Mark(api.PythonNode):
    """Attaches the detections; holds the end of the stream until `release` is set."""
    release = None
    count = 0

    def process(self):
        frame = self._src.get()
        if frame.pts.timestamp == EOF:
            self.release.wait(120)
        else:
            frame.metadata[KEY] = detections(self.count, frame.width, frame.height)
            self.count += 1
        self._dst.enqueue(frame)

    def doStop(self):
        # The framework took the end marker itself and forwards it when this returns.
        self.release.wait(120)


def run(args, storage, draw):
    avp, errors = make_avp("draw_gpu", capacity=3)
    options = {"threads": 1}
    if storage == "cuarray":
        options.update(hwaccel_flags="unsafe_output", extra_hw_frames=args.extra_hw_frames)
    nodes = [
        api.Input({"name": "input", "url": str(args.input), "dst": "packets"}),
        api.Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        api.DecVideo({"name": "decode", "src": "video_packets", "dst": "decoded",
                      "hwaccel": "draw_gpu", "pixel_format": storage, "options": options}),
    ]
    release = threading.Event()
    edge = "decoded"
    if draw:
        mark = Mark({"name": "mark", "src": edge, "dst": "marked"})
        mark.release = release
        nodes.append(mark)
        edge = "marked"
        for kind in CHAIN:
            node = {"draw_bbox": api.DrawBBox, "draw_bbox_labels": api.DrawBBoxLabels, "draw_trail": api.DrawTrail}[kind]
            nodes.append(node({"name": kind, "src": edge, "dst": f"out_{kind}", "metadata_key": KEY}))
            edge = f"out_{kind}"
    nodes.append(api.FilterVideo({"name": "verify", "src": edge, "dst": "result", "hwaccel": "draw_gpu",
                                  "graph": "hwdownload,format=nv12", "threads": 1,
                                  "defer_preliminary_init": True}))
    surfaces, formats = set(), set()

    def decoded(frame):
        if frame.pts.timestamp != EOF:
            formats.add(("decoded", frame.format.name))
            surfaces.add(frame.data_ptr[0])

    def drawn(frame):
        if frame.pts.timestamp != EOF:
            formats.add(("drawn", frame.format.name))

    result, counters, ended = [], {}, False
    output = frame = None
    try:
        for node in nodes:
            node.parameters.update(group="check", auto_restart="off")
            avp.addNode(node)
        del node
        avp.getEdge("decoded", "VideoFrame").addWiretapCallback(decoded)
        if draw:
            avp.getEdge(edge, "VideoFrame").addWiretapCallback(drawn)   # below the last draw node
        output = avp.getEdge("result", "VideoFrame")
        avp.group("check").startNodes()
        deadline = time.monotonic() + args.timeout
        while not ended and time.monotonic() < deadline and not errors:
            if len(result) == args.frames and not release.is_set():
                # Every frame has left the chain and no node has seen the end of the stream.
                counters = {kind: dict(avp.node(kind).getObject("pictures")) for kind in CHAIN} if draw else {}
                release.set()
            frame = output.tryGet(10)
            if frame is None:
                continue
            if frame.pts.timestamp == EOF:
                ended = True
                continue
            if draw:
                assert frame.metadata[KEY] == detections(len(result), frame.width, frame.height), \
                    f"frame {len(result)} lost its detection metadata"
            result.append(pixels(frame))
        assert not errors, errors
        assert ended, f"missing EOF after {len(result)} frames"
        assert len(result) == args.frames, (len(result), args.frames)
        assert len({item[:3] for item in result}) == args.frames, "duplicate output PTS"
        assert ("decoded", storage) in formats and len({f for f in formats if f[0] == "decoded"}) == 1, formats
        if draw:
            assert {f for f in formats if f[0] == "drawn"} == {("drawn", "cuda")}, formats
        if storage == "cuarray":
            assert len(surfaces) < args.frames, "CUarray surface pool did not reuse any array"
        return {"storage": storage, "draw": draw, "frames": args.frames, "unique_surfaces": len(surfaces),
                "pictures": counters, "outputs": result}
    finally:
        release.set()
        frame = output = None
        shutdown(avp, nodes)
        del avp
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    # Decoder surfaces are held by the decoded edge, the Python node, its output edge and the
    # first draw node: up to eight with edges of three.
    parser.add_argument("--extra-hw-frames", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    plain = run(args, "cuda", draw=False)
    linear = run(args, "cuda", draw=True)
    assert [item[:3] for item in linear["outputs"]] == [item[:3] for item in plain["outputs"]], "drawing changed PTS"
    same = [item[0] for item, undrawn in zip(linear["outputs"], plain["outputs"]) if item == undrawn]
    assert not same, f"nothing was drawn on {len(same)} frames (PTS {same[:8]})"
    for kind, pictures in linear["pictures"].items():
        assert pictures == {"from_array": 0, "copied": args.frames, "in_place": 0}, (kind, pictures)
    print(f"PASS linear CUDA: {args.frames} frames drawn by {' -> '.join(CHAIN)}, every node copied every frame, "
          "metadata and PTS kept", flush=True)

    array = run(args, "cuarray", draw=True)
    different = [item[0] for item, expected in zip(array["outputs"], linear["outputs"]) if item != expected]
    assert len(array["outputs"]) == len(linear["outputs"]) and not different, \
        f"{len(different)} of {args.frames} CUarray frames differ from linear decode in pixels or PTS (PTS {different[:8]})"
    first, *later = CHAIN
    assert array["pictures"][first] == {"from_array": args.frames, "copied": 0, "in_place": 0}, array["pictures"]
    for kind in later:
        pictures = array["pictures"][kind]
        assert pictures["from_array"] == 0 and pictures["copied"] + pictures["in_place"] == args.frames, (kind, pictures)
        assert pictures["in_place"] > 0, f"{kind} never drew on its input frame: {pictures}"
    print(f"PASS CUarray: {args.frames} frames byte-identical to the linear run, PTS and metadata kept, "
          f"{array['unique_surfaces']} array handles reused; pictures {json.dumps(array['pictures'])}", flush=True)
    if args.report:
        args.report.write_text(json.dumps([plain, linear, array], indent=2) + "\n")


if __name__ == "__main__":
    main()
