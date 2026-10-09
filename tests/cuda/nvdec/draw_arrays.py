"""Detection draw nodes on linear CUDA decode and on NVDEC CUarray decode.

One clip is decoded to linear CUDA frames without drawing (`plain`) and then, on linear CUDA and
on CUarray frames, through four graphs. A Python node attaches the same synthetic metadata to
every frame of both storages. The downloads are the verification boundary.

| Case | Draw nodes | Metadata |
|---|---|---|
| `undrawn` | draw_bbox -> draw_bbox_labels -> draw_trail | none: no node has anything to draw |
| `mixed` | the same chain | boxes, track ids, velocities and a trail under one key |
| `boxes3` | three draw_bbox, one key each | overlapping boxes in three colours, a viewport rectangle |
| `boxes1` | one draw_bbox with the three keys | the same |
| `chain4` | draw_bbox -> draw_bbox_labels -> draw_keypoints -> draw_trail | boxes, labels, a trail, and keypoints over all of them |
| `ml4` | one ml_debug with those four as layers | the same |
| `mlboxes` | one ml_debug with three box layers | the metadata of `boxes3` |

Checked:
- `undrawn` equals `plain` in pixels and timestamps: the picture a chain is drawn on (the copy of
  a linear frame, the linear picture of a CUarray) is the decoded picture, and a node with nothing
  to draw leaves it alone;
- every frame of the drawn cases differs from `plain`, so nothing below compares two empty draws;
- `boxes3` equals `boxes1`: three nodes drawing on one picture give what one node gives that
  draws the same boxes in the same order on its own copy;
- `ml4` equals `chain4` and `mlboxes` equals `boxes3`: one ml_debug pass gives the frame the chain
  of draw nodes gives, where boxes, a blended label background, text, dots and a trail overlap; and
  it visited fewer than half of the frame's tiles to do so (`node.object.get <node> draw`);
- every case on CUarray input is a linear CUDA frame, byte-identical to the same case on linear
  decode, with the same timestamps and the metadata still attached;
- the first node of a chain made every picture (`copied` on linear input, `from_array` on
  CUarray) and the nodes below drew on their input frame at least once (`in_place`). The split
  between `in_place` and `copied` below the first node is printed, not fixed: a node copies
  whenever its input is still referenced, and the node above releases its own reference only
  after it has put the frame;
- the decoder's arrays were reused, so the first node gave its surfaces back.
Identical pixels over inter-predicted frames also mean no node wrote into a decoder surface: a
drawn-on reference picture would show in the frames predicted from it.

The counters are read from `node.object.get <node> pictures`. A finished node is destroyed, so
the Python node holds the end of the stream back until they have been read. No wiretap sits
between two draw nodes: it would reference the frame and force the copy it was meant to observe.

Against an older build: `--linear-only --report OLD.json` runs the linear cases without reading
counters (a build before the CUarray support has neither), and `--expect OLD.json` on the new
build requires its linear cases to equal that report frame by frame.

Run on the NVIDIA host with its built module (NEURAL_NET=1, HAVE_NVCC=1) on PYTHONPATH and
--input CLIP: 8-bit 4:2:0, exactly --frames frames, with inter-predicted frames, and coded at its
display size. The copy of a linear frame has the size of its frames context, which for a decoder
is the coded size, while the picture of a CUarray has the frame's: on a clip coded larger than it
is displayed the two storages give frames of different sizes, and the rows below the picture are
not copied. 1080p H.264 is always coded 1088 high; 1080p HEVC is coded 1080 or 1088 high depending
on the encoder (ffprobe's coded_height says which); 720p is coded 720 high by both. Every case
requires its downloaded frames to have the decoded size and names the two sizes when they differ.

The metadata is compared with what the Python node attached, which it records: a payload is made
for the size of the decoded frame the node saw.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import threading
import time

from compositor_decode import EOF, api, make_avp, pixels, shutdown

KEY = "smoke_detections"
POSE = "smoke_pose"
BOX_KEYS = ("smoke_a", "smoke_b", "smoke_c")
COLORS = {"A": "red", "B": "yellow", "C": "cyan"}
DRAW = {"draw_bbox": api.DrawBBox, "draw_bbox_labels": api.DrawBBoxLabels, "draw_trail": api.DrawTrail,
        "draw_keypoints": api.DrawKeypoints, "ml_debug": getattr(api, "MlDebug", None)}   # no ml_debug in older builds
# A chain is a list of (node name, node type, parameters).
MIXED = [(kind, kind, {"metadata_key": KEY}) for kind in ("draw_bbox", "draw_bbox_labels", "draw_trail")]
BOXES3 = [(f"draw_{key[-1]}", "draw_bbox", {"metadata_key": key, "label_colors": COLORS}) for key in BOX_KEYS]
BOXES1 = [("draw_all", "draw_bbox", {"metadata_keys": list(BOX_KEYS), "label_colors": COLORS})]
DOTS = {"metadata_key": POSE, "radius": 3, "color": "green"}
CHAIN4 = [("draw_bbox", "draw_bbox", {"metadata_key": KEY}), ("draw_bbox_labels", "draw_bbox_labels", {"metadata_key": KEY}),
          ("draw_keypoints", "draw_keypoints", DOTS), ("draw_trail", "draw_trail", {"metadata_key": KEY})]
ML4 = [("ml", "ml_debug", {"layers": [
    {"kind": "boxes", "metadata_key": KEY}, {"kind": "labels", "metadata_key": KEY},
    {"kind": "keypoints", **DOTS}, {"kind": "trail", "metadata_key": KEY}]})]
MLBOXES = [("ml", "ml_debug", {"layers": [
    {"kind": "boxes", "metadata_key": key, "label_colors": COLORS} for key in BOX_KEYS]})]


def detections(index, width, height):
    """Three boxes, a label for each and a trail, all moving with the frame number."""
    step = height // 64 * (index % 16)
    boxes = [(width // 16 + step, height // 12, width // 4 + step, height // 2),
             (width // 2, height // 4 + step, width // 2 + width // 8, height // 4 + step + height // 3),
             (width - width // 5, height - height // 4 - step, width - 16, height - 16)]
    return {KEY: json.dumps({
        "coord_space": "frame", "model_width": width, "model_height": height,
        "detections": [{"xyxy": list(box), "conf": 0.9, "cls": 0, "label": "PLAYER", "track_id": number + 1,
                        "velocity": [1.5 * number, -0.5]} for number, box in enumerate(boxes)],
        "trail": [[width // 20 * (point + 1) + step, height // 2 + (height // 10 if point % 2 else -height // 10)]
                  for point in range(12)],
    })}


def everything(index, width, height):
    """The boxes, labels and trail of `detections`, a box thinner than its border, a label that
    flips under its box at the top edge, and 120 keypoints in a ring that crosses boxes, label
    backgrounds, the trail and the right and bottom edges of the frame."""
    payload = json.loads(detections(index, width, height)[KEY])
    step = height // 64 * (index % 16)
    payload["detections"] += [
        {"xyxy": [width // 3, 2, width // 3 + 3, height // 5], "conf": 0.8, "cls": 1, "label": "THIN", "track_id": 7,
         "velocity": [0.0, 0.0]},
        {"xyxy": [width // 2 + step, 1, width // 2 + step + width // 6, height // 6], "conf": 0.7, "cls": 2,
         "label": "TOP", "track_id": 8, "velocity": [-2.25, 3.0]}]
    centre_x, centre_y, radius = width * 0.72 + step, height * 0.68, height * 0.36
    ring = []
    for point in range(120):
        angle = point * 2 * math.pi / 120
        ring += [centre_x + radius * math.cos(angle) + 0.37 * (point % 3), centre_y + radius * math.sin(angle),
                 0.2 + 0.8 * (point % 5 != 0)]
    return {KEY: json.dumps(payload),
            POSE: json.dumps({"model_width": width, "model_height": height, "num_keypoints": 120,
                              "poses": [{"keypoints": ring}]})}


def boxes(index, width, height):
    """One box per key, each overlapping the next, so the order they are drawn in shows. The first
    key also carries a viewport, which draw_bbox draws as a white rectangle before its boxes."""
    step = width // 96 * (index % 16)
    payloads = {}
    for number, (key, label) in enumerate(zip(BOX_KEYS, COLORS)):
        left, top = width // 8 + number * width // 10 + step, height // 8 + number * height // 10
        payloads[key] = {"coord_space": "frame", "detections": [
            {"xyxy": [left, top, left + width // 4, top + height // 3], "conf": 0.9, "cls": number, "label": label}]}
    payloads[BOX_KEYS[0]].update(
        viewport_bbox=[width // 4 + step, height // 4, width // 4 + step + width // 2, height // 4 + height // 2],
        viewport_dst_width=width // 4 * 2, viewport_dst_height=height // 4 * 2,
        full_frame_width=width, full_frame_height=height)
    return {key: json.dumps(payload) for key, payload in payloads.items()}


def nothing(index, width, height):
    return {}


class Mark(api.PythonNode):
    """Attaches the metadata and records it in `attached`, one payload per frame, before the frame
    goes on; holds the end of the stream until `release` is set."""
    release = None
    payloads = None
    attached = None

    def process(self):
        frame = self._src.get()
        if frame.pts.timestamp == EOF:
            self.release.wait(120)
        else:
            payload = self.payloads(len(self.attached), frame.width, frame.height)
            for key, value in payload.items():
                frame.metadata[key] = value
            self.attached.append(payload)
        self._dst.enqueue(frame)

    def doStop(self):
        # The framework took the end marker itself and forwards it when this returns.
        self.release.wait(120)


def run(args, name, storage, chain=(), payloads=nothing, counters=True):
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
    attached = []
    edge = "decoded"
    if chain:
        mark = Mark({"name": "mark", "src": edge, "dst": "marked"})
        mark.release, mark.payloads, mark.attached = release, payloads, attached
        nodes.append(mark)
        edge = "marked"
        for node, kind, parameters in chain:
            nodes.append(DRAW[kind]({"name": node, "src": edge, "dst": f"out_{node}", **parameters}))
            edge = f"out_{node}"
    nodes.append(api.FilterVideo({"name": "verify", "src": edge, "dst": "result", "hwaccel": "draw_gpu",
                                  "graph": "hwdownload,format=nv12", "threads": 1,
                                  "defer_preliminary_init": True}))
    surfaces, formats = set(), set()
    decoded_sizes, output_sizes = set(), set()

    def decoded(frame):
        if frame.pts.timestamp != EOF:
            formats.add(("decoded", frame.format.name))
            surfaces.add(frame.data_ptr[0])
            decoded_sizes.add((frame.width, frame.height))

    def drawn(frame):
        if frame.pts.timestamp != EOF:
            formats.add(("drawn", frame.format.name))

    result, pictures, draws, ended = [], {}, {}, False
    output = frame = None
    try:
        for node in nodes:
            node.parameters.update(group="check", auto_restart="off")
            avp.addNode(node)
        del node
        avp.getEdge("decoded", "VideoFrame").addWiretapCallback(decoded)
        if chain:
            avp.getEdge(edge, "VideoFrame").addWiretapCallback(drawn)   # below the last draw node
        output = avp.getEdge("result", "VideoFrame")
        avp.group("check").startNodes()
        deadline = time.monotonic() + args.timeout
        while not ended and time.monotonic() < deadline and not errors:
            if len(result) == args.frames and not release.is_set():
                # Every frame has left the chain and no node has seen the end of the stream.
                if counters:
                    pictures = {node: dict(avp.node(node).getObject("pictures")) for node, _, _ in chain}
                    draws = {node: dict(avp.node(node).getObject("draw")) for node, kind, _ in chain if kind == "ml_debug"}
                release.set()
            frame = output.tryGet(10)
            if frame is None:
                continue
            if frame.pts.timestamp == EOF:
                ended = True
                continue
            # The node recorded the payload before it passed the frame on, so it is there by now.
            for key, value in (attached[len(result)] if chain else {}).items():
                assert key in frame.metadata, f"{name} on {storage}: frame {len(result)} lost metadata {key}"
                assert frame.metadata[key] == value, (
                    f"{name} on {storage}: frame {len(result)} carries another {key} than was attached: "
                    f"{frame.metadata[key]} != {value}")
            output_sizes.add((frame.width, frame.height))
            result.append(pixels(frame))
        assert not errors, errors
        assert ended, f"{name} on {storage}: missing EOF after {len(result)} frames"
        assert len(result) == args.frames, (name, storage, len(result), args.frames)
        # Compared here and not per frame: a wiretap runs after its frame was queued, so only the
        # end of the stream says that the decoded sizes are complete.
        assert len(decoded_sizes) == 1 and output_sizes == decoded_sizes, (
            f"{name} on {storage}: decoded frames are {sorted(decoded_sizes)} (width, height) and downloaded "
            f"frames {sorted(output_sizes)}. A draw node's copy of a linear frame has the size of the decoder's "
            f"frames context, so {args.input} is coded larger than it is displayed and its cases cannot be "
            f"compared; use a clip coded at its display size")
        assert len({item[:3] for item in result}) == args.frames, f"{name} on {storage}: duplicate output PTS"
        assert ("decoded", storage) in formats and len({f for f in formats if f[0] == "decoded"}) == 1, formats
        if chain:
            assert {f for f in formats if f[0] == "drawn"} == {("drawn", "cuda")}, formats
        if storage == "cuarray":
            assert len(surfaces) < args.frames, "CUarray surface pool did not reuse any array"
        return {"name": name, "storage": storage, "chain": [node for node, _, _ in chain], "frames": args.frames,
                "unique_surfaces": len(surfaces), "pictures": pictures, "draws": draws,
                "size": sorted(decoded_sizes)[0], "outputs": result}
    finally:
        release.set()
        frame = output = None
        shutdown(avp, nodes)
        del avp
        gc.collect()


def differing(left, right):
    """PTS of the frames that differ in pixels or timestamps; lists and tuples compare alike."""
    assert len(left) == len(right), (len(left), len(right))
    return [a[0] for a, b in zip(left, right) if list(a) != list(b)]


def check_pictures(case):
    """The first node made every picture; the nodes below made one only for a frame that was still
    referenced. Returns how many pictures the nodes below made."""
    frames, (first, *later) = case["frames"], case["chain"]
    made = "from_array" if case["storage"] == "cuarray" else "copied"
    expected = {"from_array": 0, "copied": 0, "in_place": 0, made: frames}
    assert case["pictures"][first] == expected, (case["name"], case["storage"], first, case["pictures"][first])
    for node in later:
        pictures = case["pictures"][node]
        assert pictures["from_array"] == 0 and pictures["copied"] + pictures["in_place"] == frames, (node, pictures)
        assert pictures["in_place"] > 0, f"{case['name']} on {case['storage']}: {node} never drew on its input frame"
    return sum(case["pictures"][node]["copied"] for node in later)


def cases(args, storage, plain, counters):
    runs = {name: run(args, name, storage, chain, payloads, counters)
            for name, chain, payloads in (("undrawn", MIXED, nothing), ("mixed", MIXED, detections),
                                          ("boxes3", BOXES3, boxes), ("boxes1", BOXES1, boxes),
                                          ("chain4", CHAIN4, everything))}
    different = differing(runs["undrawn"]["outputs"], plain["outputs"])
    assert not different, f"{storage}: {len(different)} undrawn frames differ from the decode (PTS {different[:8]})"
    for name in ("mixed", "boxes3", "boxes1", "chain4"):
        different = differing(runs[name]["outputs"], plain["outputs"])
        assert len(different) == args.frames, f"{storage} {name}: nothing drawn on {args.frames - len(different)} frames"
    different = differing(runs["boxes3"]["outputs"], runs["boxes1"]["outputs"])
    assert not different, f"{storage}: three draw_bbox differ from one on {len(different)} frames (PTS {different[:8]})"
    if DRAW["ml_debug"] is None:
        print(f"SKIP {storage}: this build has no ml_debug node", flush=True)
    else:
        runs.update({name: run(args, name, storage, chain, payloads, counters)
                     for name, chain, payloads in (("ml4", ML4, everything), ("mlboxes", MLBOXES, boxes))})
        for chain, one in (("chain4", "ml4"), ("boxes3", "mlboxes")):
            different = differing(runs[one]["outputs"], plain["outputs"])
            assert len(different) == args.frames, f"{storage} {one}: nothing drawn on {args.frames - len(different)} frames"
            different = differing(runs[chain]["outputs"], runs[one]["outputs"])
            assert not different, (f"{storage}: one ml_debug ({one}) differs from the chain of draw nodes ({chain}) "
                                   f"on {len(different)} frames (PTS {different[:8]})")
        if counters:
            width, height = runs["ml4"]["size"]
            tiles = ((width + 15) // 16) * ((height + 15) // 16)
            for name in ("ml4", "mlboxes"):
                draw = runs[name]["draws"]["ml"]
                assert draw["frames"] == args.frames and 0 < draw["tiles"] < tiles // 2, (name, draw, tiles)
            print(f"PASS {storage}: one ml_debug pass equals the chain of four draw nodes and three draw_bbox; "
                  f"last frame {json.dumps({name: runs[name]['draws']['ml'] for name in ('ml4', 'mlboxes')})} "
                  f"of {tiles} tiles", flush=True)
    if counters:
        extra = {name: check_pictures(case) for name, case in runs.items()}
        print(f"PASS {storage} pictures: one per frame by the first node of every chain; made again below it "
              f"(input still referenced) {json.dumps(extra)} of {args.frames} frames per case; "
              f"{json.dumps({name: case['pictures'] for name, case in runs.items()})}", flush=True)
    print(f"PASS {storage}: undrawn chain equals the decode, three draw_bbox equal one, "
          f"{args.frames} frames per case, metadata and PTS kept", flush=True)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    # Decoder surfaces are held by the decoded edge, the Python node, its output edge and the
    # first draw node: up to eight with edges of three.
    parser.add_argument("--extra-hw-frames", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--linear-only", action="store_true",
                        help="only the linear decode cases, without reading the pictures counters")
    parser.add_argument("--expect", type=Path,
                        help="a report of another build: its linear cases must equal this build's")
    args = parser.parse_args()

    plain = run(args, "plain", "cuda")
    linear = cases(args, "cuda", plain, counters=not args.linear_only)
    report = [plain, *linear.values()]
    if args.expect:
        compared = 0
        for other in json.loads(args.expect.read_text()):
            if other["storage"] != "cuda":
                continue
            if other["name"] != "plain" and other["name"] not in linear:
                continue      # a case this build does not run under that name
            mine = plain if other["name"] == "plain" else linear[other["name"]]
            different = differing(mine["outputs"], other["outputs"])
            assert not different, \
                f"{other['name']}: {len(different)} frames differ from {args.expect} (PTS {different[:8]})"
            compared += 1
        # An older build's report has the cases that build could run: every one of them must match.
        assert compared >= 5, f"{args.expect} has only {compared} linear cases"
        print(f"PASS linear cases equal {args.expect}: {compared} cases of {args.frames} frames", flush=True)
    if not args.linear_only:
        arrays = cases(args, "cuarray", plain, counters=True)
        for name, case in arrays.items():
            different = differing(case["outputs"], linear[name]["outputs"])
            assert not different, (f"{name}: {len(different)} of {args.frames} CUarray frames differ from linear "
                                   f"decode in pixels or PTS (PTS {different[:8]})")
        print(f"PASS CUarray equals linear decode in every case; "
              f"{arrays['mixed']['unique_surfaces']} array handles reused", flush=True)
        report += arrays.values()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
