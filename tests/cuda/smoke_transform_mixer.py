"""Remote GPU smoke for the two mixer uses of cuda_transform, built with pyplumber.transform.

1. Fit with black bars (source normalisation): a frame of any size contained in the canonical
   1920x1080 canvas, against the filter graph it replaced (scale_cuda, then pad_cuda).
   - Bars are exactly black and the picture fills exactly the box of the node's placement rule.
   - A source that needs no scaling is byte-identical to the filter graph.
   - A scaled source whose fitted size is even without rounding equals the filter graph within
     TOLERANCE. The scalers differ (four-neighbour bilinear here, scale_cuda's default there).
   - A source whose fitted size needs rounding may land in a different box: the node rounds the
     fitted size down to the chroma grid, the filter graph follows FFmpeg's rounding. The
     pictures are compared only when the boxes agree; otherwise the smoke prints both boxes.
   - An 8-bit source that needs no scaling, drawn onto a 10-bit canvas, is the same picture with
     every code times four.
2. A scaled output (rendition): the transform scales in the canvas format and the rendition's
   conversion filter follows, against scale_cuda followed by the same conversion in one filter
   graph, within TOLERANCE. One case only stamps tags (SDR to SDR), two tone-map HLG to SDR: from
   a p010le canvas and from a p210le canvas, the storage of the live HLG show.

Timestamps must survive: frames are matched by timestamp between the two paths. The downloads
are the verification boundary. Not covered: a 4:2:2, CUarray or browser frame as the input of a
normalisation.

The format the node declares is used but not asserted. Each download filter reads it when the
graph is built. transform_pillarbox has two canvases of different storage and so declares none:
its two download filters log "cuda_transform: outputs differ in size or sw_format ..." with a
stack trace and "preliminary init failed, will retry when we get first frame", then build from
their first frame. That is expected here. The mixer's router depends on the declared size; a run
with more than 32 --input sources of different sizes shows it.
"""
from pathlib import Path
import tempfile
import time

import numpy as np

from _harness import EOF, finish, frame_planes, make_avp, nv12_planes
from pyplumber import node as api
from pyplumber.mixer.backends.cuda import CudaMixerBackend
from pyplumber.mixer.inputs import build_raw420_input
from pyplumber.transform import transform_output, transform_params

CANVAS = (1920, 1080)
FPS, FRAMES, SOURCE_FRAMES = 5, 4, 4
HELD_FRAMES = 6   # per edge, waiting for the same timestamp on the other path
# Largest difference between the two scalers, in codes of 255, INSET pixels inside the picture.
# The test picture changes by up to 7 codes per source pixel down and 9 across, so a picture
# placed one source pixel off fails. Measured on an L4: at most 2 codes without a tone map. Behind
# a tone map, which is steep in places, one code of scaler difference becomes several: 9 to 11
# codes at the worst pixel, 2 codes at the 99th percentile, 0.4 % of the pixels over TOLERANCE and
# no shift of the mean. A tone-mapped output is therefore held to TOLERANCE at its 99th percentile
# and to OUTLIERS of its pixels above it.
TOLERANCE, INSET, OUTLIERS = 3, 4, 0.01
# The graph the mixer built for source normalisation before cuda_transform.
LEGACY_NORMALIZE = ("scale_cuda=w={0}:h={1}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                    "pad_cuda=w={0}:h={1}:x=(ow-iw)/2:y=(oh-ih)/2:color=black").format(*CANVAS)
# "uhd" is the one reduction: 2:1, where four-neighbour bilinear averages each 2x2 block.
SOURCES = {"pillarbox": (1440, 1080), "letterbox": (1920, 800), "upscale": (1280, 720),
           "small": (640, 480), "rounded": (854, 480), "uhd": (3840, 2160)}
# Also drawn onto a p010le canvas. It must be a source that needs no scaling: a scaled picture
# is interpolated at 10 bits and is then not four times its 8-bit result.
PROMOTED = "pillarbox"
RENDITION = (1280, 720)
# name: (canvas storage, source color, target storage, tone-map options)
RENDITIONS = {"sdr": ("nv12", "sdr", "nv12", {}),
              "hlg_to_sdr": ("p010le", "hlg", "nv12", {"tonemap": "mobius", "param": 0.9}),
              "hlg422_to_sdr": ("p210le", "hlg", "nv12", {"tonemap": "mobius", "param": 0.9})}


def pattern(width, height, index, storage="nv12"):
    """One raw frame of a smooth moving picture whose luma stays above black (32..224 of 255)."""
    x, y = np.arange(width)[None, :], np.arange(height)[:, None]
    cx, cy = np.arange(width // 2)[None, :], np.arange(height // 2)[:, None]
    phase = 2 * np.pi * index / SOURCE_FRAMES
    luma = 128 + 56 * np.sin(2 * np.pi * x / 40 + phase) + 40 * np.sin(2 * np.pi * y / 36)
    u = 128 + 48 * np.sin(2 * np.pi * cx / 48 + phase) + 0 * cy
    v = 128 + 48 * np.cos(2 * np.pi * cy / 40) + 0 * cx
    chroma = np.stack((u, v), axis=-1).reshape(height // 2, width)
    if storage == "nv12":
        return b"".join(np.rint(plane).astype(np.uint8).tobytes() for plane in (luma, chroma))
    # p010le: the 10-bit code in the high bits of a 16-bit word.
    return b"".join((np.rint(plane * 4).astype("<u2") << 6).tobytes() for plane in (luma, chroma))


def contain_box(width, height, canvas_w, canvas_h, align=2):
    """place() of src/mixer/primitives/compositor_geometry.hpp for an uncropped frame contained in
    the whole canvas of a 4:2:0 format: (x, y, w, h) of the picture."""
    down = lambda value: value - value % align   # noqa: E731
    rescale = lambda a, b, c: (a * b + c // 2) // c   # noqa: E731  av_rescale: nearest, halves up
    width, height = down(width), down(height)
    if width * canvas_h >= height * canvas_w:
        w, h = canvas_w, rescale(height, canvas_w, width)
    else:
        w, h = rescale(width, canvas_h, height), canvas_h
    w, h = down(w), down(h)
    return down((canvas_w - w) // 2), down((canvas_h - h) // 2), w, h


def picture_box(luma, black=16):
    """(x, y, w, h) of the rows and columns that are not black bars."""
    rows, cols = np.flatnonzero((luma != black).any(axis=1)), np.flatnonzero((luma != black).any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def differences(ours, reference, box):
    """Absolute differences of both planes, INSET pixels inside *box*, as one array."""
    x, y, w, h = box
    luma = (slice(y + INSET, y + h - INSET), slice(x + INSET, x + w - INSET))
    chroma = (slice(y // 2 + INSET // 2, (y + h) // 2 - INSET // 2), luma[1])
    return np.concatenate([np.abs(a[area].astype(np.int16) - b[area]).ravel()
                           for a, b, area in zip(ours, reference, (luma, chroma))])


def check_normalized(name, ours, legacy):
    """One frame of a source through the transform against the same frame through the filters."""
    width, height = SOURCES[name]
    x, y, w, h = box = contain_box(width, height, *CANVAS)
    inside = np.zeros((CANVAS[1], CANVAS[0]), bool)
    inside[y:y + h, x:x + w] = True
    assert ours[0].shape == inside.shape, (name, ours[0].shape)
    assert (ours[0][~inside] == 16).all() and (ours[1][~inside[::2]] == 128).all(), f"{name}: bars are not black"
    assert ours[0][inside].min() > 16 and picture_box(ours[0]) == box, (name, picture_box(ours[0]), box)
    if (w, h) == (width, height):
        assert all((a == b).all() for a, b in zip(ours, legacy)), f"{name}: differs from scale_cuda + pad_cuda"
        return "byte-identical to scale_cuda + pad_cuda"
    if picture_box(legacy[0]) != box:
        return f"picture at {box}; scale_cuda + pad_cuda puts it at {picture_box(legacy[0])}, not compared"
    difference = int(differences(ours, legacy, box).max())
    assert difference <= TOLERANCE, f"{name}: differs from scale_cuda + pad_cuda by {difference} codes"
    return f"within {difference} codes of scale_cuda + pad_cuda"


def check_promoted(promoted, ours):
    """The p010le output of a source against its nv12 output: the same picture, codes times four."""
    expected = (ours[0], ours[1][:, 0::2], ours[1][:, 1::2])
    assert all((plane == eight_bit.astype(np.uint16) * 4).all() for plane, eight_bit in zip(promoted, expected)), \
        "8-bit source on a p010le canvas is not the nv12 picture times four"
    return "p010le canvas: the nv12 picture with every code times four (bars 64/512)"


def check_rendition(name, ours, legacy):
    assert ours[0].shape == (RENDITION[1], RENDITION[0]) == legacy[0].shape, (name, ours[0].shape, legacy[0].shape)
    codes = differences(ours, legacy, (0, 0, *RENDITION))
    largest, most = int(codes.max()), int(np.percentile(codes, 99))
    if RENDITIONS[name][1] == "sdr":   # tags only: nothing amplifies the scalers' difference
        assert largest <= TOLERANCE, f"{name}: differs from scale_cuda + conversion by {largest} codes"
        return f"transform then conversion within {largest} codes of scale_cuda + conversion"
    assert most <= TOLERANCE and (codes > TOLERANCE).mean() <= OUTLIERS, \
        f"{name}: 99 % of pixels within {most} codes of scale_cuda + conversion, {(codes > TOLERANCE).mean():.2%} over {TOLERANCE}"
    return f"transform then tone map: 99 % of pixels within {most} codes of scale_cuda + conversion, largest {largest}"


def download(name, storage="nv12"):
    return api.FilterVideo({"name": f"download_{name}", "src": name, "dst": f"raw_{name}", "group": "probe",
                            "hwaccel": "gpu", "graph": f"hwdownload,format={storage}"})


def main():
    backend = CudaMixerBackend()
    with tempfile.TemporaryDirectory() as directory:
        avp, errors = make_avp("gpu", capacity=4)
        nodes = []
        # edge pairs to compare: label -> (edge of the new path, edge of the reference, check)
        pairs = {}
        try:
            def source(name, width, height, storage="nv12"):
                path = Path(directory) / f"{name}.raw"
                path.write_bytes(b"".join(pattern(width, height, i, storage) for i in range(SOURCE_FRAMES)))
                return build_raw420_input(avp, api, name, str(path), width=width, height=height, pixel_format=storage,
                                          fps=FPS, group="probe", hwaccel="gpu", loop=True, pinned=True)

            for name, (width, height) in SOURCES.items():
                nodes.append(api.Split({"name": f"split_{name}", "src": source(name, width, height),
                                        "dst": [f"{name}_to_transform", f"{name}_to_legacy"], "group": "probe"}))
                outputs = [transform_output(f"{name}_transform", *CANVAS, fit="contain")]
                if name == PROMOTED:
                    outputs.append(transform_output(f"{name}_promoted", *CANVAS, sw_format="p010le", fit="contain"))
                    nodes.append(download(f"{name}_promoted", "p010le"))
                    pairs[f"{name} promoted"] = (f"{name}_promoted", f"{name}_transform", check_promoted)
                nodes.append(api.CudaTransform(transform_params(
                    f"{name}_to_transform", outputs, hwaccel="gpu", name=f"transform_{name}", group="probe")))
                nodes.append(api.FilterVideo({"name": f"legacy_{name}", "src": f"{name}_to_legacy", "dst": f"{name}_legacy",
                                              "hwaccel": "gpu", "group": "probe", "graph": LEGACY_NORMALIZE}))
                nodes += [download(f"{name}_transform"), download(f"{name}_legacy")]
                pairs[name] = (f"{name}_transform", f"{name}_legacy",
                               lambda ours, legacy, name=name: check_normalized(name, ours, legacy))

            for name, (storage, color, target_storage, tonemap) in RENDITIONS.items():
                conversion = backend.conversion("sdr", target_storage, source=color, source_format=storage, **tonemap)
                canvas = source(name, *CANVAS, "nv12" if storage == "nv12" else "p010le")
                if storage == "p210le":
                    # The raw reader gives 4:2:0, as NVDEC does; the mixer's compositor makes 4:2:2.
                    nodes.append(api.FilterVideo({"name": f"canvas_{name}", "src": canvas, "dst": f"{name}_canvas",
                                                  "hwaccel": "gpu", "group": "probe", "threads": backend.graph_threads,
                                                  "graph": backend.scale(pixel_format=storage)}))
                    canvas = f"{name}_canvas"
                nodes.append(api.Split({"name": f"split_{name}", "src": canvas,
                                        "dst": [f"{name}_to_transform", f"{name}_to_legacy"], "group": "probe"}))
                nodes.append(api.CudaTransform(transform_params(
                    f"{name}_to_transform", [transform_output([f"{name}_sized"], *RENDITION, sw_format=storage)],
                    hwaccel="gpu", name=f"transform_{name}", group="probe")))
                # As the mixer builds it: only the conversion, initialised from the first frame.
                nodes.append(api.FilterVideo({"name": f"convert_{name}", "src": f"{name}_sized", "dst": f"{name}_transform",
                                              "hwaccel": "gpu", "group": "probe", "threads": backend.graph_threads,
                                              "graph": conversion, "defer_preliminary_init": True}))
                width, height = RENDITION
                nodes.append(api.FilterVideo({"name": f"legacy_{name}", "src": f"{name}_to_legacy", "dst": f"{name}_legacy",
                                              "hwaccel": "gpu", "group": "probe", "threads": backend.graph_threads,
                                              "graph": backend.scale(width=width, height=height) + "," + conversion}))
                nodes += [download(f"{name}_transform", target_storage), download(f"{name}_legacy", target_storage)]
                pairs[f"rendition {name}"] = (f"{name}_transform", f"{name}_legacy",
                                              lambda ours, legacy, name=name: check_rendition(name, ours, legacy))

            for node in nodes:
                avp.addNode(node)
            names = {edge for ours, reference, _ in pairs.values() for edge in (ours, reference)}
            edges = {name: avp.getEdge(f"raw_{name}", "VideoFrame") for name in names}
            avp.group("probe").startNodes()

            # The latest frames of every edge by timestamp; a pair is checked once both paths
            # delivered a timestamp. An edge can serve two pairs, so frames leave by age only.
            held = {name: {} for name in names}
            results = {label: {} for label in pairs}
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not errors and min(map(len, results.values())) < FRAMES:
                for name, edge in edges.items():
                    frame = edge.tryGet(5)
                    if frame is None or frame.pts.timestamp == EOF:
                        continue
                    held[name][frame.pts.timestamp] = (frame_planes(frame, "p010le") if name.endswith("_promoted")
                                                       else nv12_planes(frame))
                    for pts in sorted(held[name])[:-HELD_FRAMES]:
                        del held[name][pts]
                for label, (ours, reference, check) in pairs.items():
                    for pts in sorted(set(held[ours]) & set(held[reference]) - set(results[label])):
                        results[label][pts] = check(held[ours][pts], held[reference][pts])
            assert not errors, errors
            for label, outcomes in results.items():
                assert len(outcomes) >= FRAMES, f"{label}: {len(outcomes)} frames matched by timestamp, {FRAMES} needed"
                print(f"PASS {label}: {len(outcomes)} frames, {'; '.join(sorted(set(outcomes.values())))}", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    main()
