"""A new source takes over a parked decoder, on the NVIDIA host.

Clips of one format play in turn in one instance, with parking on. Each new clip must take over
the decoder the one before left, and must look exactly as it does from a decoder of its own:

- every picture of the taken-over decode hashes the same as from a fresh decoder;
- it decodes on the first decoder's pool (no more CUarray handles appear than the pool has), so
  no decoder or pool was created, and `status` says it took over;
- frames of the old clip that a consumer still holds keep their surfaces: the new clip never
  decodes onto one of them;
- a clip of another format (--other-format) gets a decoder of its own and leaves the parked one.

    python3 tests/cuda/nvdec/decoder_takeover.py --input A --input B [--other-format C] \
        --width 1920 --height 1080 [--cycles 4]
"""
import argparse
from collections import deque
import hashlib
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import EOF, make_avp
from pyplumber import node as api

HELD_FRAMES = 3


def pixels(frame):
    digest = hashlib.sha256()
    planes, pitches = frame.data, frame.linesize
    for plane, rows in enumerate((frame.height, (frame.height + 1) // 2)):
        view = memoryview(planes[plane])
        width = frame.width if plane == 0 else ((frame.width + 1) // 2) * 2
        for row in range(rows):
            digest.update(view[row * pitches[plane]:row * pitches[plane] + width])
    return (frame.pts.timestamp, digest.hexdigest())


def source_nodes(tag, path, args, park):
    decoder = {"name": f"decode_{tag}", "src": f"video_{tag}", "dst": f"decoded_{tag}",
               "hwaccel": "takeover_gpu", "pixel_format": "cuarray",
               "options": {"threads": 1, "extra_hw_frames": args.extra_hw_frames,
                           "hwaccel_flags": "unsafe_output+extra_hw_frames_only+private_stream"}}
    if park:
        decoder.update(park=True, park_limit=4)
    return [
        api.Input({"name": f"input_{tag}", "url": str(path), "dst": f"packets_{tag}"}),
        api.Demux({"name": f"demux_{tag}", "src": f"packets_{tag}", "routing": {"v:0": f"video_{tag}"}}),
        api.DecVideo(decoder),
        api.FilterVideo({"name": f"normalize_{tag}", "src": f"decoded_{tag}", "dst": f"normalized_{tag}",
                         "hwaccel": "takeover_gpu", "threads": 1, "defer_preliminary_init": True,
                         "graph": "setparams=color_trc=bt709:color_primaries=bt709:colorspace=bt709:range=tv,"
                                  "scale_cuda=format=nv12"}),
        api.CudaRectOverlay({"name": f"compose_{tag}", "src": [f"normalized_{tag}"], "dst": f"canvas_{tag}",
                             "hwaccel": "takeover_gpu", "width": args.width, "height": args.height,
                             "sw_format": "nv12",
                             "layers": [{"input": 0, "dst_x": 0, "dst_y": 0, "dst_w": args.width, "dst_h": args.height}]}),
        api.FilterVideo({"name": f"verify_{tag}", "src": f"canvas_{tag}", "dst": f"result_{tag}",
                         "hwaccel": "takeover_gpu", "graph": "hwdownload,format=nv12", "threads": 1,
                         "defer_preliminary_init": True}),
    ]


class Play:
    """One clip through its own group: the hashes of its pictures, the surfaces it decoded onto,
    and the last frames it decoded, held like a late consumer's until told otherwise."""

    def __init__(self, avp, errors, tag, path, args, park):
        self.tag, self.avp, self.errors = tag, avp, errors
        self.surfaces, self.held, self.hashes = set(), deque(maxlen=HELD_FRAMES), []
        self.nodes = source_nodes(tag, path, args, park)
        for node in self.nodes:
            node.parameters.update(group=tag, auto_restart="off")
            avp.addNode(node)
        avp.getEdge(f"decoded_{tag}", "VideoFrame").addWiretapCallback(self.decoded)
        self.output = avp.getEdge(f"result_{tag}", "VideoFrame")
        avp.group(tag).startNodes()

    def decoded(self, frame):
        if frame.pts.timestamp != EOF:
            self.surfaces.add(frame.data_ptr[0])
            self.held.append(frame)

    def run(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.errors:
            frame = self.output.tryGet(10)
            if frame is None:
                continue
            if frame.pts.timestamp == EOF:
                return self
            self.hashes.append(pixels(frame))
            if len(self.hashes) == 1:   # the node is up, and has its pool
                self.status = self.avp.node(f"decode_{self.tag}").getObject("status")
        raise AssertionError(f"{self.tag}: no end of stream: {self.errors}")

    def close(self):
        self.avp.executeCommandsFromString(f"group.stop {self.tag}")
        self.nodes.clear()


def shutdown(avp, timeout=20):
    worker = threading.Thread(target=avp.shutdown, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "shutdown did not release the decoders"


def reference(path, args):
    """The clip from a decoder of its own, in an instance of its own."""
    avp, errors = make_avp("takeover_gpu", capacity=3)
    play = Play(avp, errors, "fresh", path, args, park=False).run(args.timeout)
    assert not play.status["surfaces"]["took_over"], play.status
    hashes = play.hashes
    play.held.clear()
    play.close()
    shutdown(avp)
    assert hashes, path
    return hashes


def run(args):
    fresh = {path: reference(path, args) for path in args.input}
    avp, errors = make_avp("takeover_gpu", capacity=3)
    pool_surfaces, order = set(), [args.input[n % len(args.input)] for n in range(args.cycles)]
    previous = None
    for n, path in enumerate(order):
        play = Play(avp, errors, f"cycle{n}", path, args, park=True).run(args.timeout)
        assert play.hashes == fresh[path], f"cycle {n}: {path} does not match its fresh decode"
        took_over = play.status["surfaces"]["took_over"]
        pool_surfaces |= play.surfaces
        assert len(pool_surfaces) <= play.status["surfaces"]["pool"], "surfaces beyond the first decoder's pool"
        if n == 0:
            assert not took_over, "nothing was parked yet"
        else:
            assert took_over, f"cycle {n} created a decoder"
            kept = {frame.data_ptr[0] for frame in previous.held}
            assert len(kept) == HELD_FRAMES
            assert not kept & play.surfaces, "decoded onto a surface a consumer still holds"
        print(f"cycle {n} {Path(path).name}: {len(play.hashes)} pictures match, "
              f"{'took over' if took_over else 'created'} a decoder, {len(play.surfaces)} surfaces", flush=True)
        if previous:
            previous.held.clear()
        play.close()
        previous = play
    if previous:
        previous.held.clear()
    if args.other_format:
        other = Play(avp, errors, "other", args.other_format, args, park=True).run(args.timeout)
        assert not other.status["surfaces"]["took_over"], other.status
        assert not other.surfaces & pool_surfaces, "another format decoded on the parked pool"
        print(f"other format: {len(other.hashes)} pictures from a decoder of its own", flush=True)
        other.held.clear()
        other.close()
    assert not errors, errors
    shutdown(avp)
    print("PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="clip of the shared format; give two or more")
    parser.add_argument("--other-format", help="a clip that cannot use the same decoder")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--extra-hw-frames", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    run(parser.parse_args())
