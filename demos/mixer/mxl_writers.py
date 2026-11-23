#!/usr/bin/env python3
"""Writer of the demo's MXL sources and sender of its NDI ones (ndi_senders.py), one process beside
the mixer (setup_runtime.py starts it before the mixer and stops it after). It stands in for the producers outside the mixer, so it uses no GPU
and decodes nothing: each source's own clip (the source's "clip" in the show, a file of packed v210
pictures at the canvas rate that prepare_demo.plan renders once) is mapped from the page cache, and
at every grain of the MXL clock one thread copies each flow's next picture straight into its ring
through libmxl: one copy a picture, where an FFmpeg per flow also read() each into a packet. A
late grain is written as soon as it can be; a writer further behind than the smallest ring holds
skips to the clock, leaving a hole. The flows are removed at SIGINT or SIGTERM, and a flow that
cannot be written ends the process with an error (the setup starts it again).

usage: python3 -m demos.mixer.mxl_writers <mixer config>"""
import ctypes
from ctypes import POINTER, byref, c_bool, c_char_p, c_int, c_size_t, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p
import ctypes.util
import json
import mmap
import os
from pathlib import Path
import signal
import sys
import threading
import time

from demos.mixer import ndi_senders
from pyplumber.mixer.config import parse_rate
from pyplumber.mixer.inputs import v210_row_stride


# The structures of libmxl 1.1.0 (mxl/rational.h, flow.h, flowinfo.h), field for field.
class Rational(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_int64), ("denominator", ctypes.c_int64)]


class GrainInfo(ctypes.Structure):
    _fields_ = [("version", c_uint32), ("size", c_uint32), ("index", c_uint64), ("flags", c_uint32), ("grainSize", c_uint32),
                ("totalSlices", c_uint16), ("validSlices", c_uint16), ("reserved", c_uint8 * 4068)]


class CommonFlowConfigInfo(ctypes.Structure):
    _fields_ = [("id", c_uint8 * 16), ("format", c_uint32), ("flags", c_uint32), ("grainRate", Rational),
                ("maxCommitBatchSizeHint", c_uint32), ("maxSyncBatchSizeHint", c_uint32), ("payloadLocation", c_uint32),
                ("deviceIndex", ctypes.c_int32), ("reserved", c_uint8 * 72)]


class DiscreteFlowConfigInfo(ctypes.Structure):
    _fields_ = [("sliceSizes", c_uint32 * 4), ("grainCount", c_uint32), ("reserved", c_uint8 * 44)]


class FlowConfigInfo(ctypes.Structure):
    # The header's union of the discrete and the continuous (audio) half, both 64 bytes.
    _fields_ = [("common", CommonFlowConfigInfo), ("discrete", DiscreteFlowConfigInfo)]


def plan(show):
    """The canvas rate and, for each MXL source with a clip, its domain, flow definition, clip, row
    stride and grain size in bytes. The definition is the `mxl` muxer's for v210, which the mixer's
    reader checks: progressive video/v210 of the source's size, a slice a row of v210_row_stride()."""
    rate = parse_rate(show["canvas"].get("fps", 30))
    flows = []
    for s in show["sources"]:
        if s["kind"] != "mxl" or "clip" not in s:
            continue
        width, height, stride = s["width"], s["height"], v210_row_stride(s["width"])
        flows.append({"domain": s["url"], "clip": s["clip"], "stride": stride, "grain_bytes": stride * height, "definition": {
            "description": "avplumber demo source", "id": s["flow"], "label": s["id"], "parents": [],
            "tags": {"urn:x-nmos:tag:grouphint/v1.0": [f"{s['id']}:Video"]},
            "format": "urn:x-nmos:format:video", "media_type": "video/v210",
            "grain_rate": {"numerator": rate.numerator, "denominator": rate.denominator},
            "frame_width": width, "frame_height": height, "interlace_mode": "progressive", "colorspace": "BT709",
            "components": [{"name": name, "width": w, "height": height, "bit_depth": 10}
                           for name, w in (("Y", width), ("Cb", width // 2), ("Cr", width // 2))]}})
    return rate, flows


def load_mxl():
    """libmxl with the signatures of the calls used (handles and indices are 64 bits wide)."""
    mxl = ctypes.CDLL(ctypes.util.find_library("mxl") or "libmxl.so.1")   # the loader's path, as avplumber finds it
    for name, restype, *argtypes in (
            ("mxlCreateInstance", c_void_p, c_char_p, c_char_p),
            ("mxlDestroyInstance", c_int, c_void_p),
            ("mxlCreateFlowWriter", c_int, c_void_p, c_char_p, c_char_p, POINTER(c_void_p), POINTER(FlowConfigInfo), POINTER(c_bool)),
            ("mxlReleaseFlowWriter", c_int, c_void_p, c_void_p),
            ("mxlFlowWriterOpenGrain", c_int, c_void_p, c_uint64, POINTER(GrainInfo), POINTER(c_void_p)),
            ("mxlFlowWriterCommitGrain", c_int, c_void_p, POINTER(GrainInfo)),
            ("mxlGetCurrentIndex", c_uint64, POINTER(Rational)),
            ("mxlGetNsUntilIndex", c_uint64, c_uint64, POINTER(Rational)),
            ("mxlSleepForNs", None, c_uint64)):
        function = getattr(mxl, name)
        function.restype, function.argtypes = restype, argtypes
    return mxl


def map_clip(path, grain_bytes):
    """The address and picture count of the loop at *path*, mapped read-only and shared with the
    page cache. Through libc: Python's mmap gives no address for a read-only mapping. The whole
    loop is read in before this returns (MAP_POPULATE, Linux only): a page read from the disk in
    the copy would stall every flow."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype, libc.mmap.argtypes = c_void_p, (c_void_p, c_size_t, c_int, c_int, c_int, ctypes.c_long)
    with open(path, "rb") as clip:
        size = os.fstat(clip.fileno()).st_size
        if size < grain_bytes:
            raise OSError(f"{path} holds no whole picture of {grain_bytes} bytes")
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0), clip.fileno(), 0)
    if address == c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), f"cannot map {path}")
    return address, size // grain_bytes


def main(config):
    show = json.loads(Path(config).read_text())
    rate, flows = plan(show)
    stop, stopped = [], threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):   # a flag, so that no grain is left open
        signal.signal(sig, lambda *_: (stop.append(True), stopped.set()))
    senders = []
    try:
        senders = ndi_senders.start(show, config, map_clip, stopped)
        if flows:
            write_flows(flows, rate, stop)
        while senders and not stop:   # NDI alone: its threads send until a signal
            time.sleep(.2)
    finally:
        stopped.set()
        for thread in senders:
            thread.join()


def write_flows(flows, rate, stop):
    """Write every flow of *flows* at the MXL clock of *rate* until *stop* holds anything."""
    mxl, rate = load_mxl(), Rational(rate.numerator, rate.denominator)

    def check(ok, what, flow):
        if not ok:
            raise RuntimeError(f"MXL flow {flow['definition']['id']}: {what}")

    instances, writers, skipped, catch_up = {}, [], 0, sys.maxsize
    try:
        for flow in flows:
            domain = flow["domain"]
            if domain not in instances:
                instances[domain] = mxl.mxlCreateInstance(domain.encode(), None)
            check(instances[domain], f"cannot open the domain {domain}", flow)
            clip, frames = map_clip(flow["clip"], flow["grain_bytes"])
            writer, config_info, created = c_void_p(), FlowConfigInfo(), c_bool()
            status = mxl.mxlCreateFlowWriter(instances[domain], json.dumps(flow["definition"]).encode(), None,
                                             byref(writer), byref(config_info), byref(created))
            check(status == 0, f"cannot create it (status {status})", flow)
            writers.append((instances[domain], writer, flow, clip, frames))
            # What the mixer's reader requires too: MXL_DATA_FORMAT_VIDEO, MXL_PAYLOAD_LOCATION_HOST_MEMORY.
            check(config_info.common.format == 1 and config_info.common.payloadLocation == 0 and
                  config_info.discrete.sliceSizes[0] == flow["stride"], "it is not video in host memory with v210 rows as slices", flow)
            catch_up = min(catch_up, max(0, config_info.discrete.grainCount - 2))
        info, payload = GrainInfo(), c_void_p()
        index = mxl.mxlGetCurrentIndex(rate)
        while writers and not stop:
            mxl.mxlSleepForNs(mxl.mxlGetNsUntilIndex(index, rate))
            for _, writer, flow, clip, frames in writers:
                status = mxl.mxlFlowWriterOpenGrain(writer, index, byref(info), byref(payload))
                check(status == 0 and info.grainSize == flow["grain_bytes"], f"cannot open grain {index} (status {status})", flow)
                ctypes.memmove(payload, clip + index % frames * flow["grain_bytes"], flow["grain_bytes"])
                info.validSlices = info.totalSlices
                status = mxl.mxlFlowWriterCommitGrain(writer, byref(info))
                check(status == 0, f"cannot commit grain {index} (status {status})", flow)
            # A late period is made up for at once, but only as far behind the clock as every
            # flow's ring keeps grains readable (grainCount - 2 behind its head); older ones would
            # be a burst that no reader shows. Further behind, the clock's index follows and the
            # flows have a hole, which the mixer's reader passes over.
            index += 1
            if (behind := mxl.mxlGetCurrentIndex(rate) - index) > catch_up:
                index, skipped = index + behind, skipped + behind
    finally:   # releasing a writer removes its flow
        for instance, writer, *_ in writers:
            mxl.mxlReleaseFlowWriter(instance, writer)
        for instance in instances.values():
            if instance:
                mxl.mxlDestroyInstance(instance)
        print(f"MXL writers: {len(writers)} flows closed, {skipped} grain periods skipped", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
