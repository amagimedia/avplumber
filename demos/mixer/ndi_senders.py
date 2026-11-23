"""Sender of the demo's NDI sources, in the writers process beside the mixer (mxl_writers.py starts
it): it stands in for cameras and playout on the network. Each source's own clip (the source's
"clip" in the show, a file of packed UYVY pictures at the canvas rate that prepare_demo.plan
renders once) is mapped from the page cache, and one thread a source hands its pictures to the NDI
SDK, which compresses them (SpeedHQ, on the CPU) and paces them at the canvas rate. Each source is
announced on this host as stream_name(its id); the full names NDI discovery lists them by (machine
and stream) are written, once every sender exists, to the file the setup reads them from."""
import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_float, c_int, c_int64, c_void_p
import json
from pathlib import Path
import threading

from pyplumber.mixer.config import parse_rate


def stream_name(source_id):
    """What the source is announced as on this host; NDI prefixes the machine's name."""
    return f"avp-{source_id}"


def names_path(config):
    """The file the full NDI names of the show at *config* are written to, by source id."""
    return Path(f"{config}.ndi-names")


# NDI SDK 6 (Processing.NDI.Lib.h), field for field.
class SendCreate(ctypes.Structure):
    _fields_ = [("p_ndi_name", c_char_p), ("p_groups", c_char_p), ("clock_video", c_bool), ("clock_audio", c_bool)]


class VideoFrame(ctypes.Structure):
    _fields_ = [("xres", c_int), ("yres", c_int), ("FourCC", c_int), ("frame_rate_N", c_int), ("frame_rate_D", c_int),
                ("picture_aspect_ratio", c_float), ("frame_format_type", c_int), ("timecode", c_int64),
                ("p_data", c_void_p), ("line_stride_in_bytes", c_int), ("p_metadata", c_char_p), ("timestamp", c_int64)]


class Source(ctypes.Structure):
    _fields_ = [("p_ndi_name", c_char_p), ("p_url_address", c_char_p)]


FOURCC_UYVY = int.from_bytes(b"UYVY", "little")
PROGRESSIVE = 1
TIMECODE_SYNTHESIZE = 2 ** 63 - 1


def load_ndi():
    ndi = ctypes.CDLL("libndi.so.6")
    for name, restype, *argtypes in (
            ("NDIlib_initialize", c_bool),
            ("NDIlib_send_create", c_void_p, POINTER(SendCreate)),
            ("NDIlib_send_send_video_v2", None, c_void_p, POINTER(VideoFrame)),
            ("NDIlib_send_get_source_name", POINTER(Source), c_void_p),
            ("NDIlib_send_destroy", None, c_void_p)):
        function = getattr(ndi, name)
        function.restype, function.argtypes = restype, argtypes
    if not ndi.NDIlib_initialize():
        raise RuntimeError("NDI cannot run on this CPU")
    return ndi


def plan(show):
    """The NDI sources of *show* this process sends: those with a clip."""
    rate = parse_rate(show["canvas"]["fps"])
    return rate, [{"id": s["id"], "clip": s["clip"], "width": s["width"], "height": s["height"]}
                  for s in show.get("sources", []) if s.get("kind") == "ndi" and "clip" in s]


def start(show, config, map_clip, stopped):
    """Send every NDI source of *show* until *stopped* is set; returns the threads, once the full
    names are written to names_path(*config*). *map_clip(path, picture bytes)* maps a clip and
    returns its address and how many pictures it holds."""
    rate, sources = plan(show)
    if not sources:
        return []
    ndi, names, threads = load_ndi(), {}, []
    for source in sources:
        stride = 2 * source["width"]   # UYVY: two bytes a pixel, rows unpadded
        clip, pictures = map_clip(source["clip"], stride * source["height"])
        sender = ndi.NDIlib_send_create(ctypes.byref(SendCreate(stream_name(source["id"]).encode(), None, True, False)))
        if not sender:
            raise RuntimeError(f"NDI source {source['id']}: cannot create its sender")
        names[source["id"]] = ndi.NDIlib_send_get_source_name(sender).contents.p_ndi_name.decode()
        frame = VideoFrame(source["width"], source["height"], FOURCC_UYVY, rate.numerator, rate.denominator,
                           source["width"] / source["height"], PROGRESSIVE, TIMECODE_SYNTHESIZE, None, stride, None, 0)

        def send(sender=sender, frame=frame, clip=clip, pictures=pictures, picture_bytes=stride * source["height"]):
            n = 0
            try:
                while not stopped.is_set():   # the SDK waits out each picture's period (clock_video)
                    frame.p_data = clip + n % pictures * picture_bytes
                    ndi.NDIlib_send_send_video_v2(sender, ctypes.byref(frame))
                    n += 1
            finally:
                ndi.NDIlib_send_destroy(sender)
        threads.append(threading.Thread(target=send, name=f"ndi-{source['id']}", daemon=True))
    path = names_path(config)
    staged = Path(f"{path}.tmp")
    staged.write_text(json.dumps(names))
    staged.replace(path)
    for thread in threads:
        thread.start()
    return threads
