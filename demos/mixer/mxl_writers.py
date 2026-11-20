#!/usr/bin/env python3
"""Writers of the demo's MXL sources, one process group beside the mixer (setup_runtime.py starts it
before the mixer and stops it after). Each writer stands in for a producer outside the mixer, so it
uses no GPU and decodes nothing: one FFmpeg loops its source's own clip (the source's "clip" in the
show, a file of packed v210 pictures at the canvas rate that prepare_demo.plan renders once) and
copies each picture into the source's MXL video flow. The muxer numbers grains by counting packets
from the time of the first, so -re paces them from the first (FFmpeg's default burst would leave the
flow half a second ahead of the clock for good, and 0 selects that default), and -stream_loop keeps
timestamps rising over the loop's seam.

usage: python3 -m demos.mixer.mxl_writers <mixer config>"""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from pyplumber.mixer.config import parse_rate, rate_text


def main(config):
    show = json.loads(Path(config).read_text())
    rate = rate_text(parse_rate(show["canvas"].get("fps", 30)))
    writers = [subprocess.Popen(["ffmpeg", "-v", "warning", "-nostdin", "-re", "-readrate_initial_burst", "0.001",
                                 "-stream_loop", "-1", "-f", "v210", "-video_size", f"{s['width']}x{s['height']}",
                                 "-framerate", rate, "-i", s["clip"], "-c", "copy",
                                 "-f", "mxl", "-video_flow_id", s["flow"], s["url"]])
               for s in show["sources"] if s["kind"] == "mxl" and "clip" in s]
    try:
        while all(writer.poll() is None for writer in writers):
            time.sleep(1)
        for writer in writers:   # one exited: stop the rest, and the setup starts them all again
            writer.send_signal(signal.SIGINT)
    except KeyboardInterrupt:   # the setup's stop, which every writer of this process group got too
        pass
    for writer in writers:   # closing a muxer removes its flow
        writer.wait()


if __name__ == "__main__":
    main(sys.argv[1])
