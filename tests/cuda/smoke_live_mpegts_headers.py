"""Probe a nonseekable H.264/AAC transport stream starting between keyframes.

Run with the native pyplumber module and FFmpeg with h264_nvenc. The writer opens
the FIFO once. The input must retain that connection and deliver the next IDR
to NVDEC after exhausting at least one stream-info probe.
"""
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from _harness import finish, make_avp
from pyplumber import node as api


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clip = root / "source.ts"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
            "-f", "lavfi", "-i", "sine=sample_rate=48000", "-t", "8", "-c:v", "h264_nvenc",
            "-g", "100", "-bf", "0", "-b:v", "3M", "-c:a", "aac", "-muxrate", "6M", str(clip),
        ], check=True)
        fifo = root / "live.ts"
        os.mkfifo(fifo)

        def publish():
            with clip.open("rb") as source, fifo.open("wb") as transport:
                source.seek(188 * 4000)  # About one second into the four-second GOP.
                try:
                    shutil.copyfileobj(source, transport)
                except BrokenPipeError:
                    pass  # The first decoded picture completes the smoke.

        writer = threading.Thread(target=publish, daemon=True)
        writer.start()
        avp, errors = make_avp("gpu")
        source = api.InputRec({
            "name": "input", "url": str(fifo), "dst": "packets", "group": "live",
            "initial_timeout": 8, "wait_for_video_headers": True,
            "options": {"probesize": "131072", "analyzeduration": "500000"},
        })
        nodes = [source, api.Demux({
            "name": "demux", "src": "packets", "routing": {"v:0": "video"}, "group": "live",
        }), api.DecVideo({
            "name": "decode", "src": "video", "dst": "pictures", "group": "live",
            "hwaccel": "gpu", "codec": "h264_cuvid", "pixel_format": "?cuda",
        })]
        try:
            for node in nodes:
                avp.addNode(node, early_create=True)
            out = avp.getEdge("pictures", "VideoFrame")
            avp.group("live").startNodes()
            frame = out.tryGet(5000)
            assert frame is not None and (frame.width, frame.height) == (640, 360), errors
            assert not errors, errors
        finally:
            finish(avp, nodes)
        writer.join(2)
        assert not writer.is_alive()
        print("PASS decoded the next keyframe after joining mid-GOP")


if __name__ == "__main__":
    main()
