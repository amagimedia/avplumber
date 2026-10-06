import importlib
from types import SimpleNamespace
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def nvenc():
    """Skips unless FFmpeg encodes H.264 and HEVC on NVENC here: test media is never encoded in
    software. A probe encode, because an FFmpeg build lists NVENC encoders without a GPU too."""
    if not all(shutil.which(tool) for tool in ("nvidia-smi", "ffmpeg", "ffprobe")):
        pytest.skip("requires an NVIDIA host with FFmpeg and ffprobe")
    for codec in ("h264_nvenc", "hevc_nvenc"):
        probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=256x144:rate=4:duration=0.25",
                                "-pix_fmt", "yuv420p", "-c:v", codec, "-f", "null", "-"], capture_output=True)
        if probe.returncode:
            pytest.skip(f"requires an NVIDIA GPU with {codec}")


@pytest.fixture
def native_boundary(monkeypatch):
    before = set(sys.modules)
    monkeypatch.setitem(sys.modules, "_avplumber", SimpleNamespace(AVPlumber=object))
    nodes = importlib.import_module("pyplumber.node")
    builder = importlib.import_module("pyplumber.mixer").MixerGraphBuilder
    yield nodes, builder
    for name in set(sys.modules) - before:
        if name == "pyplumber.mixer" or name.startswith(("pyplumber.mixer.", "pyplumber")):
            sys.modules.pop(name, None)



@pytest.fixture
def http_server():
    from contextlib import ExitStack
    import threading
    from pyplumber.mixer.gui import serve
    with ExitStack() as cleanup:
        def start(bridge, **kwargs):
            server = serve(bridge, "127.0.0.1", 0, **kwargs)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            cleanup.callback(thread.join, 3)
            cleanup.callback(server.server_close)
            cleanup.callback(server.shutdown)
            return server
        yield start
