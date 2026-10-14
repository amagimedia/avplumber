"""The event loop passes once per event: a wait, work or a timer set from inside the loop needs no wake-up."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="EventLoop uses eventfd and poll")
def test_event_loop(cpp_binary):
    binary = cpp_binary("test_event_loop", libs=("libavutil", "libavcodec", "libavformat"),
                        sources=("src/util.cpp", "src/avutils.cpp", "deps/avcpp/src/rational.cpp",
                                 "deps/avcpp/src/timestamp.cpp"), pthread=True)
    subprocess.run([str(binary)], check=True, timeout=30)
