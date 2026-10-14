"""Work handed to an event loop from many threads all runs, and a loop pass costs the same however many
threads have handed it work."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="EventLoop uses eventfd and epoll")
def test_event_loop_todo(cpp_binary):
    binary = cpp_binary("test_event_loop_todo", libs=("libavutil", "libavcodec", "libavformat"),
                        sources=("src/util.cpp", "deps/avcpp/src/rational.cpp", "deps/avcpp/src/timestamp.cpp"),
                        pthread=True)
    subprocess.run([str(binary)], check=True, timeout=120)
