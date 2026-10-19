"""A queue's event writes its descriptor once per read, not once per signal, and loses no wake-up."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Event uses eventfd")
def test_event_coalesce(cpp_binary):
    binary = cpp_binary("test_event_coalesce", libs=("libavutil", "libavcodec", "libavformat"),
                        sources=("src/util.cpp", "deps/avcpp/src/rational.cpp", "deps/avcpp/src/timestamp.cpp"),
                        pthread=True)
    subprocess.run([str(binary)], check=True, timeout=60)
