"""DMA-BUF release ACKs stay on the connection that delivered the frame, across reconnection."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="DmabufReleaseAckQueue uses eventfd")
def test_dmabuf_release(cpp_binary):
    binary = cpp_binary("test_dmabuf_release", pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
