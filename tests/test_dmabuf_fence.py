"""The acquire fence of a DMA-BUF frame: the wait on its fd and the receive of its message."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="eventfd and SCM_RIGHTS counting")
def test_dmabuf_fence(cpp_binary):
    binary = cpp_binary("test_dmabuf_fence", pthread=True)
    subprocess.run([str(binary)], check=True, timeout=20)
