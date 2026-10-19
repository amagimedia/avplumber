"""The receive of a dma-browser frame message: its header and its DMA-BUF fd."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="eventfd and SCM_RIGHTS counting")
def test_texinfo_message(cpp_binary):
    binary = cpp_binary("test_texinfo_message")
    subprocess.run([str(binary)], check=True, timeout=20)
