"""MultiEventWait reports and drains every signalled event, whatever the size of its list."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="MultiEventWait uses eventfd and epoll")
def test_multi_event_wait(cpp_binary):
    binary = cpp_binary("test_multi_event_wait", pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
