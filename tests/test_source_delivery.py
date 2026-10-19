"""Displayed source identities exclude prewarm, aliases and rewritten hold timestamps."""
import subprocess
import shutil
import pytest


def test_source_delivery(cpp_binary):
    if not shutil.which("pkg-config"):
        pytest.skip("libavutil development tools unavailable")
    binary = cpp_binary("test_source_delivery", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=30)
