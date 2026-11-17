"""The keyer's per-key frame queue: a late paint supersedes the held copies numbered past it."""
import subprocess


def test_key_queue(cpp_binary):
    # KeyQueue is a header-only template over the standard library.
    subprocess.run([str(cpp_binary("test_key_queue"))], check=True, timeout=10)
