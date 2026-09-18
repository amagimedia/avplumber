"""Exact-picture retention and stale-frame rejection, without GPU dependencies."""
import subprocess


def test_mixer_snapshot(cpp_binary):
    subprocess.run([str(cpp_binary("test_mixer_snapshot"))], check=True, timeout=10)
