"""Cut identity and elapsed-clock accounting, without media/GPU dependencies."""
import subprocess


def test_cut_latency(cpp_binary):
    subprocess.run([str(cpp_binary("test_cut_latency", pthread=True))], check=True, timeout=10)
