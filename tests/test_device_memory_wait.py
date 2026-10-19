"""A node short of device memory tries again for a bounded time instead of failing at the first refusal."""
import subprocess


def test_device_memory_wait(cpp_binary):
    subprocess.run([str(cpp_binary("test_device_memory_wait"))], check=True, timeout=10)
