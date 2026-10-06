"""Frame-size changes, crop bounds and chroma alignment without GPU dependencies."""
import subprocess


def test_compositor_geometry(cpp_binary):
    subprocess.run([str(cpp_binary("test_compositor_geometry", libs=("libavutil",)))], check=True, timeout=10)
