"""HDR10 static metadata serialised onto an encoder context; needs libavcodec headers (no GPU)."""
import subprocess


def test_hdr_metadata(cpp_binary):
    subprocess.run([str(cpp_binary("test_hdr_metadata", libs=("libavcodec", "libavutil")))], check=True, timeout=10)
