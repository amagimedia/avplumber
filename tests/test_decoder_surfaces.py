"""The CUarray decoder's surface gate and the instance's decoder pool budget."""
import subprocess


def test_decoder_surfaces(cpp_binary):
    binary = cpp_binary("test_decoder_surfaces", libs=("libavutil",), pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
