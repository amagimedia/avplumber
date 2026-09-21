"""Source masks must not truncate or alias compositor pads above index 31."""
import subprocess


def test_mixer_source_mask(cpp_binary):
    binary = cpp_binary("test_mixer_source_mask", libs=("libavutil",),
                        sources=("src/util.cpp", "deps/avcpp/src/rational.cpp", "deps/avcpp/src/timestamp.cpp"), pthread=True)
    subprocess.run([str(binary)], check=True)
