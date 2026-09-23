import subprocess


def test_frame_subscription(cpp_binary):
    binary = cpp_binary("test_frame_subscription", libs=("libavutil", "libavcodec", "libavformat"),
                        sources=("deps/avcpp/src/rational.cpp",), pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
