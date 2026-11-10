"""A move take's plan: the union scene and the routed fallback."""
import subprocess


def test_move_plan(cpp_binary):
    binary = cpp_binary("test_move_plan", libs=("libavutil",),
                        sources=("src/util.cpp", "deps/avcpp/src/rational.cpp", "deps/avcpp/src/timestamp.cpp",
                                 "src/mixer/transition_control.cpp", "src/mixer/backends/cuda/transition_control.cpp"),
                        pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
