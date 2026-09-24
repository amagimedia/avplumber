"""Fade backend commands must retain the existing CUDA alpha expression."""
import subprocess


def test_transition_control(cpp_binary):
    binary = cpp_binary("test_transition_control", sources=(
        "src/mixer/transition_control.cpp", "src/mixer/backends/cuda/transition_control.cpp"))
    subprocess.run([str(binary)], check=True, timeout=10)
