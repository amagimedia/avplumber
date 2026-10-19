"""The lever's wipe: its clip's frames, the ramps' anchor, and the take tick by tick."""
import subprocess


def test_lever_wipe(cpp_binary):
    # TickGrid takes an av::Rational, whose constructor lives in avcpp's rational.cpp.
    binary = cpp_binary("test_lever_wipe", libs=("libavutil",), sources=("deps/avcpp/src/rational.cpp",))
    subprocess.run([str(binary)], check=True, timeout=20)
