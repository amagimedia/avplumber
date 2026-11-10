"""The program selector's move branch: on air at the destination's first ready frame, direct after."""
import subprocess


def test_move_goes_on_air_at_the_destination(cpp_library_binary):
    subprocess.run([str(cpp_library_binary("test_mixer_select_move"))], check=True, timeout=20)
