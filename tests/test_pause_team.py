"""A pause team whose synchronizing seek fails is left as it was, and what slept on the failed pause
is woken as on a resume, against the whole library."""
import subprocess


def test_a_failed_pause_wakes_the_inputs_and_leaves_the_team_as_it_was(cpp_library_binary):
    subprocess.run([str(cpp_library_binary("test_pause_team"))], check=True, timeout=30)
