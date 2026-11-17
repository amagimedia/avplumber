"""The program selector after a lever take ends into an armed wipe, and a take for a show tick
landing on that tick whichever slot's frame of it comes first, against the whole library."""
import subprocess


def test_armed_wipe_puts_the_program_slot_out(cpp_library_binary):
    subprocess.run([str(cpp_library_binary("test_mixer_select_program"))], check=True, timeout=20)
