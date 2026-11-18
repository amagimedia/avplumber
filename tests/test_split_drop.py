"""The split node drops for the outputs its `drop` names and waits for the others, on real edges
against the whole library."""
import subprocess


def test_split_drops_for_a_full_output_named_in_drop_and_waits_for_the_others(cpp_library_binary):
    subprocess.run([str(cpp_library_binary("test_split_drop"))], check=True, timeout=30)
