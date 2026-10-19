"""The lever arm's direction and pickup, and the take clock the keys follow."""
import subprocess


def test_lever(cpp_binary):
    subprocess.run([str(cpp_binary("test_lever"))], check=True, timeout=20)
