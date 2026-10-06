import subprocess


def test_deferred_release(cpp_binary):
    binary = cpp_binary("test_deferred_release", pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
