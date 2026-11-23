import subprocess


def test_fabric_wire_compatibility(cpp_binary):
    subprocess.run([cpp_binary("test_fabric_wire")], check=True)
