"""Frame-size changes, crop bounds and chroma alignment without GPU dependencies."""
import pathlib
import shutil
import subprocess


def test_compositor_geometry(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = tmp_path / 'compositor_geometry'
    compiler = shutil.which('g++') or shutil.which('clang++')
    assert compiler, 'a C++ compiler is required'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-I', str(root / 'src'),
                    str(root / 'tests/cpp/test_compositor_geometry.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
