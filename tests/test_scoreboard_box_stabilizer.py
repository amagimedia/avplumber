from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CPP = REPO_ROOT / "tests/cpp/test_scoreboard_box_stabilizer.cpp"


def test_scoreboard_box_stabilizer(tmp_path: Path) -> None:
    binary = tmp_path / "test_scoreboard_box_stabilizer"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{REPO_ROOT / 'src'}",
            str(TEST_CPP),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: all scoreboard stabilizer tests passed" in result.stdout
