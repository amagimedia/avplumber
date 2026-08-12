#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOOLS_DIR))

from run_matrix import REPO_DIR, _patch_identity  # noqa: E402


class RunMatrixTest(unittest.TestCase):
    def test_patch_identity_reads_the_repository_patch_series(self) -> None:
        expected_paths = sorted(
            (REPOSITORY_DIR / "deps" / "ffmpeg-patches").glob("*.patch")
        )

        self.assertEqual(REPO_DIR, REPOSITORY_DIR)
        self.assertEqual(len(expected_paths), 7)
        self.assertEqual(list(_patch_identity()), [path.name for path in expected_paths])


if __name__ == "__main__":
    unittest.main()
