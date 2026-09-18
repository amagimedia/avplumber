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
            (REPOSITORY_DIR / "deps" / "ffmpeg" / "8").glob("*.patch")
        )

        self.assertEqual(REPO_DIR, REPOSITORY_DIR)
        self.assertGreater(len(expected_paths), 0)
        self.assertEqual(list(_patch_identity()), [path.name for path in expected_paths])

    def test_patch_identity_shares_series_between_ffmpeg80_and_81(self) -> None:
        self.assertEqual(_patch_identity(), _patch_identity("n8.1"))
        self.assertEqual(_patch_identity("n8.0"), _patch_identity("n8.1"))

    def test_patch_identity_rejects_unknown_tags(self) -> None:
        with self.assertRaises(ValueError):
            _patch_identity("n7.1.5")

    def test_cuda_images_keep_the_default_and_select_matching_patches(self) -> None:
        for relative_path in (
            "demos/cuda-overlay/Dockerfile",
            "demos/mixer/Dockerfile",
            "demos/dmabuf-browser/consumer/Dockerfile.cuda",
        ):
            with self.subTest(dockerfile=relative_path):
                dockerfile = (REPOSITORY_DIR / relative_path).read_text()
                self.assertIn("ARG FFMPEG_TAG=n8.1", dockerfile)
                self.assertIn("COPY deps/ffmpeg /build/deps/ffmpeg", dockerfile)
                self.assertIn("/build/deps/ffmpeg/apply.sh /tmp/ffmpeg", dockerfile)

    def test_overlay_image_reports_the_selected_version(self) -> None:
        dockerfile = (REPOSITORY_DIR / "demos/cuda-overlay/Dockerfile").read_text()
        self.assertIn("ENV FFMPEG_TAG=${FFMPEG_TAG}", dockerfile)

    def test_mixer_checks_runtime_options_without_removed_command_marker(self) -> None:
        dockerfile = (REPOSITORY_DIR / "demos/mixer/Dockerfile").read_text()
        self.assertIn("-h filter=transition_cuda", dockerfile)
        self.assertIn('$1 == "mode" && $3 ~ /T/', dockerfile)
        self.assertNotIn('$1 ~ /C/', dockerfile)


if __name__ == "__main__":
    unittest.main()
