"""Compatibility wrapper for :mod:`pyplumber.auto_mixer.cli`."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from pyplumber.auto_mixer.cli import main


if __name__ == "__main__":
    main()
