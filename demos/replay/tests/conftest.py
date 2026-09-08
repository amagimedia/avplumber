import sys
from pathlib import Path

import pytest


REPLAY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_DIR))


def pytest_addoption(parser):
    parser.addoption("--replay-backend", choices=("nvidia", "cpu"), default="nvidia",
                     help="Codec backend for native playback and transcode integration tests")


@pytest.fixture(scope="module")
def playback_backend(request):
    from playback_backend import require_backend

    backend = request.config.getoption("--replay-backend")
    require_backend(backend)
    return backend
