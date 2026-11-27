import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("fastapi", reason="the REST gateway is optional: pip install -r pyplumber/rest/requirements.txt")
pytest.importorskip("httpx", reason="FastAPI's TestClient needs httpx")

from fastapi.testclient import TestClient   # noqa: E402

from pyplumber.mixer.rest import ClientBridge, mixer_router   # noqa: E402
from pyplumber.rest import create_app   # noqa: E402
from pyplumber.rest.fake import FakeEngine   # noqa: E402


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def bridge(engine):
    return ClientBridge(engine)


@pytest.fixture
def api(engine, bridge):
    """The core and the mixer router over the fake engine, at the API's prefix."""
    return TestClient(create_app(engine, [mixer_router(bridge)]), base_url="http://testserver/api/v1")
