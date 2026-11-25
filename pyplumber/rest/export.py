"""`python -m pyplumber.rest.export`: writes doc/api/openapi.json, the OpenAPI document of the core
and the mixer router, for client generators and reviews. tests/rest fails when it differs from
the code. A running gateway serves the document of what it mounted at /api/v1/openapi.json.

The document's shape follows the FastAPI and pydantic versions: generate it with the ones pinned
in requirements.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyplumber.mixer.rest import ClientBridge, mixer_router

from .app import create_app
from .fake import FakeEngine

PATH = Path(__file__).resolve().parents[2] / "doc" / "api" / "openapi.json"


def document() -> str:
    engine = FakeEngine()
    schema = create_app(engine, [mixer_router(ClientBridge(engine))]).openapi()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(document())
    print(PATH)
