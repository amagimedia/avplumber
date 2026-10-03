import importlib.util
from pathlib import Path
import threading

import pytest


spec = importlib.util.spec_from_file_location("preview_server", Path(__file__).parents[1] / "preview-server.py")
preview_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview_server)


@pytest.fixture
def server():
    """preview-server.py as a module."""
    return preview_server


@pytest.fixture
def http(server):
    """The preview server, listening on a free loopback port."""
    http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.PreviewHandler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    yield http
    http.shutdown()
    http.server_close()
    worker.join(timeout=2)
