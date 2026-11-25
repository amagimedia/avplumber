"""Which requests a browser sent on behalf of another site's page. Standard library only: the
mixer's control page (`pyplumber.mixer.gui.web`) uses it without the REST gateway's dependencies."""

from urllib.parse import urlsplit


def cross_origin(headers) -> bool:
    """A browser request sent by another origin's page, which must not drive this instance.
    `headers` is any mapping with a case-insensitive `get`. Clients other than browsers send no
    Origin; who may reach the port is the deployment's choice."""
    origin = headers.get("Origin")
    return bool(origin and urlsplit(origin).netloc != headers.get("Host")) or \
        headers.get("Sec-Fetch-Site") == "cross-site"
