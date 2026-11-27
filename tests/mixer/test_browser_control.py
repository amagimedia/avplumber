"""mixer.browser.navigate / refresh without a browser service."""
from __future__ import annotations

import json

import pytest

from pyplumber.mixer.browser_control import BrowserController, register_browser_commands
from pyplumber.mixer.config import ConfigError


class _Avp:
    def __init__(self):
        self.handlers = {}

    def registerControlCommand(self, name, handler, _json=False):
        self.handlers[name] = handler


def test_navigate_loads_the_page_in_the_named_window():
    calls = []
    urls = {"bug": "https://example.test/", "lower": "https://example.test/lt"}
    controller = BrowserController(
        dmabuf_rest="http://browser", browser_ids=("bug", "lower"),
        navigate=lambda base, window_id, url: calls.append((base, window_id, url)),
        refresh=lambda *_: None, urls=urls)
    assert controller.navigate({"id": "bug", "url": "https://example.test/a"}) == {
        "id": "bug", "url": "https://example.test/a"}
    assert calls == [("http://browser", "bug", "https://example.test/a")]
    assert urls["bug"] == "https://example.test/a" and urls["lower"] == "https://example.test/lt"


def test_refresh_reloads_the_named_window():
    calls = []
    controller = BrowserController(
        dmabuf_rest="http://browser", browser_ids=("bug",),
        navigate=lambda *_: None,
        refresh=lambda base, ids: calls.append((base, list(ids))))
    assert controller.refresh({"id": "bug"}) == {"id": "bug"}
    assert calls == [("http://browser", ["bug"])]


def test_unknown_or_non_browser_id_is_refused():
    controller = BrowserController(dmabuf_rest="http://browser", browser_ids=("bug",),
                                   navigate=lambda *_: None, refresh=lambda *_: None)
    with pytest.raises(ConfigError, match="not a browser source"):
        controller.navigate({"id": "cam1", "url": "https://example.test/"})
    with pytest.raises(ConfigError, match="not a browser source"):
        controller.refresh({"id": "cam1"})


def test_navigate_needs_a_url_and_refresh_refuses_one():
    controller = BrowserController(dmabuf_rest="http://browser", browser_ids=("bug",),
                                   navigate=lambda *_: None, refresh=lambda *_: None)
    with pytest.raises(ConfigError, match="non-empty url"):
        controller.navigate({"id": "bug", "url": ""})
    with pytest.raises(ConfigError, match="unsupported fields"):
        controller.refresh({"id": "bug", "url": "https://example.test/"})


def test_service_failure_is_a_runtime_error():
    controller = BrowserController(
        dmabuf_rest="http://browser", browser_ids=("bug",),
        navigate=lambda *_: (_ for _ in ()).throw(RuntimeError("HTTP 500")),
        refresh=lambda *_: (_ for _ in ()).throw(TimeoutError("no answer")))
    with pytest.raises(RuntimeError, match="browser.navigate"):
        controller.navigate({"id": "bug", "url": "https://example.test/"})
    with pytest.raises(RuntimeError, match="browser.refresh"):
        controller.refresh({"id": "bug"})


def test_register_browser_commands_wires_the_control_handlers():
    avp = _Avp()
    controller = BrowserController(dmabuf_rest="http://browser", browser_ids=("bug",),
                                   navigate=lambda *_: None, refresh=lambda *_: None)
    register_browser_commands(avp, controller)
    assert set(avp.handlers) == {"mixer.browser.navigate", "mixer.browser.refresh"}
    reply = json.loads(avp.handlers["mixer.browser.navigate"](json.dumps(
        {"id": "bug", "url": "https://example.test/"})))
    assert reply == {"id": "bug", "url": "https://example.test/"}
    assert json.loads(avp.handlers["mixer.browser.refresh"](json.dumps({"id": "bug"}))) == {"id": "bug"}
