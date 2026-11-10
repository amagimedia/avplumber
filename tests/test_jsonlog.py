"""The vendored OTel JSON log line format (``pyplumber.jsonlog``).

Same contract as eka-recorder-ai's ``tests/manager/test_jsonlog.py``, including the
default: JSON on stdout unless ``RECORDER_LOG_FORMAT=text`` selects the legacy line.
"""

from __future__ import annotations

import io
import json
import logging
import re

import pytest

from pyplumber.jsonlog import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    formatter_from_env,
)

_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


def _record(msg: str, *args: object, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord("pyplumber.test", level, "p", 1, msg, args, None)


def test_json_is_the_default_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RECORDER_LOG_FORMAT", raising=False)
    assert isinstance(formatter_from_env(), JsonFormatter)


def test_text_format_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECORDER_LOG_FORMAT", "text")
    assert isinstance(formatter_from_env(), TextFormatter)


def test_json_line_carries_the_standard_fields() -> None:
    line = JsonFormatter().format(_record("event=graph_built nodes=%s", "12"))
    fields = json.loads(line)
    assert _TIMESTAMP.fullmatch(fields["timestamp"])
    assert fields["body"] == "graph_built"
    assert fields["nodes"] == "12"
    assert fields["severity_text"] == "INFO"
    assert fields["severity_number"] == 9
    assert fields["service.name"] == "avplumber"
    assert fields["logger"] == "pyplumber.test"


def test_text_line_keeps_the_plain_shape() -> None:
    line = TextFormatter().format(_record("plain message %s", "x"))
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z INFO  \[pyplumber.test\] plain message x",
        line,
    )


def test_configure_logging_selects_the_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECORDER_LOG_FORMAT", "json")
    stream = io.StringIO()
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    try:
        configure_logging(stream=stream)
        logging.getLogger("pyplumber.jsonlog_test").info("event=x")
        fields = json.loads(stream.getvalue().splitlines()[-1])
        assert fields["body"] == "x"
    finally:
        root.handlers[:] = old_handlers