"""The OTel JSON log line: one JSON object per stdout line, shared by every process.

Implements the log format standard ("Log Format Standard for OpenTelemetry with JSON
Structure and Requirements"): one JSON object per line, UTF-8, no pretty-printing, no
line longer than 16 KB.  Field names follow the OpenTelemetry log data model so the
same line is parsed identically by Loki's ``| json``, by an OTel Collector
``json_parser`` and by Kloudfuse.

Vendored from eka-recorder-ai's ``ai-recorder/recorder/jsonlog.py`` (the avplumber
and avplumber-nodes repos do not depend on the recorder package). Keep the two in
sync: the field contract is the "Log Format Standard for OpenTelemetry with JSON
Structure and Requirements". The defaults differ on purpose: these repos log plain
text unless ``RECORDER_LOG_FORMAT=json``.


The existing ``event=... key=value`` message convention is kept at the call sites:
:meth:`JsonFormatter.format` splits the *un-interpolated* message on whitespace, so
placeholders (``%s``, ``%(name)s``) are substituted from ``record.args`` without ever
splitting a value that contains spaces.  ``event=x`` becomes the static ``body`` (few
distinct values per service, which is what makes ``count by (body)`` useful) and every
other ``key=value`` token becomes a flat attribute, matching the standard's
``attributes.*`` rule ("flat keys, snake_case or OTel dotted names").

Everything is computed from the record and the environment at format time; the module
imports nothing beyond the standard library so unit tests and the graph child can use
it without the engine.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

__all__ = [
    "MAX_ATTRIBUTE_BYTES",
    "MAX_LINE_BYTES",
    "JsonFormatter",
    "TextFormatter",
    "configure_logging",
    "formatter_from_env",
    "service_fields",
    "severity_of",
    "wrap_line",
]

#: One JSON line must stay under 16 KB (the standard's rule 7; Loki rejects more).
MAX_LINE_BYTES = 16 * 1024
#: Any single attribute is capped at 4 KB (the standard's rule 7).
MAX_ATTRIBUTE_BYTES = 4 * 1024

#: Python level -> standard severity text; ``WARNING`` maps to ``WARN`` and
#: ``CRITICAL`` to ``FATAL`` (Python has no ``TRACE`` level; the engine's lines that
#: do are wrapped by :func:`wrap_line`).
SEVERITY_TEXT = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}

#: The OTel severity numbers pairing with :data:`SEVERITY_TEXT` (TRACE 1, DEBUG 5,
#: INFO 9, WARN 13, ERROR 17, FATAL 21) so Grafana can sort and colour without string
#: matching.
SEVERITY_NUMBER = {
    logging.DEBUG: 5,
    logging.INFO: 9,
    logging.WARNING: 13,
    logging.ERROR: 17,
    logging.CRITICAL: 21,
}

#: A ``%`` conversion in a message token: ``%s``, ``%d``, ``%.3f``, ``%(name)s`` ...
_PLACEHOLDER = re.compile(r"%(?:\(\w+\))?[-#0 ]?\d*(?:\.\d+)?[sdiouxXeEfFgGcr]")

#: Fields the standard owns.  A parsed ``key=value`` token that would shadow one is
#: renamed under ``attr.`` instead of corrupting the line's meaning.
_RESERVED_FIELDS = frozenset(
    {
        "timestamp",
        "severity_text",
        "severity_number",
        "body",
        "service.name",
        "service.version",
        "deployment.environment",
        "trace_id",
        "span_id",
        "logger",
        "thread",
        "exception.type",
        "exception.message",
        "exception.stacktrace",
    }
)

#: The pairs of :data:`SEVERITY_TEXT` indexed by text, for foreign-line wrapping.
SEVERITY_BY_TEXT = {
    SEVERITY_TEXT[level]: number for level, number in SEVERITY_NUMBER.items()
}

_SERVICE_NAME = "service.name"
_SERVICE_VERSION = "service.version"
_DEPLOYMENT_ENVIRONMENT = "deployment.environment"

#: ``RECORDER_LOG_FORMAT``: ``text`` (the default in these repos, the legacy
#: human-readable line) or ``json`` (the standard's wire format).
_FORMAT_VAR = "RECORDER_LOG_FORMAT"


def severity_of(level: int) -> tuple[str, int]:
    """The ``(severity_text, severity_number)`` pair for a Python logging level."""
    level = min(level, logging.CRITICAL)
    return (
        SEVERITY_TEXT.get(level, "ERROR"),
        SEVERITY_NUMBER.get(level, 17),
    )


def _ambient_span_ids() -> tuple[str, str] | tuple[None, None]:
    """The active span's ids, lowercase hex, or ``(None, None)`` without one.

    Read at format time rather than through a ``logging.Filter``: root-logger filters
    never see records propagated from child loggers, while the formatter runs for
    every handler (the stdout one and the uploaded manager log alike). The lazy
    import keeps this module importable without the OpenTelemetry packages.
    """
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - the sdk ships with the package
        return None, None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None, None
    return format(context.trace_id, "032x"), format(context.span_id, "016x")


def _utc_stamp(seconds: float) -> str:
    """RFC 3339 UTC with milliseconds, never local time (``2026-10-28T08:22:25.123Z``)."""
    moment = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _cap(text: str, limit: int) -> str:
    """UTF-8 byte cap for one string field (the standard's 4 KB attribute rule)."""
    if len(text.encode("utf-8", errors="replace")) <= limit:
        return text
    while text and len(text.encode("utf-8", errors="replace")) > limit:
        text = text[: max(len(text) * 3 // 4, 1)]
    return text + "...[truncated]"


def service_fields() -> dict[str, str]:
    """``service.name``/``service.version``/``deployment.environment`` from the env.

    ``service.name`` follows the OTel convention the standard quotes (the value of
    ``OTEL_SERVICE_NAME``); the image version and the deployment environment come from
    the deployment's env vars, never hard-coded (the standard's table).
    """
    return {
        _SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME") or "avplumber",
        _SERVICE_VERSION: (
            os.environ.get("OTEL_SERVICE_VERSION")
            or os.environ.get("RECORDER_VERSION")
            or "unknown"
        ),
        _DEPLOYMENT_ENVIRONMENT: (
            os.environ.get("DEPLOYMENT_ENVIRONMENT")
            or os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT")
            or "unknown"
        ),
    }


def _fill(
    value: str,
    args: tuple[Any, ...],
    used: int,
    mapping: Mapping[str, Any] | None,
) -> str:
    """Interpolate one token's ``%`` conversions, positionally or from a mapping."""
    count = len(list(_PLACEHOLDER.finditer(value)))
    if not count:
        if "%" in value:  # a stray conversion the regex did not recognize
            raise ValueError(f"unformattable token value {value!r}")
        return value
    if mapping is not None:
        return value % mapping
    chunk = args[used : used + count]
    if len(chunk) < count:
        raise ValueError("not enough arguments for the message's placeholders")
    return value % chunk


def _split_message(record: logging.LogRecord) -> tuple[str, dict[str, Any]]:
    """Split ``event=... key=value`` into a static body and flat attributes.

    The message is split *before* interpolation so a value with spaces (``message=%s``
    carrying an exception text) stays one attribute.  Anything that does not fit the
    convention degrades safely: the whole interpolated message becomes the body, with
    no attributes parsed from it.
    """
    args = record.args if isinstance(record.args, tuple) else ()
    mapping = record.args if isinstance(record.args, Mapping) else None
    fields: dict[str, Any] = {}
    tokens = record.msg.split() if isinstance(record.msg, str) else []
    used = 0
    event: str | None = None
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or not key or _PLACEHOLDER.search(key):
            return record.getMessage(), {}
        try:
            count = len(list(_PLACEHOLDER.finditer(value)))
            value = _fill(value, args, used, mapping)
        except (ValueError, TypeError, KeyError):
            return record.getMessage(), {}
        used += count
        if key == "event":
            event = value
            continue
        fields["attr." + key if key in _RESERVED_FIELDS else key] = value
    if event is None:
        return record.getMessage(), {}
    return event, fields


class JsonFormatter(logging.Formatter):
    """Formats one :class:`logging.LogRecord` as one OTel JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            return self._format(record)
        except Exception:
            try:
                body = f"log format failure: {record.getMessage()}"
            except Exception:
                body = "log format failure"
            return json.dumps(
                {
                    "timestamp": _utc_stamp(record.created),
                    "severity_text": "ERROR",
                    "severity_number": SEVERITY_NUMBER[logging.ERROR],
                    "body": _cap(body, 2048),
                    "logger": record.name,
                },
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )

    def _format(self, record: logging.LogRecord) -> str:
        severity_text, severity_number = severity_of(record.levelno)
        body, attributes = _split_message(record)
        exc_info = record.exc_info
        if exc_info is True:  # a record built directly, outside Logger._log
            exc_info = sys.exc_info()
        fields: dict[str, Any] = {
            "timestamp": _utc_stamp(record.created),
            "severity_text": severity_text,
            "severity_number": severity_number,
            "body": body,
            **service_fields(),
            "logger": record.name,
            "thread": record.threadName,
        }
        # The ambient span's ids (recorder.otelsetup), or ones a filter stamped;
        # absent otherwise, which keeps the lines identical to the pre-OTel format.
        trace_id, span_id = _ambient_span_ids()
        trace_id = getattr(record, "trace_id", None) or trace_id
        span_id = getattr(record, "span_id", None) or span_id
        if trace_id:
            fields["trace_id"] = trace_id
        if span_id:
            fields["span_id"] = span_id
        if exc_info and exc_info[0] is not None:
            fields["exception.type"] = exc_info[0].__name__
            fields["exception.message"] = _cap(str(exc_info[1]), MAX_ATTRIBUTE_BYTES)
            # One string field with \n, never separate log lines (the standard).
            fields["exception.stacktrace"] = _cap(
                self.formatException(exc_info), 8 * 1024
            )
        for key, value in attributes.items():
            fields[key] = (
                _cap(value, MAX_ATTRIBUTE_BYTES) if isinstance(value, str) else value
            )
        return _serialize_within_bound(fields)


def _serialize_within_bound(fields: dict[str, Any]) -> str:
    """Serialize, then shed the heaviest optional fields until the line fits 16 KB.

    Order: the stack trace first (already capped), then the largest attributes; the
    body is truncated last so the human message survives.
    """
    line = _serialize(fields)
    if len(line.encode("utf-8", errors="replace")) <= MAX_LINE_BYTES:
        return line
    fields.pop("exception.stacktrace", None)
    line = _serialize(fields)
    while len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
        droppable = [
            key
            for key in fields
            if key not in _RESERVED_FIELDS or key == "exception.stacktrace"
        ]
        if not droppable:
            break
        fields.pop(max(droppable, key=lambda key: len(str(fields[key]))))
        line = _serialize(fields)
    if len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
        fields["body"] = _cap(str(fields["body"]), MAX_ATTRIBUTE_BYTES)
        line = _serialize(fields)
    return line


def _serialize(fields: dict[str, Any]) -> str:
    return json.dumps(fields, ensure_ascii=False, default=str, separators=(",", ":"))


def wrap_line(line: str, *, logger: str, severity_text: str = "INFO") -> str:
    """Wrap one foreign text line (the graph engine's output) as a JSON log line.

    The engine's own lines cannot be restructured; they travel as the ``body`` of a
    well-formed JSON object so a single non-JSON line never breaks ``| json`` for the
    stream.      A foreign line's severity is unknown, hence the ``INFO`` default.
    """
    number = SEVERITY_BY_TEXT.get(severity_text, SEVERITY_BY_TEXT["INFO"])
    return _serialize(
        {
            "timestamp": _utc_stamp(time.time()),
            "severity_text": severity_text,
            "severity_number": number,
            "body": line,
            **service_fields(),
            "logger": logger,
        }
    )


class TextFormatter(logging.Formatter):
    """Human-readable alternative for ``RECORDER_LOG_FORMAT=text`` (local debugging).

    Same content as the JSON line, minus the framing: UTC stamp, severity, logger,
    the full interpolated message, and a multi-line exception — plain text mode is for
    a human tailing a stream, never for the shipped log path.
    """

    def format(self, record: logging.LogRecord) -> str:
        severity_text, _ = severity_of(record.levelno)
        text = (
            f"{_utc_stamp(record.created)} {severity_text:<5} "
            f"[{record.name}] {record.getMessage()}"
        )
        if record.exc_info:
            exc_info = record.exc_info
            if exc_info is True:
                exc_info = sys.exc_info()
            if exc_info and exc_info[0] is not None:
                text += "\n" + self.formatException(exc_info)
        return text


def formatter_from_env() -> logging.Formatter:
    """The formatter the deployment asked for: ``text`` (default) or ``json``."""
    if os.environ.get(_FORMAT_VAR, "").strip().lower() == "json":
        return JsonFormatter()
    return TextFormatter()


def configure_logging(*, level: int = logging.INFO, stream: Any = None) -> None:
    """Point the root logger at one handler in the configured format and stop there.

    The format comes from ``RECORDER_LOG_FORMAT`` (``json`` by default). Replaces
    every existing handler (the standard's rule 1: everything on stdout is JSON; a
    text handler alongside a JSON one guarantees broken lines). Uncaught exceptions
    in the main thread and in threads are logged instead of printed bare to stderr
    (rule 1: uncaught-exception traces go through the logger, one event per line).
    """
    root = logging.getLogger()
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(formatter_from_env())
    root.handlers = [handler]
    root.setLevel(level)
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_uncaught_thread_exception


def _log_uncaught_exception(kind: Any, value: Any, traceback: Any) -> None:
    logging.getLogger().error(
        "event=uncaught_exception", exc_info=(kind, value, traceback)
    )


def _log_uncaught_thread_exception(args: Any) -> None:
    if args.exc_type is SystemExit:
        return
    logging.getLogger().error(
        "event=uncaught_thread_exception thread=%s",
        getattr(args.thread, "name", "unknown"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
