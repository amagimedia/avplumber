"""The gateway's one error shape, `{"error": {"code", "message"}}`, and what maps onto it.

Clients branch on the HTTP status and `code`; `message` is for people and carries the engine's
own text where there is one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from pyplumber.mixer.control import AvpProtocolError


log = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None   # per-field findings of an invalid request


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    """An error a route answers with: raise it from a router, the application's included."""

    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


# What an application that starts the engine itself knows and the control port does not say: a
# reason while the engine is starting, else None.
Starting = Callable[[], str | None]


def refused_while(starting: Starting | None) -> list:
    """Router dependencies that answer 503 `engine_not_ready` with the reason while the engine is
    `starting`. An engine building its graph accepts the control connection and answers nothing,
    so without them each request waits for its timeout, behind those that arrived before it."""
    if starting is None:
        return []

    def check() -> None:
        reason = starting()
        if reason:
            raise ApiError(503, "engine_not_ready", reason)
    return [Depends(check)]


# Every operation documents these; FastAPI's own 422 schema is replaced by this one.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse, "description": description} for status, description in (
        (401, "`unauthorized`: the gateway requires a bearer token and the request has none, or another one"),
        (403, "`cross_origin`: a browser sent the request for another site's page"),
        (404, "`not_available`: this show or engine has no such command (e.g. keys in a show without a keyer)"),
        (409, "`engine_rejected`: the engine refused the command, `message` is its reason; "
              "`conflict`: the resource changed since the revision the request names"),
        (422, "`invalid_request`: the request does not match the schema, `details` lists the fields"),
        (500, "`internal_error`: the gateway failed; its log has the cause"),
        (502, "`bad_engine_reply`: the engine's reply is not JSON, or lacks a field this route's schema promises"),
        (503, "`engine_unreachable`: no control connection to the engine; "
              "`engine_not_ready`: the engine is starting and does not take this command yet"),
        (504, "`engine_timeout`: the engine did not answer in time; the command may still have run"),
    )
}


def _reply(status: int, code: str, message: str, *, details=None, headers=None) -> JSONResponse:
    body = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    """Starlette picks the handler of the most specific class, so TimeoutError (an OSError)
    and JSONDecodeError get their own answers."""

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError):
        return _reply(exc.status, exc.code, exc.message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):
        # The offending input may hold what the client considers private, and a context may hold
        # an exception object, which is not JSON: location, type and message suffice.
        details = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return _reply(422, "invalid_request", "; ".join(
            f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in details), details=details)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _reply(exc.status_code, code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(AvpProtocolError)
    async def engine_refused(_request: Request, exc: AvpProtocolError):
        # The message is the engine's status line: "400 Unknown command: x", "400 manager not
        # ready: x" while the engine starts, or "500 ERROR: why".
        status, _, text = str(exc).partition(" ")
        if status == "400" and text.startswith("Unknown command"):
            return _reply(404, "not_available", text)
        if status == "400":
            return _reply(503, "engine_not_ready", text)
        return _reply(409, "engine_rejected", text.removeprefix("ERROR: "))

    @app.exception_handler(TimeoutError)
    async def engine_timeout(_request: Request, _exc: TimeoutError):
        return _reply(504, "engine_timeout", "the engine did not answer in time; the command may still have run")

    async def engine_unreachable(_request: Request, exc: Exception):
        return _reply(503, "engine_unreachable", str(exc) or type(exc).__name__)
    for unreachable in (OSError, EOFError):   # ConnectionError is an OSError
        app.add_exception_handler(unreachable, engine_unreachable)

    @app.exception_handler(json.JSONDecodeError)
    async def bad_engine_reply(_request: Request, exc: json.JSONDecodeError):
        return _reply(502, "bad_engine_reply", f"the engine's reply is not JSON: {exc}")

    @app.exception_handler(ResponseValidationError)
    async def unexpected_engine_reply(_request: Request, exc: ResponseValidationError):
        # The reply is JSON, without a field the route's schema promises (another engine version).
        fields = ", ".join(sorted({".".join(map(str, e["loc"][1:])) for e in exc.errors()}))
        return _reply(502, "bad_engine_reply", f"the engine's reply does not match this route's schema: {fields}")

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        # The text of an unexpected exception may name paths or settings: it goes to the log only.
        log.error("%s %s failed", request.method, request.url.path, exc_info=exc)
        return _reply(500, "internal_error", f"the gateway failed on this request ({type(exc).__name__}); see its log")
