"""The gateway application: the core routes, the routers it is given, and a server to embed."""

from __future__ import annotations

import ipaddress
import logging
import re
import secrets
import threading
import time
from typing import Iterable

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .client import ControlClient
from .engine import engine_router
from .errors import ERROR_RESPONSES, ApiError, install_error_handlers
from .origin import cross_origin

API_PREFIX = "/api/v1"
log = logging.getLogger(__name__)


class EngineHealth(BaseModel):
    reachable: bool
    version: str | None = None
    error: str | None = None


class Health(BaseModel):
    status: str   # "ok", or "degraded" while the engine does not answer
    engine: EngineHealth


def _operation_id(route: APIRoute) -> str:
    """`mixer_take` rather than FastAPI's `take_api_v1_mixer_takes_post`: generated clients name
    their methods after it. The tag keeps two routers' functions apart."""
    return f"{route.tags[0]}_{route.name}" if route.tags else route.name


def _operations(router: APIRouter) -> Iterable[tuple[str, str]]:
    """(method, path) of each operation, parameter names removed: `/a/{x}` and `/a/{y}` are one path."""
    for route in router.routes:
        if isinstance(route, APIRoute):
            path = re.sub(r"\{[^}]*\}", "{}", API_PREFIX + route.path)
            for method in sorted(route.methods):
                yield method, path


def create_app(client: ControlClient, routers: Iterable[APIRouter] = (), *, token: str | None = None,
               title: str = "avplumber REST API") -> FastAPI:
    """The gateway for the engine behind `client`: `/api/v1/health`, the read-only `/api/v1/engine`
    and each of `routers` under `/api/v1`, in one OpenAPI document (`/api/v1/openapi.json`, browsable
    at `/api/v1/docs`).

    `routers` are additive: one whose operation another router or the core already serves is a
    ValueError, so a client written against the core finds the same routes in every application.
    They inherit the bearer check, the cross-origin check and the error handlers; raise `ApiError`
    for an answer of their own.

    `token`: when set, every route but health and the documentation needs `Authorization: Bearer
    <token>`. Without one, restrict who reaches the port.
    """
    app = FastAPI(title=title, version="1.0.0", openapi_url=f"{API_PREFIX}/openapi.json",
                  docs_url=f"{API_PREFIX}/docs", redoc_url=None, generate_unique_id_function=_operation_id)
    app.state.token_required = bool(token)
    install_error_handlers(app)

    # Declared whether or not this deployment sets a token, so the document and the clients
    # generated from it are the same for both.
    bearer = HTTPBearer(auto_error=False, description="Needed only when the gateway was started with a token.")

    def guard(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if cross_origin(request.headers):
            raise ApiError(403, "cross_origin", "Use this API from its own origin, or from a client that is not a browser")
        if token and (credentials is None or not secrets.compare_digest(credentials.credentials.encode(), token.encode())):
            raise ApiError(401, "unauthorized", "A valid bearer token is required", {"WWW-Authenticate": "Bearer"})

    @app.get(f"{API_PREFIX}/health", response_model=Health, tags=["meta"])
    def health():
        """Whether the gateway reaches the engine. Always 200: the gateway itself is up."""
        try:
            version = (client.command("version", 2.0) or "").strip()
        except Exception as exc:
            return {"status": "degraded", "engine": {"reachable": False, "error": str(exc) or type(exc).__name__}}
        return {"status": "ok", "engine": {"reachable": True, "version": version}}

    served = {("GET", f"{API_PREFIX}/{path}"): "the core" for path in ("health", "openapi.json", "docs")}
    for router in (engine_router(client), *routers):
        name = f"router {router.prefix or router.tags or router!r}"
        for operation in _operations(router):
            if operation in served:
                raise ValueError(f"{name}: {operation[0]} {operation[1]} is already served by {served[operation]}")
            served[operation] = name
        app.include_router(router, prefix=API_PREFIX, dependencies=[Depends(guard)], responses=ERROR_RESPONSES)
    return app


def _loopback(bind: str) -> bool:
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return bind == "localhost"


def server_config(app: FastAPI, bind: str, port: int) -> uvicorn.Config:
    if not app.state.token_required and not _loopback(bind):
        log.warning("REST API on %s:%d has no token: anyone who reaches the port controls the instance", bind, port)
    return uvicorn.Config(app, host=bind, port=port, log_level="warning")


class RestServer:
    """A gateway served on a thread of the embedding application."""

    def __init__(self, app: FastAPI, bind: str, port: int):
        self.bind = bind
        self._server = uvicorn.Server(server_config(app, bind, port))
        # uvicorn installs its signal handlers only on the main thread: the application keeps its own.
        self._thread = threading.Thread(target=self._server.run, name="rest-api", daemon=True)

    def start(self, timeout: float = 10.0) -> "RestServer":
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._server.started:   # the port is taken: uvicorn logged why and left its thread
            raise RuntimeError(f"REST API could not listen on {self.bind}")
        return self

    @property
    def port(self) -> int:
        """The port listened on, for a server started on port 0."""
        return self._server.servers[0].sockets[0].getsockname()[1]

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(5)


def serve_in_thread(app: FastAPI, bind: str = "127.0.0.1", port: int = 7682) -> RestServer:
    """Serve `app` in the background and return once it listens; `close()` the result to stop it."""
    return RestServer(app, bind, port).start()
