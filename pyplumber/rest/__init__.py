"""Optional REST gateway to an avplumber instance, described by OpenAPI.

The gateway is a process (or a thread of an application's supervisor) that talks to the engine
over its control port; the engine does not link or load it. `create_app` always serves the core,
`/api/v1/health` and the read-only `/api/v1/engine`, and mounts the routers it is given: the
mixer's (`pyplumber.mixer.rest.mixer_router`) and an application's own.

FastAPI, pydantic and uvicorn (`requirements.txt` here) are needed only to build or serve an
app: they are imported on first use, so `pyplumber.rest.origin` and `pyplumber.rest.fake` load
without them.
"""


def __getattr__(name):
    if name in {"create_app", "serve_in_thread", "RestServer", "API_PREFIX"}:
        from . import app
        return getattr(app, name)
    if name in {"ApiError", "ErrorResponse"}:
        from . import errors
        return getattr(errors, name)
    if name == "ControlClient":
        from .client import ControlClient
        return ControlClient
    raise AttributeError(name)


__all__ = ["create_app", "serve_in_thread", "RestServer", "API_PREFIX", "ApiError", "ErrorResponse", "ControlClient"]
