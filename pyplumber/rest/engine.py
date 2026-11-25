"""The engine as it runs, read-only: its version, nodes and queues.

Nothing here changes the graph. `node.add`, `node.object.set`, `group.*` and the like reach files
and URLs the engine's host can open; they stay on the control port, which is loopback by default.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Path
from pydantic import BaseModel, ConfigDict

from .client import ControlClient
from .errors import Starting, refused_while

# Node and object names go into a control-protocol line as they are, where a space or a newline
# would start another argument or another command. The engine's own names fit this.
Name = Annotated[str, Path(pattern=r"^[A-Za-z0-9_.@:-]{1,128}$")]


class EngineVersion(BaseModel):
    version: str


class Node(BaseModel):
    name: str
    type: str
    working: bool
    params: dict[str, Any]   # the node's creation-time parameters, as `node.add` received them


class Nodes(BaseModel):
    nodes: list[Node]


class NodeParams(BaseModel):
    name: str
    params: dict[str, Any]


class NodeObject(BaseModel):
    value: Any


class Queue(BaseModel):
    """One graph edge as `queues.json` reports it: fill, rates and drop counters vary with the build."""
    model_config = ConfigDict(extra="allow")
    name: str


class Queues(BaseModel):
    queues: list[Queue]


def engine_router(client: ControlClient, starting: Starting | None = None) -> APIRouter:
    router = APIRouter(prefix="/engine", tags=["engine"], dependencies=refused_while(starting))

    def ask(line: str, empty: str):
        return json.loads(client.command(line) or empty)

    @router.get("/version", response_model=EngineVersion)
    def version():
        return {"version": (client.command("version") or "").strip()}

    @router.get("/nodes", response_model=Nodes)
    def nodes():
        """Every node with its type, whether it is running, and its parameters."""
        return {"nodes": ask("nodes.json", "[]")}

    @router.get("/nodes/{name}", response_model=NodeParams)
    def node(name: Name):
        return {"name": name, "params": ask(f"node.param.get {name}", "{}")}

    @router.get("/nodes/{name}/objects/{object_name}", response_model=NodeObject)
    def node_object(name: Name, object_name: Name):
        """An object the node publishes while it runs (`node.object.get`), e.g. an input's `streams`."""
        return {"value": ask(f"node.object.get {name} {object_name}", "null")}

    @router.get("/queues", response_model=Queues)
    def queues():
        return {"queues": ask("queues.json", "[]")}

    return router
