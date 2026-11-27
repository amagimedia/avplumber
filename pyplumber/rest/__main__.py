"""`python -m pyplumber.rest`: the REST gateway as a process of its own, beside a running engine.

An application that already holds a `MixerBridge` (the mixer's control page) serves the gateway
on a thread instead, with `serve_in_thread(create_app(bridge, [...]))`, so both share one
ordering of takes.
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from pyplumber.mixer.gui.bridge import MixerBridge
from pyplumber.mixer.rest import ClientBridge, mixer_router

from .app import API_PREFIX, create_app, server_config
from .fake import FakeEngine


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pyplumber.rest",
                                     description="REST gateway to a running avplumber engine, described by OpenAPI")
    parser.add_argument("--host", default="127.0.0.1", help="engine control host")
    parser.add_argument("--port", type=int, default=7777, help="engine control port")
    parser.add_argument("--mixer", metavar="NAME",
                        help="also serve /mixer for the mixer of this name (the mixer application's is 'mixer')")
    parser.add_argument("--bind", default="127.0.0.1", help="address to serve the API on")
    parser.add_argument("--http-port", type=int, default=7682)
    parser.add_argument("--token-env", metavar="VAR",
                        help="require 'Authorization: Bearer <token>', the token read from this environment variable")
    parser.add_argument("--fake", action="store_true",
                        help="serve an in-memory engine with a mixer instead of connecting to one: for writing clients")
    args = parser.parse_args(argv)

    token = None
    if args.token_env:
        token = os.environ.get(args.token_env)
        if not token:
            parser.error(f"--token-env: the environment variable {args.token_env} is empty or not set")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.fake:
        client = FakeEngine(args.mixer or "mixer")
        routers = [mixer_router(ClientBridge(client, client.mixer))]
    else:
        # A connection of its own for engine reads: a large nodes.json never holds back a take.
        client = MixerBridge(args.host, args.port, "")
        routers = [mixer_router(MixerBridge(args.host, args.port, args.mixer))] if args.mixer else []
    app = create_app(client, routers, token=token)
    logging.getLogger(__name__).info("REST API for %s on http://%s:%d%s/docs",
                                     "a fake engine" if args.fake else f"the engine at {args.host}:{args.port}",
                                     args.bind, args.http_port, API_PREFIX)
    uvicorn.Server(server_config(app, args.bind, args.http_port)).run()


if __name__ == "__main__":
    main()
