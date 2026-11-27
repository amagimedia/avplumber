#!/usr/bin/env python3
"""Smoke test of a running REST gateway (python -m pyplumber.rest): reads the engine and the
mixer, previews and cuts to another scene, checks that the mixer reports each, then puts the
original Program and Preview back. Standard library only; exits non-zero at the first failure.

    tools/rest_smoke.py                       # http://127.0.0.1:7682
    tools/rest_smoke.py --url http://<host>:7682 --token-env AVP_REST_TOKEN
    tools/rest_smoke.py --read-only           # on air: nothing is taken

The takes are visible on Program: use --read-only on a show that is on air.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


class Failure(Exception):
    pass


class Api:
    def __init__(self, url: str, token: str | None):
        self.base = url.rstrip("/") + "/api/v1"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def call(self, method: str, path: str, body=None):
        """The decoded reply, None for 204."""
        data = None if body is None else json.dumps(body).encode()
        headers = {**self.headers, **({"Content-Type": "application/json"} if data else {})}
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            raise Failure(f"{method} {path}: HTTP {exc.code} {exc.read().decode(errors='replace')}") from None
        except OSError as exc:
            raise Failure(f"{method} {path}: {exc}") from None
        print(f"ok  {method} {path}")
        return json.loads(content) if content else None

    def wait_status(self, what: str, reached, seconds: float = 5.0) -> dict:
        """The mixer status once `reached(status)`: a take reaches Program some frames after its answer."""
        deadline = time.monotonic() + seconds
        while True:
            status = self.call("GET", "/mixer/status")
            if reached(status):
                return status
            if time.monotonic() >= deadline:
                raise Failure(f"{what} not reported within {seconds:g} s; status: {json.dumps(status)}")
            time.sleep(0.1)


def run(api: Api, read_only: bool) -> None:
    health = api.call("GET", "/health")
    if health["status"] != "ok":
        raise Failure(f"the gateway does not reach the engine: {health['engine'].get('error')}")
    print(f"    engine: {health['engine']['version']}")
    print(f"    {len(api.call('GET', '/engine/nodes')['nodes'])} nodes")

    paths = api.call("GET", "/openapi.json")["paths"]
    if "/api/v1/mixer/status" not in paths:
        print("    this gateway serves no mixer (start it with --mixer NAME): engine checks only")
        return
    scenes = api.call("GET", "/mixer/scenes")["scenes"]
    before = api.call("GET", "/mixer/status")
    print(f"    Program {before['pgm_scene']!r}, Preview {before['pvw_scene']!r}, scenes {scenes}")
    if read_only:
        return
    other = next((scene for scene in scenes if scene != before["pgm_scene"]), None)
    if other is None:
        raise Failure("the show has no second scene to take")
    if before["transition"] != "idle":
        raise Failure(f"a {before['transition']} is on air: not taking over it")

    try:
        api.call("PUT", "/mixer/preview", {"scene": other})
        api.wait_status(f"Preview {other!r}", lambda status: status["pvw_scene"] == other)
        if api.call("POST", "/mixer/takes", {"type": "cut", "scene": other})["superseded"]:
            raise Failure("the cut was superseded: something else is taking on this mixer")
        api.wait_status(f"Program {other!r}", lambda status: on_program(status, other))
    finally:
        api.call("POST", "/mixer/takes", {"type": "cut", "scene": before["pgm_scene"]})
        # The mixer refuses a preview while the cut is on its way to Program.
        api.wait_status(f"Program {before['pgm_scene']!r} again", lambda status: on_program(status, before["pgm_scene"]))
        if before["pvw_scene"]:
            api.call("PUT", "/mixer/preview", {"scene": before["pvw_scene"]})


def on_program(status: dict, scene: str) -> bool:
    return status["pgm_scene"] == scene and status["transition"] == "idle"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:7682", help="the gateway, without /api/v1")
    parser.add_argument("--token-env", metavar="VAR", help="environment variable holding the gateway's bearer token")
    parser.add_argument("--read-only", action="store_true", help="only read: no preview, no take")
    args = parser.parse_args(argv)
    try:
        run(Api(args.url, os.environ.get(args.token_env) if args.token_env else None), args.read_only)
    except Failure as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1
    print("passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
