"""Drive a running mixer through every scene with cuts and fades and check that its
Janus outputs keep flowing. Run against a live deployment, not a fixture::

    python3 smoke_live_takes.py --host 127.0.0.1 --port 7777 --janus http://127.0.0.1:8088/janus --mountpoints 1 2

Every take must leave the requested scene on program with the transition idle and
every listed Janus mountpoint receiving packets younger than --max-age-ms.
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pyplumber.mixer.control import AvpConnection, mixer_command, parse_mixer_status, parse_scene_list  # noqa: E402


def janus_ages(url, mountpoints, secret):
    def post(path, body):
        req = urllib.request.Request(path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)
    sid = post(url, {"janus": "create", "transaction": "t"})["data"]["id"]
    hid = post(f"{url}/{sid}", {"janus": "attach", "plugin": "janus.plugin.streaming", "transaction": "t"})["data"]["id"]
    ages = {}
    for mp in mountpoints:
        info = post(f"{url}/{sid}/{hid}", {"janus": "message", "transaction": "t",
                                           "body": {"request": "info", "id": mp, "secret": secret}})
        ages[mp] = info["plugindata"]["data"]["info"]["media"][0].get("age_ms")
    return ages


async def run(args):
    c = AvpConnection(args.host, args.port)
    await c.connect()
    scenes = parse_scene_list(await c.command(f"mixer.scenes {args.mixer}"))
    print("scenes:", scenes, flush=True)
    failures = 0

    async def take(kind, scene, **payload):
        nonlocal failures
        await c.command(mixer_command("preview", args.mixer, scene=scene))
        await c.command(mixer_command(kind, args.mixer, scene=scene, **payload))
        await asyncio.sleep(payload.get("duration_sec", 0) + args.settle)
        status = parse_mixer_status(await c.command(f"mixer.status {args.mixer}"))
        ages = janus_ages(args.janus, args.mountpoints, args.secret) if args.janus else {}
        ok = status.pgm_scene == scene and status.transition == "idle" and \
            all(a is not None and a < args.max_age_ms for a in ages.values())
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {kind:<4} -> {scene:<18} pgm={status.pgm_scene:<18} "
              f"transition={status.transition:<5} janus age ms={ages}", flush=True)

    for scene in scenes:
        await take("cut", scene)
    for scene in scenes:
        await take("fade", scene, duration_sec=args.fade)
    for i in range(args.rapid):
        await take("cut", scenes[i % len(scenes)])
    await c.disconnect()
    print("RESULT:", "PASS" if not failures else f"FAIL ({failures} takes)", flush=True)
    return 0 if not failures else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--mixer", default="mixer")
    p.add_argument("--janus", default="", help="Janus HTTP API base URL; empty skips the output check")
    p.add_argument("--mountpoints", type=int, nargs="*", default=[1])
    p.add_argument("--secret", default="avpsecret")
    p.add_argument("--fade", type=float, default=0.5)
    p.add_argument("--rapid", type=int, default=10, help="back-to-back cuts after the scene sweep")
    p.add_argument("--settle", type=float, default=1.2, help="seconds to wait after each take before checking")
    p.add_argument("--max-age-ms", type=int, default=400)
    sys.exit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
