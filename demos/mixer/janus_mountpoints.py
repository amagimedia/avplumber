"""Janus Streaming mountpoints of the setup's extra aux buses through the Janus HTTP API, which
needs no API secret or admin key on the demo host: one per bus, its id the bus's RTP port. Only
mountpoints whose description starts with DESCRIPTION are ever destroyed."""

from contextlib import suppress
import json
from urllib.request import Request, urlopen

DESCRIPTION = "avplumber extra aux"
TIMEOUT_SEC = 3


def _call(url, message):
    request = Request(url, json.dumps({**message, "transaction": "setup"}).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=TIMEOUT_SEC) as response:
        reply = json.load(response)
    if reply.get("janus") != "success":
        raise RuntimeError(f"Janus {message['janus']}: {reply.get('error', {}).get('reason', reply)}")
    return reply


def sync(api, ports, prune=False):
    """Create the mountpoint of each bus in *ports* ({bus id: RTP port}) that Janus lacks; *prune*
    also destroys the mountpoints made here that no bus in *ports* uses."""
    session = f"{api}/{_call(api, {'janus': 'create'})['data']['id']}"
    try:
        handle = f"{session}/{_call(session, {'janus': 'attach', 'plugin': 'janus.plugin.streaming'})['data']['id']}"

        def streaming(body):
            data = _call(handle, {"janus": "message", "body": body})["plugindata"]["data"]
            if "error" in data and "already exists" not in data["error"]:   # a create that raced another
                raise RuntimeError(f"Janus streaming {body['request']}: {data['error']}")
            return data
        listed = {m["id"]: m.get("description", "") for m in streaming({"request": "list"})["list"]}
        for bus_id, port in ports.items():
            if port not in listed:
                streaming({"request": "create", "type": "rtp", "id": port, "description": f"{DESCRIPTION} {bus_id}",
                           "audio": False, "video": True, "videoport": port, "videortcpport": port + 1,
                           "videoiface": "127.0.0.1", "videopt": 96, "videocodec": "h264",
                           "videofmtp": "profile-level-id=640028;packetization-mode=1", "permanent": False})
        for mountpoint, description in listed.items() if prune else ():
            if description.startswith(DESCRIPTION) and mountpoint not in ports.values():
                streaming({"request": "destroy", "id": mountpoint, "permanent": False})
    finally:
        with suppress(Exception):   # Janus drops an idle session itself
            _call(session, {"janus": "destroy"})
