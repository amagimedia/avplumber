"""Synchronize mixer RTP mountpoints while its encoders are stopped.

Janus cannot edit an RTP codec in place. Replacements retain the existing media
ports and protection, and restore the previous contracts if any change fails.
"""

from contextlib import suppress
import json
import os
from urllib.request import Request, urlopen

from pyplumber.mixer.color import default_codec, rendition_color, rendition_format
from pyplumber.mixer.config import parse
from pyplumber.mixer.janus import janus_mountpoint_id

DESCRIPTION = "avplumber extra aux"
TIMEOUT_SEC = 3


def _create(bus_id, port, codec="h264", fmtp="profile-level-id=640028;packetization-mode=1", *, extra=True):
    return {"request": "create", "type": "rtp", "id": janus_mountpoint_id(port),
            "description": f"{DESCRIPTION if extra else 'avplumber mixer output'} {bus_id}",
            "audio": False, "video": True, "videoport": port, "videortcpport": port + 1,
            "videoiface": "127.0.0.1", "videopt": 96, "videocodec": codec,
            "videofmtp": fmtp, "permanent": False}


def outputs(show, extra_aux=0):
    """Return each program, clean feed and AUX's Janus contract from a generated show.

    *extra_aux* identifies the trailing buses that Setup owns and may prune.
    """
    cfg = parse(show)
    extra_ids = {b.id for b in cfg.aux_buses[-extra_aux:]} if extra_aux else set()
    result = {}

    def add(key, rendition, aux=False, extra=False):
        codec = rendition.codec or default_codec(cfg.working_format)
        hevc = "hevc" in codec
        color = rendition_color(cfg.out_color, codec, rendition.color or None, rendition.tonemap)
        ten_bit = not aux and rendition_format(cfg.working_format, codec, color, rendition.profile) == "p010le"
        profile = rendition.profile or (("main10" if ten_bit else "main") if hevc else "high" if aux else "baseline")
        fmtp = f"profile-id={2 if profile == 'main10' else 1}" if hevc else (
            f"profile-level-id={'640028' if profile == 'high' else '4d0028' if profile == 'main' else '42e028'};packetization-mode=1")
        result[key] = _create(key, rendition.port or 5004, "h265" if hevc else "h264", fmtp, extra=extra)

    for rendition in cfg.renditions:
        if rendition.target == "janus" and (rendition.feed != "clean" or cfg.dsk_keys):
            add(rendition.id, rendition)
    for bus in cfg.aux_buses:
        for rendition in bus.renditions:
            add(f"{bus.id}:{rendition.id}", rendition, aux=True, extra=bus.id in extra_ids)
    ids = [spec["id"] for spec in result.values()]
    if len(ids) != len(set(ids)):
        raise ValueError("Janus outputs must have distinct RTP ports and mountpoint IDs")
    return result


def _call(url, message):
    request = Request(url, json.dumps({**message, "transaction": "setup"}).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=TIMEOUT_SEC) as response:
        reply = json.load(response)
    if reply.get("janus") != "success":
        raise RuntimeError(f"Janus {message['janus']}: {reply.get('error', {}).get('reason', 'request failed')}")
    return reply


def _snapshot(info):
    """The Streaming info response uses media descriptors, also accepted by create."""
    media = info.get("media", [])
    if info.get("rtsp") or info.get("srtp") or not media or any("port" not in m for m in media):
        raise RuntimeError(f"Janus mountpoint {info['id']}: cannot preserve its RTP configuration; check JANUS_STREAMING_SECRET")
    keys = ("type", "mid", "label", "msid", "port", "rtcpport", "pt", "codec", "fmtp", "port2", "port3", "datatype")
    result = {key: info[key] for key in ("id", "name", "description", "metadata", "secret", "pin",
              "is_private", "threads", "bufferkf_ms", "bufferkf_bytes", "collision") if key in info}
    # Janus does not report bind interfaces; managed RTP sockets always bind loopback.
    streams = []
    for stream in media:
        descriptor = {**{key: stream[key] for key in keys if key in stream}, "iface": "127.0.0.1"}
        descriptor.update({dest: stream[src] for src, dest in (("videosimulcast", "simulcast"),
            ("videosvc", "svc"), ("skew_compensation", "skew")) if src in stream})
        streams.append(descriptor)
    result.update(request="create", type="rtp", permanent=False,
                  media=streams)
    return result


def _replacement(info, wanted):
    videos = [m for m in info.get("media", []) if m.get("type") == "video"]
    if len(videos) != 1:
        raise RuntimeError(f"Janus mountpoint {info['id']}: expected one video stream")
    video = videos[0]
    if video.get("codec") == wanted["videocodec"] and video.get("fmtp", "") == wanted["videofmtp"]:
        return None
    previous = _snapshot(info)
    replacement = json.loads(json.dumps(previous))
    video = next(m for m in replacement["media"] if m["type"] == "video")
    if video["port"] != wanted["videoport"]:
        raise RuntimeError(f"Janus mountpoint {info['id']}: RTP port differs from the mixer output")
    video.update(codec=wanted["videocodec"], fmtp=wanted["videofmtp"], pt=wanted["videopt"])
    return previous, replacement


def sync(api, specs, prune=False, replace=False):
    """Create missing outputs; *replace* updates codecs after the old mixer stops.

    Integer values remain accepted as legacy extra-AUX RTP ports. Pruning removes
    only extra-AUX mountpoints owned here, never programs or unrelated streams.
    """
    wanted = [value if isinstance(value, dict) else _create(key, value) for key, value in specs.items()]
    secret = os.environ.get("JANUS_STREAMING_SECRET")
    auth = {"secret": secret} if secret else {}
    session = f"{api}/{_call(api, {'janus': 'create'})['data']['id']}"
    try:
        handle = f"{session}/{_call(session, {'janus': 'attach', 'plugin': 'janus.plugin.streaming'})['data']['id']}"

        def streaming(body, race=False):
            data = _call(handle, {"janus": "message", "body": body})["plugindata"]["data"]
            if "error" in data and not (race and "already exists" in data["error"]):
                raise RuntimeError(f"Janus streaming {body['request']}: {data['error']}")
            return data

        def destroy(mid):
            streaming({"request": "destroy", "id": mid, "permanent": False, **auth})

        listed = {m["id"]: m.get("description", "") for m in streaming({"request": "list"})["list"]}
        changes = []
        for spec in wanted:
            mid = spec["id"]
            if mid not in listed:
                changes.append((None, spec))
            elif replace:
                if not listed[mid].startswith("avplumber "):
                    raise RuntimeError(f"Janus mountpoint {mid} belongs to another application")
                info = streaming({"request": "info", "id": mid, **auth})["info"]
                change = _replacement(info, spec)
                if change:
                    changes.append(change)
        undo = []
        try:
            for previous, spec in changes:
                if previous:
                    destroy(spec["id"])
                    undo.append((previous, False))
                result = streaming(spec, race=not replace)
                if "error" not in result:
                    if previous:
                        undo[-1] = (previous, True)
                    else:
                        undo.append((spec, True))
            for mid, description in listed.items() if prune else ():
                if description.startswith(DESCRIPTION) and mid not in {s["id"] for s in wanted}:
                    destroy(mid)
        except Exception as error:
            recovery = []
            for spec, created in reversed(undo):
                try:
                    if created:
                        destroy(spec["id"])
                    if "media" in spec:
                        streaming(spec)
                except Exception:
                    recovery.append(str(spec["id"]))
            if recovery:
                raise RuntimeError(f"{error}; Janus rollback failed for mountpoints {', '.join(recovery)}") from error
            raise
    finally:
        with suppress(Exception):
            _call(session, {"janus": "destroy"})
