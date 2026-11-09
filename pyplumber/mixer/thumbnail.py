"""Three fixed source atlases for clients that crop thumbnails from WebRTC video."""

from copy import deepcopy
import re

PRESET = "source_thumbnails"
BANKS = 3
PER_BANK = 64
WIDTH, HEIGHT, FPS = 1024, 576, 15


def is_thumbnail(bus):
    return isinstance(bus.get("layout"), dict) and bus["layout"].get("preset") == PRESET


def cells(cfg, page):
    first = page * PER_BANK
    return [{"role": "source", "source": i, "x": (i - first) % 8 * 128,
             "y": (i - first) // 8 * 72, "w": 128, "h": 72}
            for i in range(first, min(first + PER_BANK, len(cfg.sources)))]


def reserve_buses(show, extra_count=0):
    """Keep stable ports and put the reserved banks before the managed extra-AUX tail."""
    buses = deepcopy(show.get("aux_buses", []))
    previous = {b["id"]: b for b in buses if is_thumbnail(b)}
    ordinary = [b for b in buses if not is_thumbnail(b)]
    split = len(ordinary)
    while split and len(ordinary) - split < extra_count and re.fullmatch(r"aux[0-9]+", ordinary[split - 1]["id"]):
        split -= 1
    port = max([5012, *(r.get("port", 0) for r in show.get("renditions", [])),
                *(r.get("audio_port", 0) for r in show.get("renditions", [])),
                *(r.get(k, 0) for b in buses for r in b["renditions"] for k in ("port", "audio_port"))])
    reserved = []
    for page in range(BANKS):
        bid = f"thumbnails{page + 1}"
        if any(b["id"] == bid for b in ordinary):
            raise ValueError(f"{bid} is reserved for source thumbnails")
        if bid in previous:
            assigned = previous[bid]["renditions"][0]["port"]
        else:
            port = (port + 3) // 4 * 4 + 4
            assigned = port
        reserved.append({"id": bid, "label": f"Thumbnails {page + 1}",
                         "layout": {"preset": PRESET, "page": page}, "layouts": [], "max_layers": PER_BANK,
                         "renditions": [{"id": "monitor", "port": assigned, "width": WIDTH, "height": HEIGHT,
                                         "fps": FPS, "codec": "h264_nvenc", "profile": "baseline", "color": "sdr",
                                         "preset": "p1", "bitrate_kbps": 1500, "dpb_size": 1}]})
    return ordinary[:split] + reserved + ordinary[split:]
