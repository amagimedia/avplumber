"""Three fixed source atlases for clients that crop thumbnails from WebRTC video."""

from copy import deepcopy
import re

PRESET = "source_thumbnails"
BANKS = 3
PER_BANK = 64
WIDTH, HEIGHT, FPS = 1024, 576, 15


def is_thumbnail(bus):
    return isinstance(bus.get("layout"), dict) and bus["layout"].get("preset") == PRESET


def size_of(spec):
    """A bank's frame: its layout's `size` [w, h], or the 1024x576 the 192-input demo's clients expect."""
    return tuple(spec.get("size", (WIDTH, HEIGHT)))


def check_size(size):
    """A bank's frame fills an 8x8 grid of 16:9 cells exactly: 16:9, both sides a multiple of 8, up to 4K."""
    ok = (isinstance(size, (list, tuple)) and len(size) == 2 and all(type(v) is int for v in size)
          and size[0] % 8 == 0 and size[1] % 8 == 0 and size[0] * 9 == size[1] * 16 and 512 <= size[0] <= 3840)
    if not ok:
        raise ValueError("source_thumbnails size must be [w, h], 16:9, multiples of 8, 512 to 3840 wide")
    return list(size)


def cells(cfg, page, size=(WIDTH, HEIGHT)):
    """The bank's 8x8 grid: 64 sources from page * 64, cells of a 64th of the frame, no space lost."""
    first, w, h = page * PER_BANK, size[0] // 8, size[1] // 8
    return [{"role": "source", "source": i, "x": (i - first) % 8 * w,
             "y": (i - first) // 8 * h, "w": w, "h": h}
            for i in range(first, min(first + PER_BANK, len(cfg.sources)))]


BITRATE_KBPS, ENC_PRESET = 1500, "p1"


def encode_options(value):
    """A recipe's `source_thumbnails`: true for the defaults, or {"size": [w, h], "bitrate_kbps": n,
    "preset": "p1".."p7"} for sharper banks (the 192-input demo airs [1920, 1080], 3500 kbps, p5)."""
    value = value if isinstance(value, dict) else {}
    unknown = set(value) - {"size", "bitrate_kbps", "preset"}
    if unknown:
        raise ValueError(f"source_thumbnails takes size, bitrate_kbps and preset, not {sorted(unknown)}")
    bitrate, preset = value.get("bitrate_kbps", BITRATE_KBPS), value.get("preset", ENC_PRESET)
    if type(bitrate) is not int or not 100 <= bitrate <= 50000:
        raise ValueError("source_thumbnails bitrate_kbps must be an integer from 100 to 50000")
    if not re.fullmatch(r"p[1-7]", str(preset)):
        raise ValueError("source_thumbnails preset must be p1 to p7")
    return {"size": tuple(check_size(value["size"])) if "size" in value else (WIDTH, HEIGHT),
            "bitrate_kbps": bitrate, "preset": preset}


def reserve_buses(show, extra_count=0, options=None):
    """Keep stable ports and put the reserved banks before the managed extra-AUX tail. *options* are
    encode_options(): a bank is rebuilt at that size and encode on every setup, keeping only its port."""
    opts = options or encode_options(True)
    (width, height) = opts["size"]
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
                         "layout": {"preset": PRESET, "page": page,
                                    **({"size": [width, height]} if (width, height) != (WIDTH, HEIGHT) else {})},
                         "layouts": [], "max_layers": PER_BANK,
                         "renditions": [{"id": "monitor", "port": assigned, "width": width, "height": height,
                                         "fps": FPS, "codec": "h264_nvenc", "profile": "baseline", "color": "sdr",
                                         "preset": opts["preset"], "bitrate_kbps": opts["bitrate_kbps"],
                                         "dpb_size": 1}]})
    return ordinary[:split] + reserved + ordinary[split:]
