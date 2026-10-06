"""Synthetic source allocation and measured demo capacity policies."""
import json
import math
from pathlib import Path

from . import prepare_demo
from .demo_recipe import allocate, validate_dsk
from .instance_profiles import INSTANCE_PROFILES, InstanceType
from pyplumber.mixer.color import TRANSFER_TAGS
from pyplumber.mixer.config import parse
from pyplumber.mixer.gui.setup_runtime import (
    SetupRuntime as BaseSetupRuntime, for_mode, extra_aux_limit as output_limit,
)

DEMO_DIR = Path(__file__).resolve().parent
# The demo is 1080p only: every source is a unique 1920x1080 input, in either orientation.
PROGRAM_SIZE = (1920, 1080)
DEFAULT_SETTINGS = dict(orientation="portrait", fps=60, bit_depth=10, chroma="422",
                        source_count=16, scene_count=32, layout="balanced", weights=[8, 4, 2, 0, 2, 0, 0],
                        encodes={}, dsk=[], clean_feed=False, extra_aux=0)



def _in_range(profile, kbps):
    low, high = profile["nvenc"]["bitrate_kbps"]
    return min(max(kbps, low), high)


def current_settings(profile, settings):
    """Saved *settings* in the current shape, every program encode filled in. Saved before extra
    aux outputs, they had none; the browser ring size the page no longer sets goes, and the show
    takes the frame rate's default (config.default_browser_ring_size); one program bitrate_kbps
    sets the SDR program and its clean copy, the HLG program in the defaults' ratio to it."""
    if not isinstance(settings, dict):
        return settings
    encodes, legacy = settings.get("encodes", {}), settings.get("bitrate_kbps")
    if not isinstance(encodes, dict) or len(encodes) > 64:
        raise ValueError("encodes must map at most 64 output ids to their preset and bitrate_kbps")
    defaults = profile["nvenc"]["defaults"]
    # Bound a legacy integer before converting to float; JSON can carry arbitrarily large ints.
    scale = (min(max(legacy, 0), profile["nvenc"]["bitrate_kbps"][1]) / defaults["sdr"]["bitrate_kbps"]
             if type(legacy) is int else 1)
    programs = {o: {**e, "bitrate_kbps": _in_range(profile, round(e["bitrate_kbps"] * scale))}
                for o, e in defaults.items() if o != "aux"}
    encodes = {**programs, "extra": dict(defaults["aux"]), **encodes}
    encodes = {o: {"codec": "hevc_nvenc" if o == "hdr" else "h264_nvenc", **e} if isinstance(e, dict) else e
               for o, e in encodes.items()}
    return {"extra_aux": 0, **{k: v for k, v in settings.items() if k not in ("browser_ring_size", "bitrate_kbps")},
            "encodes": encodes}


def extra_aux_limit(profile, cfg, encodes):
    return output_limit(profile, cfg, encodes, reserve_clean=True)


def source_limit(profile, fps, bit_depth=8, chroma="420"):
    """The instance's per-rate total scaled by the canvas's share, never above what the NVDEC,
    browser and upload caps carry together (v210 inputs only on a 4:2:2 canvas). An unsupported
    pair (8-bit 4:2:2) is refused by the mode checks in recipe_for."""
    profile = for_mode(profile, bit_depth, chroma)
    total = int(profile["sources"][fps] * profile["mode_share"].get(f"{bit_depth}:{chroma}", 1.0))
    return min(total, profile["nvdec_decodes"][fps] + profile["browser_windows"] + profile["raw_upload_units"][fps]
               + (profile["hlg_v210"] if chroma == "422" else 0))



def source_counts(profile, total, weights, fps=25, reserved_browsers=0):
    """*reserved_browsers* are downstream-key pages: browser inputs outside the weighted mix."""
    counts = allocate(total, weights)
    # P010 uses twice the upload bytes of NV12; SDR/HDR decode share NVDEC.
    for indices, costs, limit, name in (
            ((2, 3), (1, 1), profile["hlg_v210"], "4:2:2 upload"),
            ((4,), (1,), profile["browser_windows"] - reserved_browsers, "Browser"),
            ((0, 1), (1, 1), profile["nvdec_decodes"][fps], "Combined NVDEC"),
            ((5, 6), (1, 2), profile["raw_upload_units"][fps], "Raw 4:2:0 upload units")):
        group = [(i, cost) for i, cost in zip(indices, costs) if i < len(weights)]
        if sum(counts[i] * cost for i, cost in group) > limit:
            size = min(limit, sum(counts[i] for i, _ in group))
            while True:
                capped = allocate(size, [weights[i] for i, _ in group]) if size else [0] * len(group)
                if sum(n * cost for n, (_, cost) in zip(capped, group)) <= limit:
                    break
                size -= 1
            remaining = [0 if i in indices else w for i, w in enumerate(weights)]
            if not any(remaining):
                raise ValueError(f"{name} is limited to {limit}; enable another source type")
            counts = source_counts(profile, total - size, remaining, fps, reserved_browsers)
            for (i, _), count in zip(group, capped):
                counts[i] = count
            break
    return counts



def recipe_for(profile, settings):
    """Accept only the bounded generic setup controls, never paths or commands; *profile* is the
    instance's entry in INSTANCE_PROFILES."""
    settings = current_settings(profile, settings)
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Expected orientation, fps, source_count, scene_count, bit_depth, chroma, layout and weights")
    for key, choices in (("orientation", ("portrait", "landscape")),
                         ("fps", (25, 30, 50, 60)),
                         ("bit_depth", (8, 10)),
                         ("chroma", ("420", "422")),
                         ("layout", ("balanced", "grids", "fullscreen"))):
        if settings[key] not in choices or key in ("fps", "bit_depth") and type(settings[key]) is not int:
            raise ValueError(f"Unsupported {key}")
    dsk = settings["dsk"]
    validate_dsk(dsk, settings["clean_feed"])
    # Key pages are sources too: they take their share of the same budget. A show above the
    # limit of its rate and canvas is scaled down to it, not refused: switching 110 SDR inputs
    # at 30 fps to a 10-bit canvas keeps the mix at the capacity of the new mode.
    limit = source_limit(profile, settings["fps"], settings["bit_depth"], settings["chroma"]) - len(dsk)
    if type(settings["source_count"]) is int and settings["source_count"] > limit >= 1:
        settings = {**settings, "source_count": limit}
    for key, maximum in (("source_count", limit), ("scene_count", profile.get("max_scenes", 192))):
        value = settings[key]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    # 30: parse_aux_buses's cap on all buses. The NVENC budget bounds it further once the show's own
    # buses are known (SetupRuntime.apply).
    if type(settings["extra_aux"]) is not int or not 0 <= settings["extra_aux"] <= 30:
        raise ValueError("extra_aux must be an integer from 0 to 30")
    low, high = profile["nvenc"]["bitrate_kbps"]
    for output, encode in settings["encodes"].items():
        if not isinstance(encode, dict) or encode.get("codec") not in ("h264_nvenc", "hevc_nvenc"):
            raise ValueError(f"encodes.{output}: codec must be h264_nvenc or hevc_nvenc")
        if output == "hdr" and encode["codec"] != "hevc_nvenc":
            raise ValueError("Program HDR requires hevc_nvenc (Main10)")
        presets = profile["nvenc"]["pct_per_fps"]["hevc" if encode["codec"] == "hevc_nvenc" else "h264"]
        if set(encode) != {"codec", "preset", "bitrate_kbps"} or encode["preset"] not in presets:
            raise ValueError(f"encodes.{output} must have a preset of {', '.join(presets)} and a bitrate_kbps")
        if type(encode["bitrate_kbps"]) is not int or not low <= encode["bitrate_kbps"] <= high:
            raise ValueError(f"encodes.{output}: bitrate_kbps must be an integer from {low} to {high}")
    weights = settings["weights"]
    most = max(profile["sources"].values())   # the page sends source counts as weights
    if not isinstance(weights, list) or len(weights) != 7 or any(type(w) is not int or not 0 <= w <= most for w in weights):
        raise ValueError(f"Provide seven integer source weights from 0 to {most}")
    if settings["bit_depth"] == 8 and (any(weights[1:4]) or weights[6]):
        raise ValueError("8-bit mode supports SDR 4:2:0 and browser sources only")
    if settings["bit_depth"] == 8 and settings["chroma"] != "420":
        raise ValueError("8-bit mode supports a 4:2:0 canvas only")
    if settings["chroma"] == "420" and any(weights[2:4]):
        raise ValueError("4:2:0 mode supports 4:2:0 and browser sources only")
    counts = source_counts(for_mode(profile, settings["bit_depth"], settings["chroma"]), settings["source_count"],
                           weights, settings["fps"], len(dsk))
    _check_hdr_decodes(profile, counts[1], settings["fps"])
    width, height = PROGRAM_SIZE
    recipe = json.loads((DEMO_DIR / "demo.example.json").read_text())
    recipe.update(source_count=settings["source_count"], scene_count=settings["scene_count"])
    def program_encode(output):
        encode = settings["encodes"][output]
        return {**encode, "profile": "main10" if output == "hdr" else "main" if encode["codec"] == "hevc_nvenc" else "baseline",
                "color": "hlg" if output == "hdr" else "sdr"}
    for rendition in recipe["renditions"]:
        rendition.update(program_encode(rendition["id"]))
    recipe["clean_rendition"] = program_encode("sdr_clean")
    canvas_width, canvas_height = (height, width) if settings["orientation"] == "portrait" else (width, height)
    recipe["canvas"].update(width=canvas_width, height=canvas_height, fps=settings["fps"])
    if settings["bit_depth"] == 8:
        recipe["canvas"].update(working_format="nv12", color="sdr")
        recipe["renditions"] = [recipe["renditions"][0]]
    else:
        recipe["canvas"]["working_format"] = "p010le" if settings["chroma"] == "420" else "p210le"
    recipe["generation"].update(width=width, height=height)
    for source in recipe["inputs"][:2]:
        source.update(profile.get("generated_decode", {}))
    recipe["inputs"].append({"id": "sdr420_raw", "kind": "generated", "color": "sdr",
                             "chroma": "420", "storage": "nv12"})
    recipe["inputs"].append({"id": "hlg420_raw", "kind": "generated", "color": "hlg",
                             "chroma": "420", "storage": "p010"})
    for source, count in zip(recipe["inputs"], counts):
        source["weight"] = count
        if source["kind"] == "browser":
            source.update(width=width, height=height)
    layouts = {"fullscreen": 2}
    mode = settings["layout"]
    if mode != "fullscreen":
        layouts.update({f"grid_{n}": 2 if mode == "grids" else 1
                        for n in (2, 4, 8, 16, 32, 64) if n <= settings["source_count"]})
        if mode == "balanced":
            layouts.update(pip=3, random=3)
            if counts[4] and sum(counts[:4]) + sum(counts[5:]):
                layouts["alpha_overlay"] = 2
                for index in (0, 5, 1, 6, 2, 3):
                    if counts[index]:
                        source = recipe["inputs"][index]
                        recipe["alpha_background"] = source["id"] + ("_001" if counts[index] > 1 else "_000")
                        if counts[index] == 1 and index in (0, 5):
                            source["pattern"] = "bars"
                        break
    recipe["dsk"], recipe["clean_feed"] = dsk, settings["clean_feed"]
    recipe["setup"] = settings
    recipe["layouts"] = layouts
    return recipe



def _check_hdr_decodes(profile, count, fps):
    maximum = profile.get("nvdec_hdr_decodes", {}).get(fps, math.inf)
    if count > maximum:
        raise ValueError(f"HDR NVDEC inputs are limited to {maximum} at {fps} fps to reserve VRAM")



class SetupRuntime(BaseSetupRuntime):
    def __init__(self, media_dir, recipe_path, bridge, instance_type, *args, **kwargs):
        instance_type = InstanceType(instance_type)
        super().__init__(media_dir, recipe_path, bridge, INSTANCE_PROFILES[instance_type], *args,
                         instance_type=instance_type.value, config_name="mixer.demo.json", **kwargs)

    def normalize_settings(self, settings):
        return current_settings(self.profile, settings)

    def make_recipe(self, settings):
        return recipe_for(self.profile, settings)

    def plan(self, recipe):
        return prepare_demo.plan(recipe, self.media_dir)

    def prepare(self, recipe, progress):
        return prepare_demo.prepare(recipe, self.media_dir, progress=progress)

    def page(self):
        return (DEMO_DIR / "setup.html").read_bytes()

    def extra_aux_limit(self, cfg, encodes):
        return extra_aux_limit(self.profile, cfg, encodes)

    def _validate_capacity(self, show, settings=None):
        cfg = parse(show)
        _check_hdr_decodes(self.profile, sum(s.kind == "video" and s.color_trc not in ("", TRANSFER_TAGS["sdr"])
                                            for s in cfg.sources), cfg.fps)
        encodes = current_settings(self.profile, settings or self.settings or {})["encodes"]
        extra_aux_limit(self.profile, cfg, encodes)
