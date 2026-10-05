#!/usr/bin/env python3
"""Prepare cached demo media and expand a JSON recipe into a mixer show.

    python3 demos/mixer/prepare_demo.py demos/mixer/demo.example.json

Needs Python, NumPy and FFmpeg; the mixer image includes them. Default inputs
are synthetic. Downloads and browser inputs are enabled only by recipe weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parents[1]))

from sdr_patterns import GENERATORS, input_kbps, encoding_options, render  # noqa: E402
from demo_recipe import DSK_WINDOWS, allocate, scenes, validate_dsk, write_atomic  # noqa: E402
from graphic_pages import graphic_url, key_rects  # noqa: E402
from hdr_patterns import write_hlg  # noqa: E402
from pyplumber.mixer.config import MAX_SOURCES, default_browser_ring_size, parse  # noqa: E402

# Janus RTP port of the clean SDR program; 5004/5006 carry the keyed program and
# 5008 the multiview. The clean feed is SDR, with an independent codec.
CLEAN_PORT = 5010


def ensure_asset(path: Path, writer) -> None:
    """Publish complete files only, retaining cached assets across retries."""
    if path.is_file() and path.stat().st_size:
        print(f"Using {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {path}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        writer(staged)
        if not staged.is_file() or not staged.stat().st_size:
            raise ValueError(f"preparation produced no media for {path}")
        staged.replace(path)


WIPE_NAMES = {"diagonal": "Diagonal sweep", "panels": "Sliding panels"}


def render_wipe(path: Path, size: str, fps: int, ffmpeg: str, pattern: str = "diagonal") -> None:
    # CPU generation is preparation only; playback uses the existing RGBA wipe chain.
    position = {"diagonal": "(X/W+0.3*Y/H)/1.3",
                "panels": "if(mod(floor(6*Y/H),2),1-X/W,X/W)"}[pattern]
    # Overshoot both ends: transparent first/last frames, fully opaque around
    # the midpoint cut. Moving colour bands expose repeated frames even there.
    alpha = f"255*if(lt(T,1),lte({position},1.2*T-0.1),gte({position},1.2*(T-1)-0.1))"
    graph = (f"color=size={size}:rate={fps}:duration=2,format=rgba,"
             "geq=r='48+32*floor(5*mod(X/W+T/2,1))':g='64+144*Y/H':"
             f"b='if(lt(mod(X/W+Y/H/4+2-T/2,0.2),0.035),240,128)':a='{alpha}'")
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", graph,
                    "-an", "-c:v", "qtrle", "-pix_fmt", "argb", str(path)], check=True)


def download(path, url):
    with urlopen(Request(url, headers={"User-Agent": "avplumber-demo"}), timeout=60) as response:
        with path.open("wb") as out:
            shutil.copyfileobj(response, out)
        expected = response.headers.get("Content-Length")
        if expected is not None and path.stat().st_size != int(expected):
            raise ValueError(f"incomplete download: {url}")


# 3x5 glyphs for source ids, one octal digit per row (4 is the left pixel). The
# mixer image's FFmpeg has no drawtext (no libfreetype), so ids are drawn here.
GLYPHS = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_-", (
    "75557 26227 71747 71717 55711 74717 74757 71222 75757 75717 25755 65656 34443 65556 74647 74644 34553 "
    "55755 72227 11152 55655 44447 57755 65555 25552 65644 25563 65655 34216 72222 55557 55552 55775 55255 "
    "55222 71247 00007 00700").split()))


def id_overlay(source_id, width, height, seconds):
    """FFmpeg input and filter that burn *source_id* into every frame, so no two
    sources share a file or a picture. The plate circles a spot of its own and
    the orbit closes on the loop: every source visibly moves on every frame, even
    steady bars. Returns the render() overlay tuple; the plate goes over [0:v]."""
    text = source_id.upper()
    mask = np.pad([[int(GLYPHS[c][row], 8) >> (2 - col) & 1 if col < 3 else 0 for c in text for col in range(4)]
                   for row in range(5)], 1)
    scale = max(1, min(height // 90, width // mask.shape[1]))
    # 190 is about HLG reference white after the gray-to-limited-range conversion,
    # so labels do not glare at peak on HDR sources; SDR shows light grey.
    plate = (np.kron(mask, np.ones((scale, scale), np.uint8)) * 190).astype(np.uint8)[:height, :width]
    h, w = plate.shape
    rng = random.Random(source_id)
    radius = height // 24
    x, y = (radius + rng.randrange(max(0, room - 2 * radius) + 1) for room in (width - w, height - h))
    phase = rng.uniform(0, 2 * math.pi)
    orbit = f"2*PI*t/{seconds}+{phase:.4f}"
    return (["-f", "rawvideo", "-pix_fmt", "gray", "-video_size", f"{w}x{h}", "-i", "pipe:0"],
            # format=auto keeps 10-bit sources at 10 bits; the default is 8-bit yuv420.
            f"[0:v][1:v]overlay=format=auto:x={x}+{radius}*cos({orbit}):y={y}+{radius}*sin({orbit})",
            plate.tobytes())


# Under assets/: the native HLG patterns of one preparation run (prepare removes it).
HLG_PATTERNS = ".hlg-patterns"


def render_hlg(path, width, height, fps, seconds, variant, encoder, ffmpeg, source_id, pattern_dir, kbps):
    # Offline conversion of a native HLG signal, not SDR samples tagged as HDR. The pattern
    # depends only on its variant and geometry, and building it in Python is most of the cost,
    # so it is built once per run and every source of the variant encodes it with its own id.
    raw = pattern_dir / f"{variant}_{width}x{height}_{fps}fps_{seconds}s.v210"
    if not raw.is_file():
        pattern_dir.mkdir(parents=True, exist_ok=True)
        staged = raw.with_name(raw.name + ".partial")
        write_hlg(staged, width, height, fps * seconds, source=variant)
        staged.replace(raw)
    inputs, graph, data = id_overlay(source_id, width, height, seconds)
    pixel_format = "yuv422p10le" if encoder == "v210" else "p010le"
    options = (["-f", "rawvideo"] if encoder in ("rawvideo", "v210") else
               ["-profile:v", "main10", *encoding_options(encoder, kbps, fps)])
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "v210", "-video_size", f"{width}x{height}",
                    "-framerate", str(fps), "-i", str(raw), *inputs, "-an", "-filter_complex",
                    graph + ",setparams=range=limited:color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
                    "-c:v", encoder,
                    "-pix_fmt", pixel_format, *options,
                    "-color_range", "tv", "-color_trc", "arib-std-b67", "-color_primaries", "bt2020",
                    "-colorspace", "bt2020nc", str(path)], input=data, check=True)


def render_sdr422(path, width, height, fps, seconds, variant, ffmpeg, source_id):
    pattern = "smptehdbars" if variant == 0 else "smptebars"
    inputs, graph, data = id_overlay(source_id, width, height, seconds)
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    f"{pattern}=size={width}x{height}:rate={fps}", *inputs, "-filter_complex", graph,
                    "-t", str(seconds), "-an", "-c:v", "v210", "-pix_fmt", "yuv422p10le", "-f", "rawvideo",
                    str(path)], input=data, check=True)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def plan(recipe, media_dir, runtime_media_dir=None, ffmpeg="ffmpeg"):
    """Validate and describe all work before creating files or downloading media."""
    count = positive_int(recipe["source_count"], "source_count")
    if count > MAX_SOURCES:
        raise ValueError(f"source_count must be at most {MAX_SOURCES}")
    scene_count = positive_int(recipe["scene_count"], "scene_count")
    canvas = recipe["canvas"]
    width, height = (positive_int(canvas[k], f"canvas.{k}") for k in ("width", "height"))
    fps = positive_int(canvas.get("fps", 30), "canvas.fps")
    if width < 64 or height < 64 or width % 2 or height % 2:
        raise ValueError("canvas dimensions must be even and at least 64")
    generation = recipe.get("generation", {})
    asset_width = positive_int(generation.get("width", width), "generation.width")
    asset_height = positive_int(generation.get("height", height), "generation.height")
    if asset_width < 64 or asset_height < 64 or asset_width % 2 or asset_height % 2:
        raise ValueError("generation dimensions must be even and at least 64")
    seconds = positive_int(generation.get("seconds", 2), "generation.seconds")
    seed = recipe.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    runtime = Path(runtime_media_dir) if runtime_media_dir else media_dir
    if not runtime.is_absolute():
        raise ValueError("--runtime-media-dir must be absolute")
    inputs = recipe["inputs"]
    counts = allocate(count, [s.get("weight", 0) for s in inputs])
    input_ids = [s["id"] for s in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("input ids must be unique")
    sources, jobs, allocation, clips = [], {}, {}, set()
    size = f"{asset_width}x{asset_height}"

    def runtime_path(path):
        return str(runtime / path.relative_to(media_dir))

    for spec, amount in zip(inputs, counts):
        name = spec["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("input ids may contain only letters, digits, underscores and hyphens")
        allocation[name] = amount
        kind = spec["kind"]
        if kind not in ("generated", "download", "file", "browser"):
            raise ValueError(f"{name}: kind must be generated, download, file or browser")
        color = spec.get("color")
        if color is not None and color not in ("sdr", "hlg", "pq"):
            raise ValueError(f"{name}: color must be sdr, hlg or pq")
        for index in range(amount):
            source = {"id": f"{name}_{index:03d}", "independent": True}
            if color is not None:
                source["color"] = color
            if kind == "generated":
                chroma = spec.get("chroma")
                if color not in ("sdr", "hlg") or chroma not in ("420", "422"):
                    raise ValueError(f"{name}: generated inputs need color sdr/hlg and chroma 420/422")
                raw = spec.get("storage")
                if raw not in (None, "nv12", "p010"):
                    raise ValueError(f"{name}: raw storage must be nv12 or p010")
                if raw and (chroma != "420" or color != ("sdr" if raw == "nv12" else "hlg")):
                    raise ValueError(f"{name}: storage {raw} is supported for {'SDR' if raw == 'nv12' else 'HLG'} 4:2:0 only")
                patterns = list(GENERATORS) if color == "sdr" and chroma == "420" else ["0", "1"]
                # Cellular noise is useful for encoder stress, but produces large
                # keyframe bursts when prominent in a WebRTC demo composition.
                pool = spec.get("patterns", [p for p in patterns if p != "cellauto"])
                if "patterns" in spec and "pattern" in spec:
                    raise ValueError(f"{name}: use pattern or patterns, not both")
                if not isinstance(pool, list) or not pool or any(p not in patterns for p in pool):
                    raise ValueError(f"{name}: patterns must be a non-empty list chosen from {patterns}")
                pattern = spec.get("pattern", pool[index % len(pool)])
                if pattern not in patterns:
                    raise ValueError(f"{name}: pattern must be one of {patterns}")
                codec = spec.get("codec", "h264" if color == "sdr" else "hevc")
                if codec not in ("h264", "hevc") or color == "hlg" and codec != "hevc":
                    raise ValueError(f"{name}: codec must be h264 or hevc; HLG requires hevc")
                if "codec" in spec and (raw or chroma == "422"):
                    raise ValueError(f"{name}: codec applies only to encoded 4:2:0 inputs")
                encoder = f"{codec}_nvenc"
                storage = raw or ("v210" if chroma == "422" else encoder)
                # Version cache names when changing generation semantics. Every
                # source has its own file, with its id burned in (id_overlay), so
                # no two inputs share decode input or page cache; the name holds
                # all that shapes the content, so re-applying reuses the file.
                extension = raw or ("v210" if chroma == "422" else "mp4")
                version = 5 if extension == "mp4" else 4   # low-DPB encodes; raw pixels are unchanged
                path = (media_dir / "assets" / f"synthetic_v{version}_{size}_{fps}fps_{seconds}s" /
                        f"{source['id']}_{color}_{chroma}_{pattern}_{storage}{f'_{input_kbps(index)}k' if extension == 'mp4' else ''}.{extension}")
                encoder = "v210" if chroma == "422" else "rawvideo" if raw else encoder
                if color == "hlg":
                    writer = lambda out, pattern=pattern, encoder=encoder, sid=source["id"], kbps=input_kbps(index): render_hlg(
                        out, asset_width, asset_height, fps, seconds, int(pattern), encoder, ffmpeg, sid,
                        media_dir / "assets" / HLG_PATTERNS, kbps)
                elif chroma == "422":
                    writer = lambda out, pattern=pattern, sid=source["id"]: render_sdr422(
                        out, asset_width, asset_height, fps, seconds, int(pattern), ffmpeg, sid)
                else:
                    writer = lambda out, pattern=pattern, encoder=encoder, sid=source["id"], kbps=input_kbps(index): render(
                        out.parent, out.stem, GENERATORS[pattern], size, fps, seconds, encoder, ffmpeg,
                        id_overlay(sid, asset_width, asset_height, seconds), kbps)
                jobs[path] = writer
                source.update(kind=raw or ("v210" if chroma == "422" else "video"),
                              path=runtime_path(path), width=asset_width, height=asset_height)
            elif kind == "browser":
                # A page of its own, or a graphic under graphics/: by default the transparency test
                # source, which the older "pattern": "alpha" also asks for.
                if spec.get("pattern", "alpha") != "alpha" or len({"url", "graphic", "pattern"} & spec.keys()) > 1:
                    raise ValueError(f'{name}: a browser input takes one of url, graphic or pattern: "alpha"')
                url = spec["url"] if "url" in spec else graphic_url(spec.get("graphic", "browser_alpha"), fps, source=source["id"])
                source.update(kind="browser", url=url, width=spec.get("width", width), height=spec.get("height", height), color="sdr")
            else:
                if kind == "download":
                    url = spec["url"]
                    if urlparse(url).scheme not in ("http", "https"):
                        raise ValueError(f"{name}: download URL must use HTTP(S)")
                    suffix = Path(urlparse(url).path).suffix or ".mp4"
                    path = media_dir / "assets" / "downloads" / (hashlib.sha256(url.encode()).hexdigest()[:20] + suffix)
                    jobs[path] = lambda out, url=url: download(out, url)
                else:
                    path = media_dir / spec["path"]
                    if not path.is_relative_to(media_dir) or ".." in path.parts or not path.is_file():
                        raise ValueError(f"{name}: file path must exist under --media-dir")
                # Every source is an independent input: never two reading one clip.
                if path in clips:
                    raise ValueError(f"{name}: a download or file feeds one source only (weight 1, unique clip)")
                clips.add(path)
                source.update(kind="video", path=runtime_path(path))
            source.update({key: spec[key] for key in ("decode_storage", "extra_hw_frames") if key in spec})
            sources.append(source)
    # Graphics need fewer pixels than the program; the compositor scales them
    # in its draw pass. Bound decode/upload cost independently of canvas size.
    wipe_scale = min(1, 960 / max(width, height))
    wipe_size = f"{max(2, round(width * wipe_scale / 2) * 2)}x{max(2, round(height * wipe_scale / 2) * 2)}"
    wipes = []
    for pattern, name in WIPE_NAMES.items():
        wipe = media_dir / "media_wipes" / f"synthetic_v2_{pattern}_{wipe_size}_{fps}fps.mov"
        jobs[wipe] = lambda out, pattern=pattern: render_wipe(out, wipe_size, fps, ffmpeg, pattern)
        wipes.append({"id": pattern, "name": name, "path": runtime_path(wipe), "duration_seconds": 2})
    scene_list = scenes(sources, width, height, scene_count, recipe["layouts"], seed,
                        alpha_background=recipe.get("alpha_background"))
    # Keys are added after scene generation: ordinary sources that scenes could
    # use, but generated layouts should not scatter graphics into grids.
    pages = recipe.get("dsk", [])
    validate_dsk(pages, recipe.get("clean_feed", False))
    rects, keys = key_rects(width, height, only=pages), []
    for page in pages:
        x, y, w, h = rects[page]
        window_w, window_h = DSK_WINDOWS[page] or (width, height)   # a fill key's window is the canvas
        sources.append({"id": f"dsk_{page}", "kind": "browser", "url": graphic_url(page, fps, source=f"dsk_{page}"),
                        "width": window_w, "height": window_h, "color": "sdr"})
        keys.append({"id": page, "source": f"dsk_{page}", "dst": {"x": x, "y": y, "w": w, "h": h}})
    renditions = recipe["renditions"]
    if recipe.get("clean_feed"):
        sdr = next((r for r in renditions if r.get("color") == "sdr" or r.get("id") == "sdr"), renditions[0])
        # clean_rendition: the fields in which the clean copy differs, such as its preset and bitrate.
        clean = {**sdr, "id": f"{sdr['id']}_clean", "feed": "clean", "port": CLEAN_PORT,
                 **recipe.get("clean_rendition", {}), "color": "sdr"}
        if clean.get("codec") != sdr.get("codec") and "profile" not in recipe.get("clean_rendition", {}):
            clean["profile"] = "main" if clean.get("codec") == "hevc_nvenc" else "baseline"
        renditions = [*renditions, clean]
    doc = {"canvas": canvas, "sources": sources, "scenes": scene_list,
           "browser_ring_size": recipe.get("browser_ring_size", default_browser_ring_size(fps)),
           "initial_scene": scene_list[0]["id"], "renditions": renditions,
           "wipes": wipes,
           "wipe_color": "sdr", "control": {"direct": True, "transition": "cut", "default_wipe": "diagonal"}}
    if "aux_buses" in recipe:
        doc["aux_buses"] = recipe["aux_buses"]
    if keys:
        doc["dsk"] = {"keys": keys}
    if "max_compositor_layers" in recipe:
        doc["max_compositor_layers"] = recipe["max_compositor_layers"]
    parse(doc)   # the whole show, distinct Janus RTP/RTCP port pairs included
    return doc, jobs, allocation


def prepare(recipe, media_dir: Path, *, runtime_media_dir=None, ffmpeg="ffmpeg", progress=None) -> Path:
    """Prepare cached assets; optional progress(completed, total) counts ready files."""
    media_dir = media_dir.resolve()
    doc, jobs, allocation = plan(recipe, media_dir, runtime_media_dir, ffmpeg)
    if not shutil.which(ffmpeg):
        raise ValueError(f"FFmpeg not found: {ffmpeg}")
    print("Independent sources: " + ", ".join(f"{name}={n}" for name, n in allocation.items()), flush=True)
    try:
        if progress:
            progress(0, len(jobs))
        for completed, (path, writer) in enumerate(jobs.items(), 1):
            ensure_asset(path, writer)
            if progress:
                progress(completed, len(jobs))
    finally:
        shutil.rmtree(media_dir / "assets" / HLG_PATTERNS, ignore_errors=True)
    config_path = media_dir / "mixer.demo.json"
    write_atomic(config_path, json.dumps(doc, indent=2) + "\n")
    print(f"Config: {config_path} ({len(doc['sources'])} sources, {len(doc['scenes'])} scenes)", flush=True)
    return config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--media-dir", type=Path, default=Path("media"))
    parser.add_argument("--runtime-media-dir", type=Path, help="media mount path seen by the mixer, if different")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    try:
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        prepare(recipe, args.media_dir, runtime_media_dir=args.runtime_media_dir, ffmpeg=args.ffmpeg)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Demo preparation failed: {error}\n")


if __name__ == "__main__":
    main()
