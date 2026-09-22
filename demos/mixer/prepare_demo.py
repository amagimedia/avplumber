#!/usr/bin/env python3
"""Prepare cached demo media and expand a JSON recipe into a mixer show.

    python3 demos/mixer/prepare_demo.py demos/mixer/demo.example.json

Needs Python, NumPy and FFmpeg; the mixer image includes them. Default inputs
are synthetic. Downloads and browser inputs are enabled only by recipe weights.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parents[1]))
sys.path.insert(0, str(DEMO_DIR / "tests"))
sys.path.insert(0, str(DEMO_DIR.parents[1] / "tests/cuda"))

from sdr_patterns import GENERATORS, render  # noqa: E402
from demo_recipe import allocate, scenes  # noqa: E402
from hdr_patterns import write_hlg  # noqa: E402
from pyplumber.mixer.config import MAX_SOURCES, parse  # noqa: E402


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


def render_hlg420(path, width, height, fps, seconds, variant, encoder, ffmpeg):
    # Offline conversion of a native HLG signal, not SDR samples tagged as HDR.
    raw = path.with_suffix(".v210")
    write_hlg(raw, width, height, fps * seconds, source=variant)
    pixel_format = "p010le" if encoder == "hevc_nvenc" else "yuv420p10le"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "v210", "-video_size", f"{width}x{height}",
                    "-framerate", str(fps), "-i", str(raw), "-an",
                    "-vf", "setparams=range=limited:color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
                    "-c:v", encoder,
                    "-profile:v", "main10", "-pix_fmt", pixel_format, "-b:v", "12M", "-g", str(fps),
                    "-color_range", "tv", "-color_trc", "arib-std-b67", "-color_primaries", "bt2020",
                    "-colorspace", "bt2020nc", str(path)], check=True)


def render_sdr422(path, width, height, fps, seconds, variant, ffmpeg):
    pattern = "smptehdbars" if variant == 0 else "smptebars"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    f"{pattern}=size={width}x{height}:rate={fps}", "-t", str(seconds),
                    "-an", "-c:v", "v210", "-pix_fmt", "yuv422p10le", "-f", "rawvideo", str(path)], check=True)


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
    sources, jobs, allocation = [], {}, {}
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
                encoder = generation.get("sdr_encoder", "h264_nvenc") if color == "sdr" else generation.get("hdr_encoder", "hevc_nvenc")
                if chroma == "420" and encoder not in (("h264_nvenc", "libx264") if color == "sdr" else ("hevc_nvenc", "libx265")):
                    raise ValueError(f"{name}: use h264_nvenc/libx264 for SDR or hevc_nvenc/libx265 for HDR")
                storage = "v210" if chroma == "422" else encoder
                # Version cache names when changing generation semantics.
                extension = "v210" if chroma == "422" else "mp4"
                version = "v2" if color == "hlg" or chroma == "422" else "v1"
                path = media_dir / "assets" / f"synthetic_{version}_{size}_{fps}fps_{seconds}s" / f"{color}_{chroma}_{pattern}_{storage}.{extension}"
                if chroma == "422":
                    if color == "hlg":
                        writer = lambda out, pattern=pattern: write_hlg(out, asset_width, asset_height, fps * seconds, source=int(pattern))
                    else:
                        writer = lambda out, pattern=pattern: render_sdr422(
                            out, asset_width, asset_height, fps, seconds, int(pattern), ffmpeg)
                elif color == "hlg":
                    writer = lambda out, pattern=pattern, encoder=encoder: render_hlg420(
                        out, asset_width, asset_height, fps, seconds, int(pattern), encoder, ffmpeg)
                else:
                    writer = lambda out, pattern=pattern, encoder=encoder: render(
                        out.parent, out.stem, GENERATORS[pattern], size, fps, seconds, encoder, ffmpeg)
                jobs[path] = writer
                source.update(kind="v210" if chroma == "422" else "video", path=runtime_path(path), width=asset_width, height=asset_height)
            elif kind == "browser":
                if "pattern" in spec:
                    if spec["pattern"] != "alpha" or "url" in spec:
                        raise ValueError(f"{name}: browser pattern must be alpha, without a url")
                    page = (DEMO_DIR / "browser_alpha.html").read_bytes()
                    url = "data:text/html;base64," + base64.b64encode(page).decode("ascii")
                else:
                    url = spec["url"]
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
                source.update(kind="video", path=runtime_path(path))
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
    doc = {"canvas": canvas, "sources": sources, "scenes": scene_list,
           "initial_scene": scene_list[0]["id"], "renditions": recipe["renditions"],
           "wipes": wipes,
           "wipe_color": "sdr", "control": {"direct": True, "transition": "cut", "default_wipe": "diagonal"}}
    cfg = parse(doc)
    # Include RTCP's adjacent port in conflict checks.
    used_ports = set()
    for rendition in cfg.renditions:
        if rendition.target != "janus":
            continue
        port = rendition.port or 5004
        if not 1 <= port < 65535 or used_ports.intersection((port, port + 1)):
            raise ValueError("Janus renditions need distinct RTP/RTCP port pairs in 1..65535")
        used_ports.update((port, port + 1))
    return doc, jobs, allocation


def prepare(recipe, media_dir: Path, *, runtime_media_dir=None, ffmpeg="ffmpeg") -> Path:
    media_dir = media_dir.resolve()
    doc, jobs, allocation = plan(recipe, media_dir, runtime_media_dir, ffmpeg)
    if not shutil.which(ffmpeg):
        raise ValueError(f"FFmpeg not found: {ffmpeg}")
    print("Independent sources: " + ", ".join(f"{name}={n}" for name, n in allocation.items()), flush=True)
    for path, writer in jobs.items():
        ensure_asset(path, writer)
    config_path = media_dir / "mixer.demo.json"
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=media_dir) as temporary:
        staged = Path(temporary) / config_path.name
        staged.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        staged.replace(config_path)
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
