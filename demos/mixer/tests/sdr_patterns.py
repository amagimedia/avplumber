#!/usr/bin/env python3
"""Render colorful SDR pattern clips for a many-source show (needs FFmpeg with lavfi).

    sdr_patterns.py media/patterns --fps 60 --seconds 20 --encoder h264_nvenc

One clip per generator below; each is a distinct NVDEC-decodable source, far
cheaper at run time than raw v210 (which streams ~330 MB/s per 1080p60 input).
"""
import argparse
import pathlib
import subprocess

GENERATORS = {
    "testsrc2": "testsrc2",
    "bars": "smptehdbars",
    "rgbtest": "rgbtestsrc",
    "mandelbrot": "mandelbrot",
    "gradients": "gradients=speed=0.05:nb_colors=6",
    "life": "life=mold=10:life_color=#ffcc00:death_color=#3300aa:rule=B3/S23",
    "sierpinski": "sierpinski=type=triangle:seed=7",
    "cellauto": "cellauto=rule=30:scroll=1",
}


def render(directory: pathlib.Path, name: str, graph: str, size: str, fps: int, seconds: int,
           encoder: str, ffmpeg: str) -> pathlib.Path:
    raw = encoder == "rawvideo"
    out = directory / f"{name}.{'nv12' if raw else 'mp4'}"
    source = f"{graph}{':' if '=' in graph else '='}size={size}:rate={fps}"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", source, "-t", str(seconds),
                    "-c:v", encoder,
                    *(["-f", "rawvideo", "-pix_fmt", "nv12"] if raw else
                      ["-b:v", "12M", "-maxrate", "16M", "-g", str(fps), "-pix_fmt", "yuv420p"]), str(out)],
                   check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--encoder", default="h264_nvenc", help="h264_nvenc, or libx264 without a GPU")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--only", nargs="*", choices=sorted(GENERATORS), help="subset of generators")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, graph in GENERATORS.items():
        if args.only and name not in args.only:
            continue
        print(render(args.directory, name, graph, args.size, args.fps, args.seconds, args.encoder, args.ffmpeg))


if __name__ == "__main__":
    main()
