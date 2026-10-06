#!/usr/bin/env python3
"""Render colorful SDR pattern clips for a many-source show (needs FFmpeg with lavfi).

    python3 -m demos.mixer.sdr_patterns media/patterns --fps 60 --seconds 20 --encoder h264_nvenc

One clip per generator below; each is a distinct NVDEC-decodable source, far
cheaper at run time than raw v210 (which streams ~330 MB/s per 1080p60 input).
Every pattern carries moving grain and encodes at a contribution-like bitrate (3 to
10 Mbit/s average, 16 Mbit/s peak), so a clip costs a decoder about what camera footage
does: a plain pattern would encode to a few hundred kbit/s and flatter the source limits.
"""
import argparse
import pathlib
import subprocess

# Temporal luma noise, like camera grain: per-frame detail for the encoder and decoder.
GRAIN = "noise=c0s=10:c0f=t"
# Contribution-like input bitrates: each source averages one, 3 to 10 Mbit/s, peaking at 16.
INPUT_KBPS = range(3000, 10001, 1000)
PEAK_KBPS = 16000


def input_kbps(index: int) -> int:
    """The bitrate of source *index*, spread over INPUT_KBPS."""
    return INPUT_KBPS[index * 3 % len(INPUT_KBPS)]


def rate(kbps: int) -> list:
    return ["-b:v", f"{kbps}k", "-maxrate", f"{PEAK_KBPS}k", "-bufsize", f"{2 * kbps}k"]


def encoding_options(encoder: str, kbps: int, fps: int) -> list:
    # Bound each demo decoder's reference surfaces without reducing its playout buffer.
    references = ["-bf", "0", "-dpb_size", "2"] if encoder in ("h264_nvenc", "hevc_nvenc") else []
    return [*rate(kbps), "-g", str(fps), *references]


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
           encoder: str, ffmpeg: str, overlay=None, kbps: int = INPUT_KBPS[-1]) -> pathlib.Path:
    """*overlay* is (extra FFmpeg input arguments, filter_complex over [0:v] and [1:v],
    stdin bytes for that input), e.g. prepare_demo.id_overlay()."""
    raw = encoder == "rawvideo"
    out = directory / f"{name}.{'nv12' if raw else 'mp4'}"
    source = f"{graph}{':' if '=' in graph else '='}size={size}:rate={fps},{GRAIN}"
    inputs, graph_filter, data = overlay or ([], None, None)
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", source, *inputs,
                    *(["-filter_complex", graph_filter] if graph_filter else []), "-t", str(seconds),
                    "-c:v", encoder,
                    *(["-f", "rawvideo", "-pix_fmt", "nv12"] if raw else
                      [*encoding_options(encoder, kbps, fps), "-pix_fmt", "yuv420p"]), str(out)],
                   input=data, check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--encoder", default="h264_nvenc", help="h264_nvenc, or rawvideo for raw NV12")
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
