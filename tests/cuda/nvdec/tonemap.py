#!/usr/bin/env python3
"""Compare CUarray tone-map inputs to linear CUDA on an NVIDIA host.

Synthetic uploads/downloads are test boundaries. Real HEVC decodes also exercise
opaque NVDEC array planes and producer-stream lifetime with the fixed pool.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from filters import FORMATS, compare, fixture, run, run_decoded, single_graph


def operation(source: str, target: str, output_format: str | None = None) -> str:
    result = f"tonemap_cuda=transfer_in={source}:transfer_out={target}"
    return result + (f":format={output_format}" if output_format else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--sdr-clip", type=Path, help="HEVC NV12 fixture")
    parser.add_argument("--hdr-clip", type=Path, help="HEVC P010 HLG fixture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="cuarray-tonemap-") as directory:
        for fmt in FORMATS:
            path = Path(directory) / f"{fmt}.raw"
            path.write_bytes(fixture(fmt, 0))
            for source in ("sdr", "hlg", "pq"):
                identity = operation(source, source) + ",format=cuarray"
                actual = run(args.ffmpeg, [path], fmt, single_graph("cuarray", identity, fmt))
                compare(results, args.output, fmt, f"identity-{source}", path.read_bytes(), actual)
                for target in ("sdr", "hlg", "pq"):
                    for output in (("nv12",) if target == "sdr" else ("p010le", "p210le")):
                        op = operation(source, target, output)
                        baseline = run(args.ffmpeg, [path], fmt, single_graph("cuda", op, output), output)
                        actual = run(args.ffmpeg, [path], fmt, single_graph("cuarray", op, output), output)
                        compare(results, args.output, fmt, f"{source}-{target}-{output}", baseline, actual)
    for clip, source in ((args.sdr_clip, "sdr"), (args.hdr_clip, "hlg")):
        if not clip:
            continue
        for target, output in (("sdr", "nv12"), ("hlg", "p010le"), ("hlg", "p210le"), ("pq", "p010le")):
            # Match the mixer's per-frame metadata resolution, then require the
            # output contract to remain linear even for an identity frame.
            tags = ("bt709:color_primaries=bt709:color_trc=bt709" if source == "sdr"
                    else "bt2020nc:color_primaries=bt2020:color_trc=arib-std-b67")
            op = f"setparams=range=tv:colorspace={tags}," + operation("auto", target, output) + ",format=cuda"
            baseline = run_decoded(args.ffmpeg, clip, ("cuda",), op, output_format=output)
            actual = run_decoded(args.ffmpeg, clip, ("cuarray",), op, output_format=output)
            compare(results, args.output, source, f"decode-{target}-{output}", baseline, actual)
    print(f"PASS: {len(results)} CUarray tonemap pixel comparisons", flush=True)


if __name__ == "__main__":
    main()
