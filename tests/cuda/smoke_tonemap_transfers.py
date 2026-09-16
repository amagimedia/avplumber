"""Exercise the shipped FFmpeg CUDA filter against display-light references.

Run on an NVIDIA host: python3 tests/cuda/smoke_tonemap_transfers.py --ffmpeg <binary>
CPU uploads/downloads are fixture I/O only; all conversions run on CUDA frames.
No mixer instance or encoder is started or modified by this test.
"""

import argparse
import itertools
import subprocess

import numpy as np

from tonemap_reference import (
    convert_transfer_codes, display_light, encode_display_light, rgb_to_codes,
    tonemap_codes,
)

W, H = 98, 66  # Exercise CUDA pitch padding and partial thread blocks.
TRANSFERS = ("sdr", "hlg", "pq")
TAGS = {
    "sdr": ("bt709", "bt709", "bt709"),
    "hlg": ("bt2020nc", "bt2020", "arib-std-b67"),
    "pq": ("bt2020nc", "bt2020", "smpte2084"),
}


def pack(codes, depth):
    maximum = (1 << depth) - 1
    y = np.clip(np.rint(codes[..., 0]), 0, maximum)
    uv = codes[..., 1:].reshape(H // 2, 2, W // 2, 2, 2).mean(axis=(1, 3))
    uv = np.clip(np.rint(uv), 0, maximum)
    dtype = "<u2" if depth == 10 else "u1"
    shift = 6 if depth == 10 else 0
    return b"".join((p.astype(dtype) << shift).tobytes() for p in (y, uv))


def unpack(data, depth):
    a = np.frombuffer(data, "<u2" if depth == 10 else "u1")
    if depth == 10:
        assert not np.any(a & 63), "P010 storage padding bits are nonzero"
        a = a >> 6
    y = a[:W * H].reshape(H, W)
    uv = a[W * H:].reshape(H // 2, W // 2, 2).repeat(2, axis=0).repeat(2, axis=1)
    return np.concatenate((y[..., None], uv), axis=-1).astype(float)


def fixture(transfer, depth):
    # Neutral checkpoints, saturated primaries, secondaries and near-black ramps.
    palette = np.array([[0, 0, 0], [1, 1, 1], [.18, .18, .18], [.5, .5, .5],
                        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                        [0, 1, 1], [1, 0, 1], [.8, .4, .2], [.02, .02, .02]])
    yy, xx = np.indices((H, W))
    rgb = palette[(xx // 2 + yy // 2) % len(palette)]
    # Vary luma inside each chroma block to check four-pixel chroma averaging.
    rgb = rgb * np.where((xx % 2 + yy % 2) == 0, 1.0, .8)[..., None]
    return pack(rgb_to_codes(rgb, transfer, depth), depth)


def run_filter(ffmpeg, data, source, input_depth, graph, output_depth, *, fail=None, full_range=False):
    space, primaries, trc = TAGS[source]
    fmt_in = "p010le" if input_depth == 10 else "nv12"
    fmt_out = "p010le" if output_depth == 10 else "nv12"
    filters = (
        f"setparams=range={'pc' if full_range else 'tv'}:colorspace={space}"
        f":color_primaries={primaries}:color_trc={trc},hwupload_cuda,"
        f"{graph},hwdownload,format={fmt_out},showinfo"
    )
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-filter_threads", "1",
           "-f", "rawvideo", "-pixel_format", fmt_in, "-video_size", f"{W}x{H}",
           "-framerate", "60", "-i", "pipe:0", "-vf", filters, "-frames:v", "3",
           "-c:v", "rawvideo", "-threads:v", "1", "-pix_fmt", fmt_out,
           "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(cmd, input=data * 3, capture_output=True, timeout=90)
    log = result.stderr.decode(errors="replace")
    if fail:
        assert result.returncode and fail in log, (graph, log)
        return None, log
    assert result.returncode == 0, (graph, log)
    size = W * H * 3 // 2 * (2 if output_depth == 10 else 1)
    assert len(result.stdout) == 3 * size, (graph, len(result.stdout), log)
    frames = [result.stdout[i * size:(i + 1) * size] for i in range(3)]
    assert frames[0] == frames[1] == frames[2], "Repeated frames differ"
    return frames[0], log


def assert_tags(log, transfer):
    space, primaries, trc = TAGS[transfer]
    for tag in ("color_range:tv", f"color_space:{space}",
                f"color_primaries:{primaries}", f"color_trc:{trc}"):
        assert tag in log, (tag, log)


def check_reference():
    # Published PQ and HLG reference-white checkpoints, independent of the CUDA code.
    white = np.full((1, 3), 203.0)
    assert np.max(abs(encode_display_light(white, "pq") - .580689)) < 1e-6
    assert np.max(abs(encode_display_light(white, "hlg") - .749877)) < 2e-6
    for transfer in TRANSFERS:
        rgb = np.linspace(0, 1, 303).reshape(-1, 3)
        restored = encode_display_light(display_light(rgb, transfer), transfer)
        assert np.max(abs(restored - rgb)) < 1e-6, transfer
    assert np.allclose(display_light(np.ones((1, 3)), "sdr"), 203)


def check_directions(ffmpeg):
    for source, target, depth in itertools.product(TRANSFERS, TRANSFERS, (8, 10)):
        data = fixture(source, depth)
        output_depth = depth if source == target else (8 if target == "sdr" else 10)
        graph = f"tonemap_cuda=transfer_in={source}:transfer_out={target}:tonemap=none:desat=0"
        result, log = run_filter(ffmpeg, data, source, depth, graph, output_depth)
        assert_tags(log, target)
        if source == target:
            assert result == data, f"{source}: identity changed pixels"
        else:
            expected = pack(convert_transfer_codes(unpack(data, depth), source, target, depth), output_depth)
            error = np.max(abs(unpack(result, output_depth) - unpack(expected, output_depth)))
            assert error <= 2, (source, target, depth, error)
        print(f"PASS {source}->{target} {depth}-bit input, pixels and metadata", flush=True)


def check_white_and_roundtrip(ffmpeg):
    for target, white, peak in itertools.product(("hlg", "pq"), (100, 203), (1000, 2000)):
        data = pack(np.broadcast_to([235, 128, 128], (H, W, 3)), 8)
        forward = f"tonemap_cuda=transfer_in=sdr:transfer_out={target}:sdr_white={white}:hdr_peak={peak}"
        result, _ = run_filter(ffmpeg, data, "sdr", 8, forward, 10)
        encoded = encode_display_light(np.full((1, 3), white), target, white, peak)
        expected_y = np.rint(encoded[0, 0] * 876 + 64)
        values = unpack(result, 10)
        assert np.max(abs(values[..., 0] - expected_y)) <= 1, (target, white, peak)
        assert np.max(abs(values[..., 1:] - 512)) <= 1
    # Preserve saturated SDR colors, including after conversion to a P210 canvas
    # and back to P010 for delivery. No tone mapping in the return leg.
    data = fixture("sdr", 8)
    for target in ("hlg", "pq"):
        graph = (f"tonemap_cuda=transfer_in=sdr:transfer_out={target},"
                 "scale_cuda=format=p210le,scale_cuda=format=p010le,"
                 f"tonemap_cuda=transfer_in={target}:transfer_out=sdr:tonemap=none:desat=0")
        # Constant vertically, to isolate color errors from the P210 chroma
        # resampling. Pixel/chroma averaging is covered by check_directions.
        block = unpack(data, 8)[0:1, ::2].repeat(H, 0).repeat(2, 1)
        flat = pack(block, 8)
        result, _ = run_filter(ffmpeg, flat, "sdr", 8, graph, 8)
        error = np.max(abs(unpack(result, 8) - unpack(flat, 8)))
        # Near-zero RGB components amplify HDR quantization when decoded through
        # SDR gamma. Check the independently quantized roundtrip as well as the
        # six-code bound observed at the saturated endpoints of this fixture.
        forward = pack(convert_transfer_codes(unpack(flat, 8), "sdr", target, 8), 10)
        expected = pack(convert_transfer_codes(unpack(forward, 10), target, "sdr", 10), 8)
        assert np.max(abs(unpack(result, 8) - unpack(expected, 8))) <= 2
        assert error <= 6, (target, error)
        print(f"PASS SDR->{target}->P210->SDR color roundtrip (max {error} codes)", flush=True)
    data = fixture("hlg", 10)
    graph = ("tonemap_cuda=transfer_in=hlg:transfer_out=pq,"
             "tonemap_cuda=transfer_in=pq:transfer_out=hlg")
    block = unpack(data, 10)[0:1, ::2].repeat(H, 0).repeat(2, 1)
    data = pack(block, 10)
    result, _ = run_filter(ffmpeg, data, "hlg", 10, graph, 10)
    error = np.max(abs(unpack(result, 10) - unpack(data, 10)))
    assert error <= 3, error
    print(f"PASS HLG->PQ->HLG display-light roundtrip (max {error} codes)", flush=True)
    print("PASS reference-white mapping at 100/203 nits and 1000/2000-nit HLG peaks", flush=True)


def check_legacy(ffmpeg):
    for source, op in itertools.product(("hlg", "pq"),
                                       ("none", "linear", "gamma", "clip", "reinhard", "hable", "mobius")):
        data = fixture(source, 10)
        codes = unpack(data, 10)
        y, u, v = tonemap_codes(codes[..., 0], codes[..., 1], codes[..., 2], source, 10,
                                "direct" if op == "none" else op, desat=0)
        expected = np.stack((y, u, v), axis=-1)
        result, log = run_filter(ffmpeg, data, source, 10,
                                f"tonemap_cuda=transfer={source}:tonemap={op}:desat=0", 8)
        error = np.max(abs(unpack(result, 8) - unpack(pack(expected, 8), 8)))
        assert error <= 2, (source, op, error)
        assert_tags(log, "sdr")
    data = fixture("hlg", 10)
    default, _ = run_filter(ffmpeg, data, "hlg", 10, "tonemap_cuda", 8)
    alias, _ = run_filter(ffmpeg, data, "hlg", 10, "tonemap_cuda=t=hlg", 8)
    assert default == alias
    print("PASS legacy HLG/PQ: all seven operators, default and t alias", flush=True)


def check_downmapping(ffmpeg):
    for source, op in itertools.product(("hlg", "pq"),
                                       ("linear", "gamma", "clip", "reinhard", "hable", "mobius")):
        data = fixture(source, 10)
        graph = f"tonemap_cuda=transfer_in={source}:transfer_out=sdr:tonemap={op}:desat=0.5"
        result, _ = run_filter(ffmpeg, data, source, 10, graph, 8)
        expected = pack(convert_transfer_codes(unpack(data, 10), source, "sdr", 10,
                                               tonemap=op, desat=0.5), 8)
        error = np.max(abs(unpack(result, 8) - unpack(expected, 8)))
        assert error <= 2, (source, op, error)
    print("PASS explicit HLG/PQ->SDR operators and highlight desaturation", flush=True)


def check_invalid(ffmpeg):
    data = fixture("sdr", 8)
    cases = [
        ("transfer_in=sdr", "Set both transfer_in"),
        ("transfer_in=sdr:transfer_out=hlg:transfer=pq", "without legacy transfer/t"),
        ("transfer_in=sdr:transfer_out=hlg:sdr_white=2000:hdr_peak=1000", "sdr_white must not exceed"),
        ("transfer=hlg", "legacy mode needs P010"),
    ]
    for options, message in cases:
        run_filter(ffmpeg, data, "sdr", 8, "tonemap_cuda=" + options, 10, fail=message)
    run_filter(ffmpeg, data, "sdr", 8, "tonemap_cuda=transfer_in=sdr:transfer_out=hlg", 10,
               fail="Full-range input is unsupported", full_range=True)
    print("PASS invalid/conflicting options and full-range input rejected", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    check_reference()
    print("PASS independent reference checkpoints", flush=True)
    if not args.reference_only:
        check_directions(args.ffmpeg)
        check_white_and_roundtrip(args.ffmpeg)
        check_downmapping(args.ffmpeg)
        check_legacy(args.ffmpeg)
        check_invalid(args.ffmpeg)


if __name__ == "__main__":
    main()
