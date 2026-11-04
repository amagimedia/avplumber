"""Decode feed shapes the demo clips do not have: B-frames, H.264 at a high level with the most
references, an H.264 SPS without VUI (the level then decides the reorder depth), long GOPs with
parameter sets in band, a resolution change mid-stream, the largest HEVC DPB, and a supplied
interlaced sample. Each case reads the stream's headers and the pool FFmpeg will ask for
(pyplumber.mixer.config.nvdec_pool_surfaces), decodes CUarray against linear, then keeps frames back
the way a late output does, and reports what the pool did: its size, a refusal above 32 surfaces,
exhaustion, decode errors.

    python3 tests/cuda/nvdec/live_feeds.py --work <dir> --ffmpeg ffmpeg --output live_feeds.json \\
        --hwaccel-flags unsafe_output+extra_hw_frames_only+private_stream   # the show's, on the develop series

Needs an NVIDIA host. Fixtures are generated with NVENC, so a run costs a few seconds of encoder
time; the no-VUI fixture is an NVENC stream whose SPS this script rewrites. NVENC codes frames only,
so --interlaced <file> takes a field-coded or MBAFF sample from elsewhere (a broadcast 1080i H.264
transport stream, or x264 --interlaced --tff); without one that case reports what is needed.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pyplumber.mixer.config import (NVDEC_MAX_SURFACES, h264_level_reorder, nvdec_codec_surfaces,   # noqa: E402
                                    nvdec_pool_surfaces)

FRAMES = 120
BFRAMES = ["-bf", "3", "-b_ref_mode", "middle", "-g", "30"]
# One keyframe every 10 s, with parameter sets in front of each and nowhere else: a contribution feed.
LONG_GOP = ["-bf", "0", "-g", "300", "-bsf:v", "dump_extra=freq=keyframe"]
# Sixteen references at level 5.1: the most an H.264 SPS may claim (refs + reorder + 5 + extra).
MAX_REFS_51 = ["-bf", "2", "-g", "30", "-dpb_size", "16", "-level", "5.1"]
# name: (encoder, codec, container, encoder options, how the clip is finished after NVENC)
CASES = {
    "hevc_b_frames": ("hevc_nvenc", "hevc", "mp4", BFRAMES, None),
    "h264_b_frames": ("h264_nvenc", "h264", "mp4", BFRAMES, None),
    "h264_high_level": ("h264_nvenc", "h264", "mp4", ["-bf", "2", "-g", "30", "-bsf:v", "h264_metadata=level=5.1"], None),
    "h264_refs16_level51": ("h264_nvenc", "h264", "mp4", MAX_REFS_51, None),
    # The same stream with its VUI cut off: libavcodec derives the reorder depth from the level
    # (15 at 1080p level 5.1), so the pool is refs + 15 + 5 + extra, whatever the encoder meant.
    "h264_refs16_level51_no_vui": ("h264_nvenc", "h264", "ts", MAX_REFS_51, "strip_vui"),
    # B-frames under an SPS that claims level 3.0 and has no VUI: libavcodec derives a reorder depth
    # of 0 at 1080p (8100 MaxDpbMbs over 8160 macroblocks), sizes the pool for it, then grows
    # has_b_frames as pictures arrive out of order (h264_slice.c "Increasing reorder buffer") and
    # holds more frames than the pool was sized for: exhaustion at run time, no error at open.
    "h264_b_frames_wrong_level_no_vui": ("h264_nvenc", "h264", "ts", [*BFRAMES, "-bsf:v", "h264_metadata=level=3.0"],
                                         "strip_vui"),
    "hevc_max_dpb": ("hevc_nvenc", "hevc", "mp4", ["-bf", "3", "-b_ref_mode", "middle", "-g", "30", "-dpb_size", "16"], None),
    "hevc_long_gop_in_band": ("hevc_nvenc", "hevc", "ts", LONG_GOP, None),
    "h264_long_gop_in_band": ("h264_nvenc", "h264", "ts", LONG_GOP, None),
    # 1080p then 720p in one transport stream: a new SPS mid-stream recreates the decoder and its pool.
    "h264_sps_change": ("h264_nvenc", "h264", "ts", ["-bf", "0", "-g", "30"], "sps_change"),
    "hevc_sps_change": ("hevc_nvenc", "hevc", "ts", ["-bf", "0", "-g", "30"], "sps_change"),
}
EXHAUSTED = "No decoder surfaces left"
# What FFmpeg's verbose log says about a CUarray pool, and the lines that follow a lost picture.
POOL_LINES = {
    "surfaces": r"unsafe_output \+ cuarray: (\d+) decode surfaces",
    "refused": r"(?:Requested CUarray pool needs|stream needs) (\d+) surfaces",
    "capped": r"NVDEC requires (\d+) decode surfaces, exceeding",
}
LOG_COUNTS = {
    "exhausted": EXHAUSTED, "decode_errors": "Decode error", "hwaccel_setup_failed": "Failed setup for format cuarray",
    "h264_frame_num_gaps": "Frame num gap", "hevc_rps_errors": "Error constructing the frame RPS",
    "hw_decode_failed": "hardware accelerator failed",
}
# The SPS fields that size the pool and say how the pictures are coded, as trace_headers prints them.
HEADER_FIELDS = ("profile_idc", "level_idc", "general_level_idc", "max_num_ref_frames", "max_num_reorder_frames",
                 "vui_parameters_present_flag", "bitstream_restriction_flag", "sps_max_dec_pic_buffering_minus1",
                 "sps_max_num_reorder_pics", "frame_mbs_only_flag", "mb_adaptive_frame_field_flag",
                 "field_seq_flag", "pic_width_in_mbs_minus1", "pic_height_in_map_units_minus1",
                 "pic_width_in_luma_samples", "pic_height_in_luma_samples")
H264_HIGH_PROFILES = (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135)


def generate(ffmpeg, path, encoder, options, size="1920x1080", frames=FRAMES, ts_offset=0.0):
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=size={size}:rate=30", "-frames:v", str(frames), "-pix_fmt", "nv12",
                    "-c:v", encoder, "-preset", "p4", "-tune", "ull", *options,
                    "-output_ts_offset", str(ts_offset), str(path)], check=True, timeout=60)


class Bits:
    """Reads the RBSP of a NAL unit: fixed-length fields and Exp-Golomb codes (H.264 9.1)."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def u(self, n):
        value = 0
        for _ in range(n):
            value = value << 1 | self.data[self.pos >> 3] >> (7 - (self.pos & 7)) & 1
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self):
        code = self.ue()
        return (code + 1) // 2 if code % 2 else -(code // 2)


def rbsp_of(payload):
    """The NAL payload without its emulation prevention bytes (H.264 7.4.1)."""
    return payload.replace(b"\x00\x00\x03", b"\x00\x00")


def payload_of(rbsp):
    return re.sub(rb"\x00\x00(?=[\x00-\x03])", b"\x00\x00\x03", rbsp)


def vui_flag_position(sps_rbsp):
    """The bit position of vui_parameters_present_flag in an SPS RBSP (H.264 7.3.2.1.1)."""
    bits = Bits(sps_rbsp)
    profile_idc = bits.u(8)
    bits.u(16)          # constraint flags and level_idc
    bits.ue()           # seq_parameter_set_id
    chroma_format_idc = 1
    if profile_idc in H264_HIGH_PROFILES:
        chroma_format_idc = bits.ue()
        if chroma_format_idc == 3:
            bits.u(1)   # separate_colour_plane_flag
        bits.ue(), bits.ue(), bits.u(1)   # bit depths, qpprime_y_zero_transform_bypass_flag
        if bits.u(1):   # seq_scaling_matrix_present_flag
            for i in range(8 if chroma_format_idc != 3 else 12):
                if bits.u(1):
                    last = following = 8
                    for _ in range(16 if i < 6 else 64):
                        if following:
                            following = (last + bits.se() + 256) % 256
                        last = following or last
    bits.ue()           # log2_max_frame_num_minus4
    poc_type = bits.ue()
    if poc_type == 0:
        bits.ue()
    elif poc_type == 1:
        bits.u(1), bits.se(), bits.se()
        for _ in range(bits.ue()):
            bits.se()
    bits.ue(), bits.u(1), bits.ue(), bits.ue()   # max_num_ref_frames, gaps, width, height
    if not bits.u(1):   # frame_mbs_only_flag
        bits.u(1)       # mb_adaptive_frame_field_flag
    bits.u(1)           # direct_8x8_inference_flag
    if bits.u(1):       # frame_cropping_flag
        for _ in range(4):
            bits.ue()
    return bits.pos


def strip_vui(stream):
    """The stream with every SPS cut off at vui_parameters_present_flag, set to 0. The SPS keeps its
    length, padded with trailing zero bytes, so this works in a transport stream, whose packets and
    timestamps stay as they are: a raw H.264 stream with B-frames has no timestamps at all."""
    out = bytearray(stream)
    for match in re.finditer(rb"\x00\x00\x01(\x67.*?)(?=\x00\x00\x00?\x01|\Z)", stream, re.S):
        nal = match.group(1)
        rbsp = rbsp_of(nal[1:])
        cut = vui_flag_position(rbsp)
        # The bits before the flag, the flag as 0, then rbsp_trailing_bits: a one and zeros to the byte.
        bits = (int.from_bytes(rbsp, "big") >> (8 * len(rbsp) - cut)) << 2 | 1
        bits <<= -(cut + 2) % 8
        new = nal[:1] + payload_of(bits.to_bytes((cut + 2 + 7) // 8, "big"))
        if len(new) > len(nal):
            raise ValueError("an SPS without VUI came out longer")
        out[match.start(1):match.end(1)] = new.ljust(len(nal), b"\x00")
    return bytes(out)


def finish(ffmpeg, name, clip, encoder, codec, options, post, work):
    """The clip a case decodes, after its post-processing; (path, expected frames)."""
    if post == "strip_vui":
        stripped = work / f"{name}.stripped.ts"
        stripped.write_bytes(strip_vui(clip.read_bytes()))
        return stripped, FRAMES
    if post == "sps_change":
        second = work / f"{name}.720p.ts"
        generate(ffmpeg, second, encoder, options, size="1280x720", ts_offset=FRAMES / 30)
        joined = work / f"{name}.joined.ts"
        joined.write_bytes(clip.read_bytes() + second.read_bytes())
        return joined, 2 * FRAMES
    return clip, FRAMES


def headers(ffmpeg, clip):
    """The SPS fields of HEADER_FIELDS, from FFmpeg's trace_headers on the first packets; a field
    coded per sub-layer keeps its last (highest) value."""
    run = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "trace", "-i", str(clip), "-c:v", "copy",
                          "-bsf:v", "trace_headers", "-frames:v", "2", "-f", "null", "-"],
                         capture_output=True, text=True, errors="replace", timeout=60)
    found = {}
    for key in HEADER_FIELDS:
        values = re.findall(rf"\b{key}\b[^=\n]*=\s*(-?\d+)", run.stderr)
        if values:
            found[key] = int(values[-1])
    return found


def expected_pool(codec, info, extra):
    """The pool FFmpeg asks for, from the headers; None when they say too little."""
    try:
        if codec == "h264":
            refs = info["max_num_ref_frames"]
            if info.get("bitstream_restriction_flag"):
                reorder = info["max_num_reorder_frames"]
            else:
                width = 16 * (info["pic_width_in_mbs_minus1"] + 1)
                height = 16 * (info["pic_height_in_map_units_minus1"] + 1) * (2 - info["frame_mbs_only_flag"])
                reorder = h264_level_reorder(info["level_idc"], width, height) if refs else 0
            own = nvdec_codec_surfaces("h264", refs=refs, reorder=reorder)
        elif codec == "hevc":
            own = nvdec_codec_surfaces("hevc", max_dec_pic_buffering=info["sps_max_dec_pic_buffering_minus1"] + 1)
        else:
            own = nvdec_codec_surfaces(codec)
    except KeyError:
        return None
    return nvdec_pool_surfaces(own, extra)


def probe(script, clip, codec, report, expected_frames, *args):
    """Runs one decode. A failure is a result here, not an error of this script."""
    run = subprocess.run([sys.executable, str(script), str(clip), *(["--codec", codec] if codec else []),
                          "--expected-frames", str(expected_frames), "--verbose", "--output", str(report), *args],
                         capture_output=True, text=True, errors="replace", timeout=240)
    log = run.stdout + run.stderr
    result = {"exit": run.returncode, **{name: log.count(text) for name, text in LOG_COUNTS.items()}}
    for name, pattern in POOL_LINES.items():
        values = re.findall(pattern, log)
        if values:
            result[name] = int(values[-1])
    if run.returncode:
        result["tail"] = log[-400:]
    return result


def interlaced_case(args, ffprobe):
    """A supplied field-coded or MBAFF sample: its codec, frame count and field order, by ffprobe."""
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,field_order,nb_read_frames", "-of", "json", str(args.interlaced)],
                         capture_output=True, text=True, check=True, timeout=600)
    stream = json.loads(out.stdout)["streams"][0]
    return stream["codec_name"], int(stream["nb_read_frames"]), stream.get("field_order", "unknown")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--extra-hw-frames", type=int, default=12, help="the headroom a show gives its decoders")
    parser.add_argument("--hwaccel-flags", default="unsafe_output+private_stream", help="the show's decoder hwaccel_flags")
    parser.add_argument("--hold", type=int, default=9, help="frames a late output keeps: Playout's queue plus held")
    parser.add_argument("--interlaced", type=Path, help="a field-coded or MBAFF sample to decode as well")
    parser.add_argument("--only", nargs="*", choices=sorted(CASES), help="a subset of the generated cases")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("probe.py")
    extra = ["--extra-hw-frames", str(args.extra_hw_frames), "--hwaccel-flags", args.hwaccel_flags]
    results = {}
    cases = {name: case for name, case in CASES.items() if not args.only or name in args.only}
    clips = {}
    for name, (encoder, codec, container, options, post) in cases.items():
        clip = args.work / f"{name}.{container}"
        try:
            generate(args.ffmpeg, clip, encoder, options)
            clips[name] = (codec, *finish(args.ffmpeg, name, clip, encoder, codec, options, post, args.work), post)
        except (subprocess.CalledProcessError, ValueError, IndexError) as error:
            results[name] = {"fixture_failed": str(error)}
            print(name, "fixture failed:", error, flush=True)
    if args.interlaced:
        codec, frames, field_order = interlaced_case(args, args.ffprobe)
        clips["interlaced_sample"] = (codec, args.interlaced, frames, None)
        results["interlaced_sample"] = {"field_order": field_order}
    else:
        results["interlaced_sample"] = {"needs_sample": "no field-coded or MBAFF sample given (--interlaced): NVENC "
                                        "codes frames only, so one must come from a broadcast encoder or x264 --interlaced"}
    hold = ["--action", "hold", "--hold", str(args.hold)]
    for name, (codec, clip, frames, post) in clips.items():
        info = headers(args.ffmpeg, clip)
        pool = expected_pool(codec, info, args.extra_hw_frames)
        linear = args.work / f"{name}.linear.json"
        if post == "sps_change":
            # A size change mid-stream also reinitialises the probe's download filter, so the pixel
            # comparison would report the filter, not the decoder: these cases hold frames only.
            runs = {"linear": ["--mode", "cuda", *hold], "late_output": ["--mode", "cuarray", *hold, *extra]}
        else:
            runs = {"linear": ["--mode", "cuda"], "cuarray": ["--mode", "cuarray", "--reference", str(linear), *extra],
                    "late_output": ["--mode", "cuarray", *hold, *extra]}
        # A stream past the limit gives up headroom (reservePool in decoders.cpp) before it is refused.
        result = {**results.get(name, {}), "headers": info, "expected_pool": pool,
                  "fits": None if pool is None else pool - args.extra_hw_frames <= NVDEC_MAX_SURFACES}
        for run, options in runs.items():
            result[run] = probe(script, clip, codec, args.work / f"{name}.{run}.json", frames, *options)
        arrays = [result[run] for run in ("cuarray", "late_output") if run in result]
        refusal = next((r["refused"] for r in arrays if r.get("refused")), None)
        logged = next((r["surfaces"] for r in arrays if r.get("surfaces")), None)
        if refusal and pool is not None:
            result["pool_matches_model"] = refusal == pool - args.extra_hw_frames
        else:
            result["pool_matches_model"] = None if pool is None or logged is None else logged == min(pool, NVDEC_MAX_SURFACES)
        results[name] = result
        print(name, json.dumps({"expected_pool": pool, **{k: {f: v.get(f) for f in ("exit", "surfaces", "refused", "capped",
                                                                                      "exhausted", "decode_errors")}
                                                          for k, v in result.items() if isinstance(v, dict) and "exit" in v}}),
              flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1))
    # A stream with no room for its pool at all must be refused cleanly, never exhaust or corrupt;
    # one that fits must decode like the linear path, with the pool the model predicts.
    failed = []
    for name, result in results.items():
        if "fixture_failed" in result:
            failed.append(name)
        elif "linear" in result:
            runs = [result[k] for k in ("linear", "cuarray", "late_output") if k in result]
            refused = result["fits"] is False
            if (result["linear"]["exit"] or any(r["exhausted"] for r in runs) or result["pool_matches_model"] is False or
                    (not refused and any(r["exit"] for r in runs)) or
                    (refused and not all("refused" in r or r["hwaccel_setup_failed"] for r in runs[1:]))):
                failed.append(name)
    print("failed:", failed or "none", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
