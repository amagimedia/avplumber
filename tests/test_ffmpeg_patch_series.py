"""Guard the shared FFmpeg series and its GPU band-blur registration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_ffmpeg_patch_manifest():
    series = ROOT / "deps/ffmpeg/8"
    manifest = dict(line.split("=", 1) for line in
                    (series / "bases.env").read_text().splitlines() if "=" in line)
    assert len(list(series.glob("*.patch"))) == int(manifest["patch_count"])
    for base in ("n80", "n81"):
        for suffix in ("commit", "tree"):
            value = manifest[f"{base}_{suffix}"]
            assert len(value) == 40
            assert all(c in "0123456789abcdef" for c in value)


def test_band_blur_registered_with_cuda_kernel():
    patch = (ROOT / "deps/ffmpeg/8/0010-avfilter-band-blur-cuda.patch").read_text()
    for required in ("band_blur_cuda_filter_deps", "CONFIG_BAND_BLUR_CUDA_FILTER",
                     "ff_vf_band_blur_cuda", "vf_band_blur_cuda.cu",
                     "BandBlurCompositeY", "BandBlurCompositeUV"):
        assert required in patch
