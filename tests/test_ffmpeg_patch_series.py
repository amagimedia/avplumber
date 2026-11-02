"""Guard the FFmpeg series and its GPU band-blur registration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "deps/ffmpeg/9"


def _manifest():
    return dict(line.split("=", 1) for line in (SERIES / "bases.env").read_text().splitlines() if "=" in line)


def test_ffmpeg_patch_manifest():
    manifest = _manifest()
    assert len(list(SERIES.glob("*.patch"))) == int(manifest["patch_count"])
    for suffix in ("commit", "tree"):
        value = manifest[f"upstream_{suffix}"]
        assert len(value) == 40
        assert all(c in "0123456789abcdef" for c in value)


def test_mixer_image_defaults_to_the_pinned_base():
    dockerfile = (ROOT / "docker-compose/mixer/Dockerfile.fedora44").read_text()
    assert f"ARG FFMPEG_TAG={_manifest()['upstream_ref']}\n" in dockerfile


def test_band_blur_registered_with_cuda_kernel():
    patch = (SERIES / "0009-avfilter-band-blur-cuda.patch").read_text()
    for required in ("band_blur_cuda_filter_deps", "CONFIG_BAND_BLUR_CUDA_FILTER",
                     "ff_vf_band_blur_cuda", "vf_band_blur_cuda.cu",
                     "BandBlurCompositeY", "BandBlurCompositeUV"):
        assert required in patch
