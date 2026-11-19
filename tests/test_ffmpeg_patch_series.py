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


def test_libmxl_patch_is_opt_in():
    """Nothing of the MXL patch is compiled unless FFmpeg is configured with libmxl."""
    patch = (SERIES / "0026-avformat-libmxl-demuxer-muxer.patch").read_text()
    for required in ('mxl_muxer_deps="libmxl swresample"', 'mxl_demuxer_deps="libmxl swresample"',
                     "enabled libmxl            && require_pkg_config libmxl",
                     "OBJS-$(CONFIG_MXL_DEMUXER)", "OBJS-$(CONFIG_MXL_MUXER)",
                     "ff_mxl_demuxer", "ff_mxl_muxer"):
        assert required in patch
    # Outside its own sources the patch only adds lines.
    removed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith(("---", "-- "))]
    assert removed == []


def test_mixer_image_builds_mxl_only_on_request():
    dockerfile = (ROOT / "docker-compose/mixer/Dockerfile.fedora44").read_text()
    assert "ARG WITH_MXL=0\n" in dockerfile
    assert "FROM mxl-${WITH_MXL} AS ffmpeg\n" in dockerfile
    assert 'if [ "${WITH_MXL}" = 1 ]; then mxl=--enable-libmxl;' in dockerfile
