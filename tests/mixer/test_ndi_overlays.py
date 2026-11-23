"""The NDI overlay loops: file layout, seamless loop, alpha and safe area; no FFmpeg or GPU."""
import pytest

np = pytest.importorskip("numpy")
from demos.mixer.ndi_overlays import FONTS, NAVY, STREAMS, Font, coverage, loop, render

SMALL = dict(width=384, height=216, fps=10, seconds=10)


@pytest.fixture(scope="module", params=STREAMS)
def small(request):
    count, frame = loop(request.param, **SMALL)
    return request.param, count, [frame(n) for n in range(count + 1)]


def test_render_writes_whole_bgra_frames_back_to_back(tmp_path):
    path = tmp_path / "bugs.bgra"
    assert render(path, "bugs", 192, 108, 5, 6) == 30
    data = np.fromfile(path, np.uint8).reshape(30, 108, 192, 4)
    assert (data[7] == loop("bugs", 192, 108, 5, 6)[1](7)).all()


def test_loop_is_seamless_and_never_repeats_a_frame(small):
    _, count, frames = small
    assert count == 100
    assert (frames[count] == frames[0]).all()
    assert all((a != b).any() for a, b in zip(frames, frames[1:]))


def test_streams_are_transparent_outside_their_graphics(small):
    stream, _, frames = small
    alpha = np.stack(frames)[..., 3]
    graphics = alpha[:, 120:] if stream == "lower_third" else alpha[:, :60]
    assert graphics.any()
    assert not alpha.sum() - graphics.sum()


@pytest.mark.parametrize("stream", STREAMS)
def test_graphics_stay_in_the_graphics_safe_area(stream):
    count, frame = loop(stream)
    alpha = np.maximum.reduce([frame(n)[..., 3] for n in range(0, count, 15)])
    assert alpha[54:1026, 96:1824].any()
    assert not alpha[:54].any() and not alpha[1026:].any() and not alpha[:, :96].any() and not alpha[:, 1824:].any()


def test_name_plate_is_off_at_both_ends_of_the_loop():
    count, frame = loop("lower_third", **SMALL)
    plate = slice(0, 190)   # everything above the crawl band (rows 191 to 205)
    assert not frame(0)[plate, ..., 3].any() and not frame(count - 1)[plate, ..., 3].any()
    assert frame(count // 4)[plate, ..., 3].any()


def test_crawl_moves_a_whole_number_of_pixels_a_frame():
    frame = loop("lower_third", **SMALL)[1]
    middle = slice(120, 300)
    band = [frame(n)[193:203] for n in range(3)]
    for before, after in zip(band, band[1:]):
        assert any((after[:, middle] == before[:, middle.start + s:middle.stop + s]).all() for s in range(1, 20))


def test_alpha_is_straight_not_premultiplied():
    count, frame = loop("lower_third")
    picture = frame(count // 4)
    fade = picture[830, 900:1100]   # the plate's right end fading out, no text there
    assert ((fade[:, 3] > 0) & (fade[:, 3] < 200)).sum() > 100
    assert (np.abs(fade[fade[:, 3] > 0, :3].astype(int) - NAVY[::-1]) <= 1).all()


def test_coverage_is_exact_along_a_row_and_kerning_applies():
    square = [(2.25, 3.5, 2.25, 9.0), (2.25, 9.0, 7.75, 9.0), (7.75, 9.0, 7.75, 3.5), (7.75, 3.5, 2.25, 3.5)]
    assert coverage(square, 12, 12).sum() == pytest.approx(5.5 * 5.5, abs=1e-4)
    bold = Font(FONTS / "NotoSans-Bold.ttf")
    assert bold.width("AV", 100) < bold.width("A", 100) + bold.width("V", 100)


@pytest.mark.parametrize("stream,options", [("ticker", {}), ("bugs", {"fps": 29.97}),
                                            ("lower_third", {"seconds": 4})])
def test_rejects_unknown_streams_and_loops_that_cannot_close(stream, options):
    with pytest.raises(ValueError):
        loop(stream, **{**SMALL, **options})
