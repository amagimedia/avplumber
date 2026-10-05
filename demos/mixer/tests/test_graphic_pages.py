"""Linking a graphic page: the shell, the motion engine and one graphic inlined into a data: URL;
and what the mixer reads from the graphics' OGraf manifests: the key list, windows and placement.
Pure text, so it needs neither FFmpeg, NumPy nor the mixer."""
import base64
import json
import re

import pytest

from graphic_pages import GRAPHICS_DIR, graphic_url, key_graphics, key_rects, manifests

SHELL = ("<!doctype html>\n<html lang=\"en\">\n<body>\n<script>/*@motion.js*/</script>\n"
         "<script>/*@graphic.js*/</script>\n<script>Motion.host(graphic);</script>\n</body>\n</html>\n")
ENGINE = "const Motion = { host() {} };  // é, and a marker a careless replace would fill: /*@graphic.js*/\n"
GRAPHIC = "const graphic = { note: 'the shell says <html lang> and /*@motion.js*/' };\n"


@pytest.fixture
def root(tmp_path):
    (tmp_path / "host.html").write_text(SHELL, encoding="utf-8")
    (tmp_path / "motion.js").write_text(ENGINE, encoding="utf-8")
    (tmp_path / "plate").mkdir()
    (tmp_path / "plate" / "graphic.js").write_text(GRAPHIC, encoding="utf-8")
    return tmp_path


def page(url: str) -> str:
    prefix, payload = url.split(",", 1)
    assert prefix == "data:text/html;base64"
    return base64.b64decode(payload, validate=True).decode("utf-8")


def test_page_is_the_shell_with_engine_then_graphic_inlined(root):
    assert page(graphic_url("plate", 50, root=root)) == (
        "<!doctype html>\n<html data-fps=\"50\" lang=\"en\">\n<body>\n"
        f"<script>{ENGINE}</script>\n<script>{GRAPHIC}</script>\n"
        "<script>Motion.host(graphic);</script>\n</body>\n</html>\n")


def test_data_become_escaped_attributes_after_the_frame_rate(root):
    html = page(graphic_url("plate", 60, root=root, source="browser_007", label='A "quoted" <name> & more'))
    assert ('<html data-fps="60" data-source="browser_007" '
            'data-label="A &quot;quoted&quot; &lt;name&gt; &amp; more" lang="en">') in html
    assert html.count("data-fps=") == 1


def test_same_inputs_give_the_same_url(root):
    # The mixer reopens a browser window whose URL changed, so linking must be repeatable.
    assert graphic_url("plate", 25, root=root, source="a") == graphic_url("plate", 25, root=root, source="a")
    assert graphic_url("plate", 25, root=root, source="a") != graphic_url("plate", 25, root=root, source="b")


@pytest.mark.parametrize("fps", [0, -25, True, "50", None, float("nan")])
def test_frame_rate_must_be_a_positive_number(root, fps):
    with pytest.raises(ValueError, match="fps must be a positive number"):
        graphic_url("plate", fps, root=root)


@pytest.mark.parametrize("key", ["Source", "my-key", "9lives", "a b"])
def test_data_names_must_be_plain_lowercase_words(root, key):
    # data-my-key would read back as dataset.myKey: the graphic would not find the name it declared.
    with pytest.raises(ValueError, match="data name"):
        graphic_url("plate", 50, root=root, **{key: "x"})


@pytest.mark.parametrize("name", ["missing", "../plate", "plate/", ""])
def test_unknown_graphic_is_named(root, name):
    with pytest.raises(ValueError, match="no graphic"):
        graphic_url(name, 50, root=root)


@pytest.mark.parametrize("text", ["const s = '</script>';", "const s = '</SCRIPT >';", "// <!-- note"])
def test_script_text_that_would_end_the_inline_script_is_refused(root, text):
    (root / "plate" / "graphic.js").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"plate/graphic\.js"):
        graphic_url("plate", 50, root=root)


@pytest.mark.parametrize("shell", [SHELL.replace("/*@motion.js*/", ""), SHELL + "<script>/*@graphic.js*/</script>",
                                   SHELL.replace("<html", "<main")])
def test_a_shell_without_exactly_one_of_each_marker_is_refused(root, shell):
    (root / "host.html").write_text(shell, encoding="utf-8")
    with pytest.raises(ValueError, match=r"host\.html"):
        graphic_url("plate", 50, root=root)


def test_the_real_shell_and_engine_link(tmp_path):
    for name in ("host.html", "motion.js"):
        (tmp_path / name).write_bytes((GRAPHICS_DIR / name).read_bytes())
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "graphic.js").write_text(
        "const graphic = Motion.graphic({ html: '<div id=\"a\"></div>' });\n", encoding="utf-8")
    html = page(graphic_url("sample", 60, root=tmp_path, source="browser_001"))
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts == [(GRAPHICS_DIR / "motion.js").read_text(encoding="utf-8"),
                       "const graphic = Motion.graphic({ html: '<div id=\"a\"></div>' });\n", "Motion.host(graphic);"]
    assert '<html data-fps="60" data-source="browser_001" lang="en">' in html
    assert "/*@" not in html
    assert not re.search(r"\b(src|href)=", html), "the page must load nothing at run time"


# ---- The graphics that ship, and their manifests -------------------------------------------------

MANIFEST = {"$schema": "https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json",
            "id": "test.plate", "name": "Plate", "main": "graphic.js",
            "supportsRealTime": True, "supportsNonRealTime": True}
KEY = {"window": {"width": 400, "height": 100}, "key": {"order": 1, "anchor": "bottom-left"}}


def add_graphic(root, name, manifest):
    (root / name).mkdir(exist_ok=True)
    (root / name / "graphic.js").write_text(GRAPHIC, encoding="utf-8")
    if manifest is not None:
        (root / name / f"{name}.ograf.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_every_graphic_has_a_manifest_and_links_at_every_rate():
    found = manifests()
    assert set(found) == {path.parent.name for path in GRAPHICS_DIR.glob("*/graphic.js")}
    assert set(found) == {path.parent.name for path in GRAPHICS_DIR.glob("*/*.ograf.json")}
    assert {"browser_alpha", "lower_third", "ticker", "bug_left", "bug_right"} <= set(found)
    assert len({manifest["id"] for manifest in found.values()}) == len(found)
    for name in found:
        for fps in (25, 30, 50, 60):
            html = page(graphic_url(name, fps, source=f"dsk_{name}"))
            assert f'<html data-fps="{fps}" data-source="dsk_{name}" lang="en">' in html
            assert "/*@" not in html and "Motion.graphic({" in html


def test_the_key_list_and_the_windows_come_from_the_manifests():
    # In the setup page's order; each window is the graphic's own rectangle.
    assert list(key_graphics().items()) == [
        ("lower_third", {"label": "Lower third · animated name plate", "window": (1016, 172), "anchor": "bottom-left", "above": "ticker"}),
        ("ticker", {"label": "Ticker · second lower-third layer", "window": (1920, 80), "anchor": "bottom"}),
        ("bug_left", {"label": "Corner bug · top left", "window": (152, 152), "anchor": "top-left"}),
        ("bug_right", {"label": "Clock bug · top right", "window": (304, 152), "anchor": "top-right"})]


def test_every_key_window_is_a_size_the_browser_service_exports():
    compose = (GRAPHICS_DIR.parent / "compose.yaml").read_text(encoding="utf-8")
    (allowed,) = re.findall(r"^\s*DMA_BROWSER_ALLOWED_DIMS: (\S+)$", compose, flags=re.M)
    for name, key in key_graphics().items():
        assert "{}x{}".format(*key["window"]) in allowed.split(","), f"allow the window of {name} in compose.yaml"


@pytest.mark.parametrize("canvas,rects", [
    # A 1080p canvas, landscape or portrait, shows every window 1:1; the ticker spans the width.
    ((1920, 1080), {"lower_third": (32, 764, 1016, 172), "ticker": (0, 968, 1920, 80),
                    "bug_left": (32, 32, 152, 152), "bug_right": (1584, 32, 304, 152)}),
    ((1080, 1920), {"lower_third": (32, 1640, 1016, 172), "ticker": (0, 1844, 1080, 44),
                    "bug_left": (32, 32, 152, 152), "bug_right": (744, 32, 304, 152)}),
    ((1280, 720), {"lower_third": (22, 508, 678, 114), "ticker": (0, 644, 1280, 54),
                   "bug_left": (22, 22, 102, 102), "bug_right": (1056, 22, 202, 102)}),
    ((3840, 2160), {"lower_third": (64, 1528, 2032, 344), "ticker": (0, 1936, 3840, 160),
                    "bug_left": (64, 64, 304, 304), "bug_right": (3168, 64, 608, 304)}),
    ((96, 64), {"lower_third": (2, 46, 60, 10), "ticker": (0, 58, 96, 4),
                "bug_left": (2, 2, 10, 10), "bug_right": (76, 2, 18, 10)}),
])
def test_keys_sit_where_their_manifests_place_them(canvas, rects):
    # The rectangles of the hand-written placement these manifests replaced.
    assert key_rects(*canvas) == rects
    for x, y, w, h in rects.values():
        assert x >= 0 and y >= 0 and x + w <= canvas[0] and y + h <= canvas[1] and not (x % 2 or y % 2 or w % 2 or h % 2)


def test_a_new_key_needs_only_its_manifest(root):
    add_graphic(root, "plate", {**MANIFEST, "v_avplumber": KEY})
    add_graphic(root, "score", {**MANIFEST, "id": "test.score", "name": "Score", "description": "top strip",
                                "v_avplumber": {"window": {"width": 1920, "height": 60}, "key": {"order": 0, "anchor": "top"}}})
    add_graphic(root, "note", {**MANIFEST, "id": "test.note", "name": "Note", "v_avplumber": {
        "window": {"width": 200, "height": 50}, "key": {"order": 2, "anchor": "bottom-right", "above": "plate"}}})
    add_graphic(root, "test_card", {**MANIFEST, "id": "test.card", "name": "Not a key"})
    assert list(key_graphics(root).items()) == [
        ("score", {"label": "Score · top strip", "window": (1920, 60), "anchor": "top"}),
        ("plate", {"label": "Plate", "window": (400, 100), "anchor": "bottom-left"}),
        ("note", {"label": "Note", "window": (200, 50), "anchor": "bottom-right", "above": "plate"})]
    assert key_rects(1920, 1080, root) == {"score": (0, 32, 1920, 60), "plate": (32, 948, 400, 100),
                                           "note": (1688, 866, 200, 50)}


@pytest.mark.parametrize("change,message", [
    (None, r"plate: no manifest plate\.ograf\.json"),
    ({"name": None}, r"plate\.ograf\.json: needs \"name\""),
    ({"main": "index.js"}, r"plate\.ograf\.json: \"main\" must be graphic\.js"),
    ({"$schema": "https://example.org/schema.json"}, r"plate\.ograf\.json: \"\$schema\" must be https://ograf\.ebu\.io/"),
    ({"window": {"width": 400, "height": 100}}, r"plate\.ograf\.json: unknown field \"window\"; OGraf allows only its own and v_ fields"),
    ({"v_avplumber": {**KEY, "placement": "left"}}, r"plate\.ograf\.json: v_avplumber: unknown field \"placement\""),
    ({"v_avplumber": {"key": KEY["key"]}}, r"plate\.ograf\.json: v_avplumber\.window: needs whole, positive \"width\" and \"height\""),
    ({"v_avplumber": {**KEY, "window": {"width": 400.5, "height": 100}}}, r"v_avplumber\.window: needs whole, positive"),
    ({"v_avplumber": {**KEY, "key": {"order": 1, "anchor": "left"}}}, r"v_avplumber\.key\.anchor: must be one of top, bottom, top-left"),
    ({"v_avplumber": {**KEY, "key": {"anchor": "top"}}}, r"v_avplumber\.key\.order: must be a number"),
    ({"v_avplumber": {**KEY, "key": {**KEY["key"], "above": "ticker"}}}, r"v_avplumber\.key\.above: no key graphic \"ticker\""),
    ({"v_avplumber": {**KEY, "key": {**KEY["key"], "above": "plate"}}}, r"v_avplumber\.key\.above: \"plate\" stacks above itself"),
    ({"v_avplumber": {**KEY, "key": {"order": 1, "anchor": "top-left", "above": "other"}}}, r"v_avplumber\.key\.above: only a bottom anchor stacks"),
])
def test_a_wrong_manifest_is_named(root, change, message):
    add_graphic(root, "other", {**MANIFEST, "id": "test.other", "v_avplumber": KEY})
    add_graphic(root, "plate", None if change is None else {key: value for key, value in {**MANIFEST, **change}.items() if value is not None})
    with pytest.raises(ValueError, match=message):
        key_rects(1920, 1080, root)


def test_two_graphics_cannot_share_an_id(root):
    add_graphic(root, "plate", MANIFEST)
    add_graphic(root, "other", MANIFEST)
    with pytest.raises(ValueError, match=r"plate\.ograf\.json: id \"test\.plate\" is also the id of other"):
        manifests(root)
