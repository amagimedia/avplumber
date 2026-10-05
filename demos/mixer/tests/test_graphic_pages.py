"""Linking a graphic page: the shell, the motion engine and one graphic inlined into a data: URL.
Pure text, so it needs neither FFmpeg, NumPy nor the mixer."""
import base64
import re

import pytest

from graphic_pages import GRAPHICS_DIR, graphic_url

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
