#!/usr/bin/env python3
"""Cut the Noto Sans fonts here down to what ndi_overlays.py reads (needs fontTools and uharfbuzz):

    python3 demos/mixer/fonts/subset.py NotoSans-Regular.ttf NotoSans-Bold.ttf

The sources are the static, unhinted Noto Sans TTFs of
https://github.com/notofonts/notofonts.github.io (fonts/NotoSans/unhinted/ttf), under OFL.txt.
Keeps Latin-1 and common punctuation, decomposes accented glyphs into plain outlines, and moves the
GPOS pair kerning, as HarfBuzz applies it, into a format 0 'kern' table: the overlay renderer reads
only simple glyf outlines, hmtx and kern.
"""
import itertools
from pathlib import Path
import sys

from fontTools import subset
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0
import uharfbuzz as hb

UNICODES = [*range(0x20, 0x7F), *range(0xA0, 0x100), *range(0x2010, 0x2016), *range(0x2018, 0x201F),
            0x2020, 0x2021, 0x2022, 0x2026, 0x2030, 0x2039, 0x203A, 0x20AC, 0x2122]


def kerning(path, cmap):
    """Pair adjustments in font units: the first glyph's shaped advance minus its plain one."""
    font = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
    pairs = {}
    for a, b in itertools.product(cmap, repeat=2):
        buf = hb.Buffer()
        buf.add_str(chr(a) + chr(b))
        buf.guess_segment_properties()
        hb.shape(font, buf, {"kern": True, "liga": False, "clig": False})
        first = buf.glyph_infos[0].codepoint
        delta = buf.glyph_positions[0].x_advance - font.get_glyph_h_advance(first)
        if delta and len(buf.glyph_infos) == 2:
            pairs[cmap[a], cmap[b]] = delta
    return pairs


def cut(path):
    options = subset.Options()
    options.drop_tables += ["GPOS", "GSUB", "GDEF"]
    options.layout_features = []
    options.notdef_outline = True
    options.glyph_names = False
    font = TTFont(path)
    cmap = {c: g for c, g in font.getBestCmap().items() if c in UNICODES}
    pairs = kerning(path, cmap)
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=UNICODES)
    subsetter.subset(font)
    glyphs, glyf = font.getGlyphSet(), font["glyf"]
    for name in font.getGlyphOrder():
        if glyf[name].isComposite():
            pen = DecomposingRecordingPen(glyphs)
            glyphs[name].draw(pen)
            flat = TTGlyphPen(None)
            pen.replay(flat)
            glyf[name] = flat.glyph()
    kept = set(font.getGlyphOrder())
    table = KernTable_format_0()
    table.version, table.coverage, table.format, table.tupleIndex = 0, 1, 0, None
    table.kernTable = {pair: value for pair, value in pairs.items() if set(pair) <= kept}
    font["kern"] = newTable("kern")
    font["kern"].version, font["kern"].kernTables = 0, [table]
    font.save(Path(__file__).resolve().parent / Path(path).name)


if __name__ == "__main__":
    for source in sys.argv[1:]:
        cut(source)
