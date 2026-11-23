#!/usr/bin/env python3
"""Looped broadcast graphics for the demo's NDI sources with alpha (needs NumPy):

    python3 -m demos.mixer.ndi_overlays lower_third.bgra --stream lower_third

A stream is a full canvas of straight (not premultiplied) BGRA pictures, back to back in a
headerless file. "lower_third" carries the name plate with the news crawl under it, "bugs" the two
corner bugs. The look follows the HTML keys in dsk/ (colours, proportions, and Noto Sans, which
those pages get as system-ui in the browser image), kept inside the EBU R 95 graphics-safe area.
Every motion repeats a whole number of times per loop and the crawl moves a whole number of pixels
a frame, so the last picture flows into the first. The right bug shows LIVE, not the HTML key's
wall clock: a clock cannot loop.
"""
import argparse
from fractions import Fraction
import math
from pathlib import Path
import struct

import numpy as np

STREAMS = ("lower_third", "bugs")
FONTS = Path(__file__).resolve().parent / "fonts"
NAVY, ORANGE, WHITE, RED = (0x10, 0x1C, 0x2B), (0xFF, 0x5A, 0x1F), (0xFF, 0xFF, 0xFF), (0xFF, 0x2B, 0x2B)
# What the lower third and the crawl show when no scene asks for its own; the lower third splits
# the loop evenly between its people.
PEOPLE = (("Ada Lovelace", "Analyst · Engine No. 1"), ("Grace Hopper", "Compiler desk"))
HEADLINES = ("Straight alpha over the program", "Keys cut on the next frame", "Scenes change underneath")
CRAWL_SPEED = 180   # pixels a second on a 1080-line canvas
SUBSAMPLES = 16     # sample rows a pixel when filling outlines


class Font:
    """The simple glyf outlines, advances and format 0 kerning of a font cut by fonts/subset.py.
    The mixer image's FFmpeg is built without freetype (no drawtext) and the image has no Pillow,
    so the overlays set and fill text themselves."""

    def __init__(self, path):
        data = self._data = Path(path).read_bytes()
        tables = {data[12 + 16 * i:16 + 16 * i].decode("latin-1"): struct.unpack_from(">I", data, 20 + 16 * i)[0]
                  for i in range(struct.unpack_from(">H", data, 4)[0])}
        head, hhea, kern = tables["head"], tables["hhea"], tables["kern"]
        self.em, = struct.unpack_from(">H", data, head + 18)
        self.cap_height, = struct.unpack_from(">h", data, tables["OS/2"] + 88)
        glyphs, = struct.unpack_from(">H", data, tables["maxp"] + 4)
        metrics, = struct.unpack_from(">H", data, hhea + 34)
        self.advance = struct.unpack_from(f">{2 * metrics}H", data, tables["hmtx"])[::2]
        self.advance += self.advance[-1:] * (glyphs - metrics)
        long_offsets = struct.unpack_from(">h", data, head + 50)[0] == 1
        offsets = struct.unpack_from(f">{glyphs + 1}{'I' if long_offsets else 'H'}", data, tables["loca"])
        self._loca = [tables["glyf"] + offset * (1 if long_offsets else 2) for offset in offsets]
        self.cmap = self._unicode(data, tables["cmap"])
        pairs, = struct.unpack_from(">H", data, kern + 10)
        values = struct.unpack_from(">" + "HHh" * pairs, data, kern + 18)
        self.kern = {(values[i], values[i + 1]): values[i + 2] for i in range(0, len(values), 3)}
        self._outlines = {}

    @staticmethod
    def _unicode(data, cmap):
        for i in range(struct.unpack_from(">H", data, cmap + 2)[0]):
            platform, encoding, offset = struct.unpack_from(">HHI", data, cmap + 4 + 8 * i)
            if (platform, encoding) == (3, 1):   # Windows Unicode BMP, format 4
                break
        else:
            raise ValueError("font has no Unicode BMP cmap")
        table = cmap + offset
        segments = struct.unpack_from(">H", data, table + 6)[0] // 2
        ends, starts, deltas, ranges = (struct.unpack_from(f">{segments}H", data, table + 14 + k * 2 * segments + (k > 0) * 2)
                                        for k in range(4))
        glyphs, at = {}, table + 16 + 6 * segments
        for i, (start, end, delta, offset) in enumerate(zip(starts, ends, deltas, ranges)):
            for char in range(start, min(end, 0xFFFE) + 1):
                glyph = struct.unpack_from(">H", data, at + 2 * i + offset + 2 * (char - start))[0] if offset else char
                if glyph and (glyph + delta) & 0xFFFF:
                    glyphs[chr(char)] = (glyph + delta) & 0xFFFF
        return glyphs

    def _outline(self, glyph):
        """Line segments (n, 4: x0 y0 x1 y1, font units, y up) of a simple glyph's contours."""
        if glyph in self._outlines:
            return self._outlines[glyph]
        data, at = self._data, self._loca[glyph]
        lines = []
        if at < self._loca[glyph + 1]:
            contours, = struct.unpack_from(">h", data, at)
            ends = struct.unpack_from(f">{contours}H", data, at + 10)
            at += 12 + 2 * contours + struct.unpack_from(">H", data, at + 10 + 2 * contours)[0]
            flags = []
            while len(flags) <= ends[-1]:
                flags += [data[at]] * (1 + (data[at + 1] if data[at] & 8 else 0))
                at += 2 if data[at] & 8 else 1
            axes = []
            for short, same in ((2, 16), (4, 32)):
                value, axis = 0, []
                for flag in flags:
                    if flag & short:
                        value += data[at] if flag & same else -data[at]
                        at += 1
                    elif not flag & same:
                        value += struct.unpack_from(">h", data, at)[0]
                        at += 2
                    axis.append(value)
                axes.append(axis)
            begin = 0
            for end in ends:
                lines += _contour([(x, y, flag & 1) for x, y, flag in zip(axes[0][begin:end + 1], axes[1][begin:end + 1],
                                                                          flags[begin:end + 1])])
                begin = end + 1
        self._outlines[glyph] = np.array(lines, float).reshape(-1, 4)
        return self._outlines[glyph]

    def width(self, text, size, tracking=0.0):
        """Advance width in pixels of *text* at *size* pixels an em, *tracking* em between glyphs."""
        return self.line(text, size, tracking=tracking)[1]

    def line(self, text, size, x=0.0, baseline=0.0, tracking=0.0):
        """Segments in pixels (y down) of *text* set from *x* on *baseline*, and its advance width."""
        pen, previous, parts = 0.0, None, []
        for char in text:
            glyph = self.cmap.get(char, 0)
            pen += self.kern.get((previous, glyph), 0)
            parts.append(self._outline(glyph) + (pen, 0, pen, 0))
            pen += self.advance[glyph] + tracking * self.em
            previous = glyph
        scale = size / self.em
        segments = np.concatenate([np.zeros((0, 4)), *parts]) * (scale, -scale, scale, -scale) + (x, baseline, x, baseline)
        return segments, max(0.0, pen - tracking * self.em) * scale


def _contour(points):
    """Lines along a closed TrueType contour of (x, y, on curve) points, quadratic arcs flattened."""
    full = []
    for i, point in enumerate(points):   # the on-curve point implied between two off-curve ones
        before = points[i - 1]
        if not point[2] and not before[2]:
            full.append(((point[0] + before[0]) / 2, (point[1] + before[1]) / 2, 1))
        full.append(point)
    first = next(i for i, point in enumerate(full) if point[2])
    full = full[first:] + full[:first + 1]
    lines, i = [], 0
    while i < len(full) - 1:
        (x0, y0, _), (x1, y1, on) = full[i], full[i + 1]
        if on:
            lines.append((x0, y0, x1, y1))
            i += 1
            continue
        x2, y2, _ = full[i + 2]
        # A chord of 1/n of the arc strays |p0 - 2 p1 + p2| / 4n² from it: keep that within 2 units.
        n = max(1, math.ceil(math.sqrt(math.hypot(x0 - 2 * x1 + x2, y0 - 2 * y1 + y2) / 8)))
        t = np.linspace(0, 1, n + 1)
        xs, ys = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2
        lines += zip(xs[:-1], ys[:-1], xs[1:], ys[1:])
        i += 2
    return lines


def coverage(segments, height, width):
    """Anti-aliased nonzero fill of closed outlines given as line *segments* (n, 4: x0 y0 x1 y1 in
    pixels, y down) on a height x width grid: SUBSAMPLES rows a pixel, exact coverage along each."""
    x0, y0, x1, y1 = np.asarray(segments, float).reshape(-1, 4).T
    first, last = (np.clip(np.ceil(y * SUBSAMPLES - 0.5), 0, height * SUBSAMPLES).astype(int)
                   for y in (np.minimum(y0, y1), np.maximum(y0, y1)))
    counts = last - first
    edge = np.repeat(np.arange(counts.size), counts)
    row = first[edge] + np.arange(edge.size) - np.repeat(np.cumsum(counts) - counts, counts)
    x = x0[edge] + ((row + 0.5) / SUBSAMPLES - y0[edge]) * ((x1 - x0) / np.where(y1 == y0, 1, y1 - y0))[edge]
    x = np.clip(x, 0, width)
    column, winding = np.floor(x).astype(int), np.sign(y1 - y0)[edge]
    stride = width + 2
    cells = height * SUBSAMPLES * stride
    at = row * stride + column
    # A crossing at x enters (or leaves) the part right of x: split it between its pixel and the next.
    steps = np.bincount(at, winding * (1 - x + column), cells) + np.bincount(at + 1, winding * (x - column), cells)
    cover = np.cumsum(steps.reshape(height * SUBSAMPLES, stride)[:, :width], axis=1)
    return np.minimum(np.abs(cover), 1).reshape(height, SUBSAMPLES, width).mean(axis=1, dtype=np.float32)


def _box(height, width, left, top, right, bottom, radius=0.0):
    """Coverage of a rectangle with rounded corners on a height x width grid of pixel centres."""
    y, x = np.mgrid[0:height, 0:width] + 0.5
    hx, hy = (right - left) / 2, (bottom - top) / 2
    qx, qy = np.abs(x - (left + right) / 2) - hx + radius, np.abs(y - (top + bottom) / 2) - hy + radius
    distance = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius
    return np.clip(0.5 - distance, 0, 1).astype(np.float32)


def _disc(height, width, cx, cy, radius):
    y, x = np.mgrid[0:height, 0:width] + 0.5
    return np.clip(radius + 0.5 - np.hypot(x - cx, y - cy), 0, 1).astype(np.float32)


def _paint(cover, colour, opacity=1.0):
    """A premultiplied RGBA layer of *colour* over *cover*."""
    alpha = cover * opacity
    return np.dstack([alpha * (c / 255) for c in colour] + [alpha]).astype(np.float32)


def _over(below, above):
    below *= 1 - above[..., 3:]
    below += above
    return below


def _blit(below, above, x, opacity=1.0, left=0):
    """Composite *above* onto *below* at column x, clipped to columns from *left* on."""
    a, b = max(x, left, 0), min(x + above.shape[1], below.shape[1])
    if b > a and opacity > 0:
        _over(below[:, a:b], above[:, a - x:b - x] * np.float32(opacity))


def _straight_bgra(layer):
    alpha = layer[..., 3:]
    colour = np.divide(layer[..., 2::-1], alpha, out=np.zeros_like(layer[..., :3]), where=alpha > 0)
    return np.rint(np.concatenate([colour, alpha], axis=2) * 255).clip(0, 255).astype(np.uint8)


def _bezier(x1, y1, x2, y2):
    """CSS cubic-bezier() timing function."""
    def ease(u):
        lo, hi = 0.0, 1.0
        for _ in range(24):
            s = (lo + hi) / 2
            lo, hi = (s, hi) if 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s * s * x2 + s ** 3 < u else (lo, s)
        return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s ** 3
    return ease


SETTLE, EASE_IN = _bezier(.2, .8, .2, 1), _bezier(.42, 0, 1, 1)   # the HTML lower third's in and out


def _phase(t, start, length, ease=lambda u: u):
    return 0.0 if t <= start else 1.0 if t >= start + length else ease((t - start) / length)


def _crawl_strip(font, headlines, size, height, baseline, minimum):
    """The crawl as a premultiplied strip whose width is a whole number of copies of *headlines*,
    an orange dot centred in each gap; the gaps share the room left over (at least 2.2 em)."""
    widths = [font.width(text, size) for text in headlines]
    copies = max(1, int(minimum // (sum(widths) + 2.2 * size * len(widths))))
    period = max(minimum, math.ceil(sum(widths) + 2.2 * size * len(widths)))
    gap = (period - copies * sum(widths)) / (copies * len(widths))
    segments, dots, x = [], [], 0.0
    for _ in range(copies):
        for text, text_width in zip(headlines, widths):
            dots.append(x + gap / 2)
            segments.append(font.line(text, size, x + gap, baseline)[0])
            x += gap + text_width
    strip = _paint(coverage(np.concatenate(segments), height, period), WHITE)
    r = 0.13 * size
    for dot in dots:
        at = math.floor(dot - 2 * r)
        _blit(strip, _paint(_disc(height, math.ceil(4 * r) + 1, dot - at, baseline - 0.36 * size, r), ORANGE), at)
    return strip


def _lower_third(regular, bold, width, height, fps, count, people, headlines):
    """Name plate (orange accent bar, navy plate fading out to the right, name over role) that
    builds on and off once a person, then the crawl band under it; both within graphics safe."""
    u, left, bottom = min(width, height) / 1080, round(0.05 * width), round(0.95 * height)
    if count % len(people) or count / len(people) / fps < 2.5:
        raise ValueError(f"{count} frames cannot give each of {len(people)} people an even 2.5 s or more")
    cycle = count // len(people)
    # Crawl band: LATEST tag, then headlines moving a whole number of pixels a frame.
    band_h, band_w, size = round(72 * u), width - 2 * left, 36 * u
    baseline = band_h / 2 + bold.cap_height / bold.em * size / 2
    band = _paint(np.ones((band_h, band_w), np.float32), NAVY, 0xDD / 255)
    tag_w = round(bold.width("LATEST", size, 0.06) + 0.9 * band_h)
    _over(band[:, :tag_w], _paint(np.ones((band_h, tag_w), np.float32), ORANGE))
    _over(band, _paint(coverage(bold.line("LATEST", size, 0.45 * band_h, baseline, 0.06)[0], band_h, band_w), WHITE))
    crawl_w = band_w - tag_w
    step = max(1, round(CRAWL_SPEED * u / fps))
    strip = _crawl_strip(regular, headlines, size, band_h, baseline, step * count)
    if strip.shape[1] != step * count:   # the headlines outgrow one loop at this speed: go faster
        step = math.ceil(strip.shape[1] / count)
        strip = _crawl_strip(regular, headlines, size, band_h, baseline, step * count)
    columns = np.arange(crawl_w)
    edge = np.clip(np.minimum(columns + 0.5, crawl_w - 0.5 - columns) / (0.75 * band_h), 0, 1)
    fade = (edge ** 2 * (3 - 2 * edge))[None, :, None].astype(np.float32)   # smoothstep at both ends
    # Name plate: accent bar, then the plate slides out from behind it; the text follows.
    plate_h, accent_w, plate_w = round(132 * u), round(22 * u), round(994 * u)
    pad, name_size, role_size = 40 * u, 52 * u, 29 * u
    ramp = np.clip((1 - (np.arange(plate_w) + 0.5) / plate_w) / 0.3, 0, 1)
    plate = _paint(np.broadcast_to(ramp, (plate_h, plate_w)), NAVY, 0xEE / 255)
    texts = [(_paint(coverage(bold.line(name, name_size, pad, 0.5 * plate_h)[0], plate_h, plate_w), WHITE),
              _paint(coverage(regular.line(role, role_size, pad, 0.8 * plate_h)[0], plate_h, plate_w), WHITE, 0.85))
             for name, role in people]
    plate_top = bottom - band_h - round(18 * u) - plate_h

    def draw_band(n):
        layer = band.copy()
        _over(layer[:, tag_w:], strip[:, (columns + n * step) % strip.shape[1]] * fade)
        return layer

    def draw_plate(n):
        layer = np.zeros((plate_h, accent_w + plate_w, 4), np.float32)
        t, out = (n % cycle) / fps, (cycle - 0.8 * fps) / fps   # seconds into this person, and when they go
        bar = _phase(t, 0, 0.3, SETTLE) * (1 - _phase(t, out + 0.5, 0.2, EASE_IN))
        slide = _phase(t, 0.15, 0.6, SETTLE) * (1 - _phase(t, out + 0.15, 0.4, EASE_IN))
        shown = 1 - _phase(t, out, 0.2)
        x = accent_w - round(1.04 * plate_w * (1 - slide))
        _blit(layer, plate, x, slide, accent_w)
        for sprite, start in zip(texts[n // cycle % len(texts)], (0.35, 0.45)):
            entered = _phase(t, start, 0.5, SETTLE)
            _blit(layer, sprite, x + round(0.4 * name_size * (1 - entered)), slide * entered * shown, accent_w)
        if bar:
            half = plate_h * bar / 2
            _over(layer[:, :accent_w], _paint(_box(plate_h, accent_w, 0, plate_h / 2 - half, accent_w, plate_h / 2 + half),
                                              ORANGE))
        return layer

    return [(left, bottom - band_h, draw_band), (left, plate_top, draw_plate)]


def _bugs(bold, width, height, fps, count):
    """Top-left station bug (AVP on a navy disc inside a turning orange ring, a turn about every
    4 s) and top-right LIVE pill with a pulsing dot (once a second), level with each other."""
    u, left, top, right = min(width, height) / 1080, round(0.05 * width), round(0.05 * height), round(0.95 * width)
    side = round(152 * u)
    y, x = np.mgrid[0:side, 0:side] + 0.5
    centre = side / 2
    radius = np.hypot(x - centre, y - centre)
    angle = np.arctan2(x - centre, centre - y) / (2 * np.pi) % 1   # clockwise from twelve o'clock
    ring = np.clip(np.minimum(0.44 * side + 0.5 - radius, radius - 0.376 * side + 0.5), 0, 1)
    turns = max(1, round(count / fps / 4))
    mark = _paint(_disc(side, side, centre, centre, 0.31 * side), NAVY, 0xCC / 255)
    size = 40 * u
    text_w = bold.width("AVP", size)
    _over(mark, _paint(coverage(bold.line("AVP", size, centre - text_w / 2, centre + bold.cap_height / bold.em * size / 2)[0],
                                side, side), WHITE))

    def draw_mark(n):
        phase = (angle - n * turns / count) % 1
        inside = np.where(phase < 0.3, np.minimum(phase, 0.3 - phase), -np.minimum(phase - 0.3, 1 - phase))
        lit = np.clip(inside * 2 * np.pi * radius + 0.5, 0, 1)   # the arc's ends, anti-aliased along the ring
        arc = _paint(ring * lit, ORANGE) + _paint(ring * (1 - lit), WHITE, 0x40 / 255)
        return _over(mark.copy(), arc)

    pill_h = round(70 * u)
    dot = 11 * u
    label, label_size = "LIVE", 30 * u
    pill_w = round(pill_h / 2 + dot + 0.25 * pill_h + bold.width(label, label_size, 0.08) + 0.42 * pill_h)
    pill = _paint(_box(pill_h, pill_w, 0, 0, pill_w, pill_h, pill_h / 2), NAVY, 0xCC / 255)
    _over(pill, _paint(coverage(bold.line(label, label_size, pill_h / 2 + dot + 0.25 * pill_h,
                                          pill_h / 2 + bold.cap_height / bold.em * label_size / 2, 0.08)[0],
                                pill_h, pill_w), WHITE))
    pulses = max(1, round(count / fps))

    def draw_pill(n):
        beat = n * pulses / count % 1
        layer = pill.copy()
        _over(layer[:, :pill_h], _paint(_disc(pill_h, pill_h, pill_h / 2, pill_h / 2, dot * (1 + 1.2 * beat)), RED,
                                        0.5 * (1 - beat) ** 2))
        _over(layer[:, :pill_h], _paint(_disc(pill_h, pill_h, pill_h / 2, pill_h / 2, dot), RED,
                                        0.8 + 0.2 * math.cos(2 * math.pi * beat)))
        return layer

    return [(left, top, draw_mark), (right - pill_w, top + (side - pill_h) // 2, draw_pill)]


def loop(stream, width=1920, height=1080, fps=30, seconds=10, people=PEOPLE, headlines=HEADLINES):
    """(frames in the loop, frame(n) -> straight BGRA array height x width x 4) of *stream*; frame
    is periodic in n. *people* ((name, role), ...) and *headlines* vary the lower_third stream."""
    count = Fraction(fps) * Fraction(seconds)
    if count.denominator != 1 or count < 1:
        raise ValueError(f"{seconds} s at {fps} fps is not a whole number of frames")
    count = int(count)
    regular, bold = Font(FONTS / "NotoSans-Regular.ttf"), Font(FONTS / "NotoSans-Bold.ttf")
    if stream == "lower_third":
        parts = _lower_third(regular, bold, width, height, float(fps), count, tuple(people), tuple(headlines))
    elif stream == "bugs":
        parts = _bugs(bold, width, height, float(fps), count)
    else:
        raise ValueError(f"stream must be one of {', '.join(STREAMS)}")

    def frame(n):
        picture = np.zeros((height, width, 4), np.uint8)
        for left, top, draw in parts:
            layer = draw(n % count)
            picture[top:top + layer.shape[0], left:left + layer.shape[1]] = _straight_bgra(layer)
        return picture

    return count, frame


def render(path, stream="lower_third", width=1920, height=1080, fps=30, seconds=10, people=PEOPLE,
           headlines=HEADLINES):
    """Write *stream*'s loop to *path*: frames of width * height * 4 bytes (straight-alpha BGRA, top
    row first), back to back. Returns the number of frames."""
    count, frame = loop(stream, width, height, fps, seconds, people, headlines)
    with open(path, "wb") as out:
        for n in range(count):
            out.write(frame(n))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--stream", choices=STREAMS, default="lower_third")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=Fraction, default=Fraction(30))
    parser.add_argument("--seconds", type=Fraction, default=Fraction(10))
    args = parser.parse_args()
    count = render(args.path, args.stream, args.width, args.height, args.fps, args.seconds)
    print(f"{args.path}: {count} frames of {args.width}x{args.height} BGRA")


if __name__ == "__main__":
    main()
