#pragma once
// Layer transform geometry: a layer drawn through a float inverse affine per canvas plane, for a
// sub-pixel source crop (a smooth zoom), rotation about a pivot and a soft 1-sample edge. Pure host
// math, no CUDA; the rect table carries the result (cuda_rect_table.h) and the affine kernel entry
// draws it. Rotation is in degrees, clockwise on the y-down canvas.
#include "compositor_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace avp::mixer {

struct RectF {
    double x = 0, y = 0, w = 0, h = 0;
    bool operator==(const RectF &o) const { return x == o.x && y == o.y && w == o.w && h == o.h; }
};

/// One transformed layer, resolved against its source frame and the canvas.
struct LayerTransform {
    RectF crop;                  // exact source rect, storage pixels; the affine maps it onto `box`
    RectF edges;                 // `crop` within the visible picture: the quad test and coverage
    Rect box;                    // placed destination box before rotation, canvas pixels, unclipped
    double rotation_deg = 0;     // clockwise about the pivot, (-180, 180]
    double pivot_x = 0, pivot_y = 0;   // canvas pixels
    Rect clip;                   // the canvas, or the AUX tile the layer is drawn into

    bool operator==(const LayerTransform &o) const {
        auto same = [](const Rect &a, const Rect &b) { return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h; };
        return crop == o.crop && edges == o.edges && same(box, o.box) && rotation_deg == o.rotation_deg &&
               pivot_x == o.pivot_x && pivot_y == o.pivot_y && same(clip, o.clip);
    }
};

/// One canvas plane of a transformed layer, in that plane's units (luma samples, chroma sites).
struct AffinePlane {
    // Destination continuous coordinates (sample X covers [X, X+1), canvas-absolute) to source
    // continuous coordinates (storage-absolute): u = m0 x + m1 y + m2, v = m3 x + m4 y + m5.
    float m[6];
    float edge[4];        // u0, v0, u1, v1 of the edges
    float aa[2];          // destination samples per source unit across a u-edge and a v-edge
    float footprint[2];   // source units per destination sample along u and v
    Rect bounds;          // the rotated box's bounding box grown by 1 sample, clipped to `clip`
    Rect taps;            // the integer source rect the taps clamp to
};

/// `deg` in (-180, 180]; throws for a non-finite value.
inline double normalizedRotation(double deg) {
    if (!std::isfinite(deg)) throw std::invalid_argument("rotation_deg must be a finite number of degrees");
    double r = std::fmod(deg, 360.);
    if (r <= -180.) r += 360.;
    else if (r > 180.) r -= 360.;
    return r == 0. ? 0. : r;
}

/// Sine and cosine of a clockwise rotation, exact at quarter turns.
inline void rotationSinCos(double deg, double &s, double &c) {
    const double r = normalizedRotation(deg);
    if (r == 0.) { s = 0.; c = 1.; }
    else if (r == 90.) { s = 1.; c = 0.; }
    else if (r == 180.) { s = 0.; c = -1.; }
    else if (r == -90.) { s = -1.; c = 0.; }
    else { s = std::sin(r * M_PI / 180.); c = std::cos(r * M_PI / 180.); }
}

/// The plane's geometry for a canvas plane subsampled by (dst_sub_x, dst_sub_y) and a source plane
/// subsampled by (src_sub_x, src_sub_y): the luma map conjugated by S = diag(2^-sub_x, 2^-sub_y)
/// on each side. Empty when the clipped bounds are.
inline std::optional<AffinePlane> affinePlane(const LayerTransform &t, int dst_sub_x, int dst_sub_y,
                                              int src_sub_x, int src_sub_y) {
    if (t.box.w <= 0 || t.box.h <= 0 || !(t.crop.w > 0) || !(t.crop.h > 0)) return {};
    double s, c;
    rotationSinCos(t.rotation_deg, s, c);
    const double sx = t.crop.w / t.box.w, sy = t.crop.h / t.box.h, px = t.pivot_x, py = t.pivot_y;
    // Undo the rotation about the pivot, then the box-to-crop scale.
    const double L[6] = {sx * c, sx * s, t.crop.x + sx * (px - t.box.x - c * px - s * py),
                         -sy * s, sy * c, t.crop.y + sy * (py - t.box.y + s * px - c * py)};
    const double ksx = std::ldexp(1., -src_sub_x), ksy = std::ldexp(1., -src_sub_y);
    const double kdx = std::ldexp(1., -dst_sub_x), kdy = std::ldexp(1., -dst_sub_y);
    AffinePlane p{};
    const double m[6] = {ksx * L[0] / kdx, ksx * L[1] / kdy, ksx * L[2],
                         ksy * L[3] / kdx, ksy * L[4] / kdy, ksy * L[5]};
    for (int i = 0; i < 6; ++i) p.m[i] = float(m[i]);
    const double e[4] = {ksx * t.edges.x, ksy * t.edges.y, ksx * (t.edges.x + t.edges.w), ksy * (t.edges.y + t.edges.h)};
    for (int i = 0; i < 4; ++i) p.edge[i] = float(e[i]);
    const double gu = std::hypot(m[0], m[1]), gv = std::hypot(m[3], m[4]);
    p.footprint[0] = float(gu);
    p.footprint[1] = float(gv);
    p.aa[0] = float(1. / gu);
    p.aa[1] = float(1. / gv);
    const int tx = int(std::floor(e[0])), ty = int(std::floor(e[1]));
    p.taps = {tx, ty, int(std::ceil(e[2])) - tx, int(std::ceil(e[3])) - ty};
    if (p.taps.w <= 0 || p.taps.h <= 0) return {};

    // Bounds in luma: the rotated box's corners, grown by a sample for the soft edge, clipped.
    double x0 = INFINITY, y0 = INFINITY, x1 = -INFINITY, y1 = -INFINITY;
    for (int corner = 0; corner < 4; ++corner) {
        const double qx = t.box.x + (corner & 1 ? t.box.w : 0), qy = t.box.y + (corner & 2 ? t.box.h : 0);
        const double X = px + c * (qx - px) - s * (qy - py), Y = py + s * (qx - px) + c * (qy - py);
        x0 = std::min(x0, X); y0 = std::min(y0, Y); x1 = std::max(x1, X); y1 = std::max(y1, Y);
    }
    const double bx0 = std::max(std::floor(x0) - 1, double(t.clip.x)), by0 = std::max(std::floor(y0) - 1, double(t.clip.y));
    const double bx1 = std::min(std::ceil(x1) + 1, double(t.clip.x) + t.clip.w);
    const double by1 = std::min(std::ceil(y1) + 1, double(t.clip.y) + t.clip.h);
    if (bx1 <= bx0 || by1 <= by0) return {};
    const int px0 = int(std::floor(bx0 * kdx)), py0 = int(std::floor(by0 * kdy));
    p.bounds = {px0, py0, int(std::ceil(bx1 * kdx)) - px0, int(std::ceil(by1 * kdy)) - py0};
    return p;
}

}
