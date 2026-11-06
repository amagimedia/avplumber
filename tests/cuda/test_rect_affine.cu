// NVIDIA integration test of the rect kernel's affine entries (sub-pixel crops, rotation, soft
// edges). Every sample an affine layer draws is compared with a CPU reference of the same formulas
// (map, taps, filter, coverage, blend); quarter turns at 1:1 must be exact copies; tables without
// an affine layer must draw on the affine entries exactly as on the filter entry; linear and
// CUarray sources must agree byte for byte. Prints a timing table.
//   nvcc -std=c++17 -O2 -I/usr/local/include tests/cuda/test_rect_affine.cu -lcuda -o /tmp/test_rect_affine
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "../../src/nodes/hwaccel/cuda_rect_scale.cu"
#include "../../src/nodes/hwaccel/cuda_rect_sampler.h"
#include "../../src/mixer/primitives/compositor_affine.hpp"
#include "rect_test_harness.cuh"

using avp::mixer::AffinePlane;
using avp::mixer::LayerTransform;
using avp::mixer::Rect;
using avp::mixer::RectF;

// An affine layer in luma pixels: an exact crop drawn into a box rotated about its centre.
struct AffineDraw {
    const Frame *source;
    RectF crop;
    Rect box;
    double deg = 0;
    int filter = AVP_RECT_FILTER_BILINEAR;
    float param = 0.f;
};

static int kindFor(const Frame &source, const Format &canvas) {
    return &source.format == &canvas ? AVP_RECT_KIND_YUV
         : source.format.depth > canvas.depth ? AVP_RECT_KIND_DEMOTE : AVP_RECT_KIND_PROMOTE;
}

// The table entry CudaRectDraw::fillTableEntry builds for a transformed YUV layer.
static AvpRectLayer affineEntry(const AffineDraw &d, const Format &canvas, int cw, int ch, bool arrays) {
    const Frame &source = *d.source;
    LayerTransform t;
    t.crop = d.crop;
    const double x0 = std::max(d.crop.x, 0.), y0 = std::max(d.crop.y, 0.);
    t.edges = {x0, y0, std::min(d.crop.x + d.crop.w, double(source.width)) - x0,
               std::min(d.crop.y + d.crop.h, double(source.height)) - y0};
    t.box = d.box;
    t.rotation_deg = avp::mixer::normalizedRotation(d.deg);
    t.pivot_x = d.box.x + d.box.w / 2.;
    t.pivot_y = d.box.y + d.box.h / 2.;
    t.clip = {0, 0, cw, ch};
    AvpRectLayer out{};
    out.kind = kindFor(source, canvas);
    out.yuv_texture = arrays;
    out.src_bytes = source.format.bytes;
    out.src_shift = source.format.shift;
    out.mul = std::ldexp(1.f, canvas.depth - source.format.depth);
    out.filter = d.filter;
    out.filter_param = d.param;
    out.affine = 1;
    for (int p = 0; p < 2; ++p) {
        const auto plane = avp::mixer::affinePlane(t, p, p ? canvas.sub_y : 0, p, p ? source.format.sub_y : 0);
        if (!plane) throw std::runtime_error("affine test layer has no bounds");
        out.src[p] = arrays ? source.texture[p] : source.linear[p];
        out.src_pitch[p] = arrays ? 0 : source.pitch;
        out.sx[p] = plane->taps.x; out.sy[p] = plane->taps.y; out.sw[p] = plane->taps.w; out.sh[p] = plane->taps.h;
        out.dx[p] = plane->bounds.x; out.dy[p] = plane->bounds.y; out.dw[p] = plane->bounds.w; out.dh[p] = plane->bounds.h;
        std::copy(plane->m, plane->m + 6, out.m[p]);
        std::copy(plane->edge, plane->edge + 4, out.edge[p]);
        std::copy(plane->aa, plane->aa + 2, out.aa[p]);
        std::copy(plane->footprint, plane->footprint + 2, out.footprint[p]);
    }
    return out;
}

static float saturate(float v) { return std::fmin(std::fmax(v, 0.f), 1.f); }

// The kernel's map and coverage, in the same float operations.
static float coverage(const AvpRectLayer &L, int p, int X, int Y, float &u, float &v) {
    const float *m = L.m[p], *e = L.edge[p];
    const float x = X + 0.5f, y = Y + 0.5f;
    u = std::fmaf(m[0], x, std::fmaf(m[1], y, m[2]));
    v = std::fmaf(m[3], x, std::fmaf(m[4], y, m[5]));
    const float cu = saturate(std::fmaf(std::fmin(u - e[0], e[2] - u), L.aa[p][0], 0.5f));
    const float cv = saturate(std::fmaf(std::fmin(v - e[1], e[3] - v), L.aa[p][1], 0.5f));
    return cu * cv;
}

// The unrounded value an affine YUV layer gives lane `c` of (X, Y) over `below`; false where it
// leaves the sample untouched.
static bool affineReference(const AvpRectLayer &L, const Frame &source, const Format &canvas, int p, int X, int Y, int c,
                            double below, double &value, float *cov_out = nullptr) {
    if (X < L.dx[p] || Y < L.dy[p] || X >= L.dx[p] + L.dw[p] || Y >= L.dy[p] + L.dh[p]) return false;
    float u, v;
    const float cov = coverage(L, p, X, Y, u, v);
    if (cov_out) *cov_out = cov;
    if (!(cov > 0.f)) return false;
    const PlaneView view{source, p, L.sx[p], L.sy[p], L.sw[p], L.sh[p]};
    const float fx = u - 0.5f - L.sx[p], fy = v - 0.5f - L.sy[p];
    switch (L.filter) {
    case AVP_RECT_FILTER_BICUBIC: value = bicubic(view, fx, fy, L.filter_param, c); break;
    case AVP_RECT_FILTER_BICUBIC_A0: value = bicubic(view, fx, fy, 0., c); break;
    case AVP_RECT_FILTER_MULTISAMPLE4: value = multisample(view, fx, fy, L.footprint[p][0], L.footprint[p][1], 4, c); break;
    case AVP_RECT_FILTER_MULTISAMPLE8: value = multisample(view, fx, fy, L.footprint[p][0], L.footprint[p][1], 8, c); break;
    default: value = bilinear(view, fx, fy, c);
    }
    if (L.kind != AVP_RECT_KIND_YUV) value *= L.mul;
    value = std::min(std::max(value, 0.), double((256 << (canvas.depth - 8)) - 1));
    if (cov < 1.f) value = cov * value + (1. - cov) * below;
    return true;
}

struct Tally { size_t samples = 0, edge = 0, off_tie = 0; double worst = 0; };

// `out` holds `layer` drawn over `below` (or over the clear values when null), sample by sample.
static void compareAffine(const Canvas &out, const Canvas *below, const AvpRectLayer &layer, const Frame &source,
                          const std::string &what, Tally &tally, bool exact = false) {
    for (int p = 0; p < 2; ++p)
        for (int y = 0; y < (p ? out.uv_rows : out.height); ++y)
            for (int x = 0; x < (p ? out.width / 2 : out.width); ++x)
                for (int c = 0; c < (p ? 2 : 1); ++c) {
                    const int got = out.code(p, x, y, c), under = below ? below->code(p, x, y, c) : out.clearCode(p);
                    double want;
                    float cov = 0.f;
                    const std::string at = what + ": plane " + std::to_string(p) + " " + std::to_string(x) + "," +
                                           std::to_string(y) + " lane " + std::to_string(c);
                    if (!affineReference(layer, source, out.format, p, x, y, c, under, want, &cov)) {
                        if (got != under) throw std::runtime_error(at + " changed outside the quad");
                        continue;
                    }
                    ++tally.samples;
                    if (cov < 1.f) ++tally.edge;
                    const double distance = std::abs(got - want);
                    tally.worst = std::max(tally.worst, distance);
                    if (exact ? got != int(want + 0.5) : distance > 0.5 + kTieSlack)
                        throw std::runtime_error(at + " is " + std::to_string(got) + ", reference " + std::to_string(want));
                    if (got != int(want + 0.5)) ++tally.off_tie;
                }
}

struct FilterCase { const char *name; int code; float param; };
static const FilterCase kFilters[] = {
    {"bilinear", AVP_RECT_FILTER_BILINEAR, 0.f},
    {"bicubic A=0 2x2", AVP_RECT_FILTER_BICUBIC_A0, 0.f},
    {"bicubic A=-0.5", AVP_RECT_FILTER_BICUBIC, -0.5f},
    {"multisample 4", AVP_RECT_FILTER_MULTISAMPLE4, 0.f},
    {"multisample 8", AVP_RECT_FILTER_MULTISAMPLE8, 0.f},
};

static Kernel affineKernel(bool arrays) { return arrays ? composite_planes_affine : composite_planes_yuv_affine; }

// Sources 240x144 on a 512x288 canvas.
static void arbitrary_angles_match_the_reference(const Frame &source, const Format &canvas_format) {
    struct Case { const char *name; RectF crop; Rect box; double deg; };
    const Case cases[] = {
        {"17 deg 1:1", {0, 0, 240, 144}, {136, 72, 240, 144}, 17},
        {"33.3 deg 0.6x", {0, 0, 240, 144}, {184, 100, 144, 86}, 33.3},
        {"-71 deg 1.7x", {0, 0, 240, 144}, {52, 22, 408, 245}, -71},
        {"12 deg crop, half off the left", {30.25, 10.5, 160, 96}, {-100, 60, 320, 192}, 12},
    };
    for (const FilterCase &f : kFilters) {
        Tally tally;
        for (const Case &k : cases)
            for (bool arrays : {false, true}) {
                const AffineDraw d{&source, k.crop, k.box, k.deg, f.code, f.param};
                const AvpRectLayer layer = affineEntry(d, canvas_format, 512, 288, arrays);
                Canvas out(canvas_format, 512, 288);
                out.draw(affineKernel(arrays), {layer});
                compareAffine(out, nullptr, layer, source, std::string(source.format.name) + " -> " + canvas_format.name +
                              ", " + f.name + ", " + k.name, tally);
                if (!arrays) {   // linear_and_arrays_identical
                    Canvas again(canvas_format, 512, 288);
                    again.draw(composite_planes_affine, {affineEntry(d, canvas_format, 512, 288, true)});
                    if (again.pixels != out.pixels) throw std::runtime_error(std::string(k.name) + ": arrays and linear differ");
                }
            }
        std::printf("PASS arbitrary angles %s -> %s, %-16s %zu samples (%zu on the soft edge), worst %.4f codes, "
                    "%zu across a tie\n", source.format.name, canvas_format.name, f.name, tally.samples, tally.edge,
                    tally.worst, tally.off_tie);
    }
}

static void quarter_turns_are_exact_copies(const Frame &source, const Format &canvas_format) {
    const RectF crop{40, 8, 128, 128};
    const Rect box{180, 80, 128, 128};
    for (double deg : {90., 180., 270.}) {
        const AvpRectLayer layer = affineEntry({&source, crop, box, deg}, canvas_format, 512, 288, false);
        Canvas out(canvas_format, 512, 288);
        out.draw(composite_planes_yuv_affine, {layer});
        Tally tally;
        // 4:2:2 chroma under a quarter turn is resampled, not copied: only its luma is exact.
        const bool exact = source.format.sub_y == canvas_format.sub_y && source.format.sub_y == 1;
        compareAffine(out, nullptr, layer, source, "quarter turn " + std::to_string(int(deg)), tally, exact);
        if (tally.edge) throw std::runtime_error("a 1:1 quarter turn has soft-edge samples");
        if (deg == 90.)   // clockwise: destination (X, Y) is source (col Y - by, row 127 - (X - bx))
            for (int y = 0; y < 128; ++y)
                for (int x = 0; x < 128; ++x)
                    if (out.code(0, box.x + x, box.y + y, 0) !=
                        std::min(int(source.code(0, int(crop.x) + y, int(crop.y) + 127 - x, 0) * layer.mul + 0.5f),
                                 (256 << (canvas_format.depth - 8)) - 1))
                        throw std::runtime_error("90 deg is not the clockwise transpose");
    }
    std::printf("PASS quarter turns %s -> %s: exact%s\n", source.format.name, canvas_format.name,
                source.format.sub_y == 1 && canvas_format.sub_y == 1 ? " copies, luma and chroma" : " luma, chroma by reference");
}

static void subpixel_offset_is_bilinear_of_the_shift(const Frame &source, const Format &canvas_format) {
    const AvpRectLayer layer = affineEntry({&source, {100.25, 20, 120, 100}, {200, 40, 120, 100}}, canvas_format, 512, 288, false);
    Canvas out(canvas_format, 512, 288);
    out.draw(composite_planes_yuv_affine, {layer});
    Tally tally;
    compareAffine(out, nullptr, layer, source, "x + 0.25 at 1:1", tally);
    const PlaneView view{source, 0, layer.sx[0], layer.sy[0], layer.sw[0], layer.sh[0]};
    for (int x = 0; x < 120; ++x) {   // luma row 50: bilinear at fx = X + 100.25 - 0.5 within the taps
        const double want = layer.mul * bilinear(view, float(x + 200 + 0.5f) - 200 + 100.25f - 0.5f - layer.sx[0], 50 + 20 - layer.sy[0], 0);
        if (std::abs(out.code(0, 200 + x, 90, 0) - want) > 0.5 + kTieSlack)
            throw std::runtime_error("sub-pixel offset is not the bilinear of the shift");
    }
    std::printf("PASS sub-pixel offset %s -> %s, worst %.4f\n", source.format.name, canvas_format.name, tally.worst);
}

static void outside_the_quad_is_untouched(const Frame &source, const Format &canvas_format) {
    const Draw base{&source, 0, 0, 240, 144, 0, 0, 512, 288};
    const AvpRectLayer under = entry(base, canvas_format, false);
    const AvpRectLayer layer = affineEntry({&source, {0, 0, 240, 144}, {136, 72, 240, 144}, 23}, canvas_format, 512, 288, false);
    Canvas below(canvas_format, 512, 288), out(canvas_format, 512, 288);
    below.draw(composite_planes_yuv_filter, {under});
    out.draw(composite_planes_yuv_affine, {under, layer});
    Tally tally;
    compareAffine(out, &below, layer, source, "23 deg over a full-canvas layer", tally);
    std::printf("PASS over a layer %s -> %s: %zu soft-edge samples blend with the layer below\n", source.format.name,
                canvas_format.name, tally.edge);
}

static void no_reads_outside_the_crop(std::mt19937 &rng, const Format &format) {
    Frame a(format, 240, 144, rng), b(format, 240, 144, rng);
    // b keeps a's pixels inside the crop (40..200, 30..114) and 16 px of noise around it differs.
    for (int p = 0; p < 2; ++p)
        for (int y = 0; y < a.rows(p); ++y)
            for (int x = 0; x < 240; ++x) {
                const int ly = p ? y << format.sub_y : y, lx = p ? x & ~1 : x;
                if (lx >= 40 && lx < 200 && ly >= 30 && ly < 114)
                    std::memcpy(&b.host[p][size_t(y) * b.pitch + x * format.bytes], &a.host[p][size_t(y) * a.pitch + x * format.bytes],
                                format.bytes);
            }
    for (int p = 0; p < 2; ++p) check(cuMemcpyHtoD(b.linear[p], b.host[p].data(), b.host[p].size()), "upload");
    const auto draw = [&](const Frame &f) {
        Canvas out(format, 512, 288);
        out.draw(composite_planes_yuv_affine, {affineEntry({&f, {40, 30, 160, 84}, {120, 60, 300, 158}, 37, AVP_RECT_FILTER_BICUBIC, -0.5f},
                                                           format, 512, 288, false)});
        return out.pixels;
    };
    if (draw(a) != draw(b)) throw std::runtime_error("the affine draw read outside its crop");
    std::printf("PASS %s: a rotated bicubic layer reads only its crop\n", format.name);
}

static void affine_entry_draws_plain_tables_like_the_filter_entry(const Frame &left, const Frame &right, const Format &canvas_format,
                                                                  CUdeviceptr key_pixels) {
    auto key = [&](float opacity) {
        AvpRectLayer out{};
        out.kind = AVP_RECT_KIND_RGBA;
        out.src[0] = key_pixels;
        out.src_pitch[0] = 32 * 4;
        out.step = 4;
        out.r_off = 0; out.g_off = 1; out.b_off = 2; out.a_off = 3;
        out.sw[0] = out.sh[0] = 32;
        out.mul = opacity;
        for (int p = 0; p < 2; ++p) {
            out.dx[p] = 300 >> p; out.dy[p] = 40 >> (p ? canvas_format.sub_y : 0);
            out.dw[p] = 120 >> p; out.dh[p] = 100 >> (p ? canvas_format.sub_y : 0);
        }
        return out;
    };
    for (const FilterCase &f : kFilters)
        for (bool arrays : {false, true}) {
            Draw l{&left, 0, 0, 240, 144, 0, 0, 256, 288, f.code, f.param}, r{&right, 36, 20, 180, 108, 256, 0, 256, 200};
            auto layers = table({l, r}, canvas_format, arrays);
            if (!arrays) {
                Canvas lean(canvas_format, 512, 288), full(canvas_format, 512, 288);
                lean.draw(composite_planes_yuv_affine, layers);
                full.draw(composite_planes_filter, layers);
                if (lean.pixels != full.pixels) throw std::runtime_error(std::string(f.name) + ": lean affine entry differs on a plain table");
            }
            layers.push_back(key(1.f));
            layers.push_back(key(0.4f));
            Canvas affine(canvas_format, 512, 288), filter(canvas_format, 512, 288);
            affine.draw(composite_planes_affine, layers);
            filter.draw(composite_planes_filter, layers);
            if (affine.pixels != filter.pixels) throw std::runtime_error(std::string(f.name) + ": affine entry differs on a plain table");
        }
    std::printf("PASS %s and %s -> %s: plain tables (every filter, keys, a faded key, arrays) draw on the affine entries "
                "as on the filter entry\n", left.format.name, right.format.name, canvas_format.name);
}

static void identity_affine_matches_plain_entries(const Frame &source, const Format &canvas_format) {
    struct Case { int cx, cy, cw, ch, dx, dy, dw, dh; };
    const Case cases[] = {{0, 0, 240, 144, 16, 0, 240, 144}, {0, 0, 240, 144, 16, 0, 480, 288},
                          {0, 0, 240, 144, 100, 40, 80, 48}, {0, 0, 240, 144, 20, 20, 360, 216}};
    size_t samples = 0, differ = 0;
    int worst = 0;
    for (const FilterCase &f : kFilters)
        for (const Case &k : cases) {
            const AvpRectLayer plain = entry({&source, k.cx, k.cy, k.cw, k.ch, k.dx, k.dy, k.dw, k.dh, f.code, f.param}, canvas_format, false);
            const AvpRectLayer affine = affineEntry({&source, {double(k.cx), double(k.cy), double(k.cw), double(k.ch)},
                                                     {k.dx, k.dy, k.dw, k.dh}, 0, f.code, f.param}, canvas_format, 512, 288, false);
            Canvas a(canvas_format, 512, 288), b(canvas_format, 512, 288);
            a.draw(composite_planes_filter, {plain});
            b.draw(composite_planes_yuv_affine, {affine});
            for (size_t i = 0; i < a.pixels.size(); i += canvas_format.bytes) {
                const int x = canvas_format.bytes == 1 ? a.pixels[i] : (a.pixels[i] | a.pixels[i + 1] << 8) >> canvas_format.shift;
                const int y = canvas_format.bytes == 1 ? b.pixels[i] : (b.pixels[i] | b.pixels[i + 1] << 8) >> canvas_format.shift;
                ++samples;
                if (x != y) { ++differ; worst = std::max(worst, std::abs(x - y)); }
            }
        }
    // The plain path computes (o + 0.5) * sn / dn - 0.5, the affine one a fused multiply-add: on noise
    // a last-bit difference crosses a rounding tie about once in 500 samples, by one code.
    if (worst > 1 || differ * 200 > samples)
        throw std::runtime_error("identity affine differs from the plain entries: " + std::to_string(differ) + " of " +
                                 std::to_string(samples) + ", worst " + std::to_string(worst));
    std::printf("PASS identity affine %s -> %s vs plain entries: %zu of %zu samples differ, by at most %d code\n",
                source.format.name, canvas_format.name, differ, samples, worst);
}

// An RGBA key rotated over a YUV layer: the alpha times the coverage, straight and premultiplied.
static void rotated_rgba_key_matches_reference(std::mt19937 &rng, const Frame &source, const Format &canvas_format) {
    const int kw = 96, kh = 64;
    std::vector<unsigned char> rgba(size_t(kw) * kh * 4);
    for (auto &value : rgba) value = static_cast<unsigned char>(rng());
    CUdeviceptr pixels = 0;
    check(cuMemAlloc(&pixels, rgba.size()), "allocate key");
    check(cuMemcpyHtoD(pixels, rgba.data(), rgba.size()), "upload key");
    const Draw base{&source, 0, 0, 240, 144, 0, 0, 512, 288};
    const AvpRectLayer under = entry(base, canvas_format, false);
    Canvas below(canvas_format, 512, 288);
    below.draw(composite_planes_filter, {under});
    const int dst_scale = 1 << (canvas_format.depth - 8);
    const float maxv = 256.f * dst_scale - 1.f;
    for (int premultiplied : {0, 1})
        for (float opacity : {1.f, 0.4f}) {
            LayerTransform t;
            t.crop = t.edges = {0, 0, double(kw), double(kh)};
            t.box = {150, 60, 192, 128};
            t.rotation_deg = 20;
            t.pivot_x = 246; t.pivot_y = 124;
            t.clip = {0, 0, 512, 288};
            const AffinePlane plane = *avp::mixer::affinePlane(t, 0, 0, 0, 0);
            AvpRectLayer key{};
            key.kind = AVP_RECT_KIND_RGBA;
            key.src[0] = pixels; key.src_pitch[0] = kw * 4;
            key.step = 4; key.r_off = 0; key.g_off = 1; key.b_off = 2; key.a_off = 3;
            key.premultiplied = premultiplied;
            key.mul = opacity;
            key.affine = 1;
            key.sx[0] = plane.taps.x; key.sy[0] = plane.taps.y; key.sw[0] = plane.taps.w; key.sh[0] = plane.taps.h;
            key.dx[0] = plane.bounds.x; key.dy[0] = plane.bounds.y; key.dw[0] = plane.bounds.w; key.dh[0] = plane.bounds.h;
            key.dx[1] = plane.bounds.x >> 1; key.dy[1] = plane.bounds.y >> canvas_format.sub_y;
            key.dw[1] = (plane.bounds.x + plane.bounds.w + 1) / 2 - key.dx[1];
            key.dh[1] = (plane.bounds.y + plane.bounds.h + (1 << canvas_format.sub_y) - 1) / (1 << canvas_format.sub_y) - key.dy[1];
            std::copy(plane.m, plane.m + 6, key.m[0]);
            std::copy(plane.edge, plane.edge + 4, key.edge[0]);
            std::copy(plane.aa, plane.aa + 2, key.aa[0]);
            Canvas out(canvas_format, 512, 288);
            out.draw(composite_planes_affine, {under, key});
            // Reference: per luma position, coverage x alpha; chroma averages its block's positions.
            auto texel = [&](int x, int y, int c) { return float(rgba[(size_t(std::clamp(y, 0, kh - 1)) * kw + std::clamp(x, 0, kw - 1)) * 4 + c]); };
            auto sample = [&](float fx, float fy, float *rgb, float &a) {
                const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
                const float tx = fx - ix, ty = fy - iy;
                float v[4];
                for (int c = 0; c < 4; ++c) {
                    const float top = texel(ix, iy, c) + tx * (texel(ix + 1, iy, c) - texel(ix, iy, c));
                    const float bottom = texel(ix, iy + 1, c) + tx * (texel(ix + 1, iy + 1, c) - texel(ix, iy + 1, c));
                    v[c] = top + ty * (bottom - top);
                }
                a = v[3] / 255.f;
                for (int c = 0; c < 3; ++c) rgb[c] = premultiplied && a > 0 ? std::min(v[c] / a, 255.f) : v[c];
                convert_graphic_rgb(rgb, 0, 100.f, 1000.f);
            };
            auto weight = [&](int lx, int ly, float *rgb) {
                float u, v;
                const float cov = coverage(key, 0, lx, ly, u, v);
                if (!(cov > 0.f)) return 0.f;
                float a;
                sample(u - 0.5f - key.sx[0], v - 0.5f - key.sy[0], rgb, a);
                return a > 0.f ? cov * a : 0.f;
            };
            for (int y = 0; y < 288; ++y)
                for (int x = 0; x < 512; ++x) {
                    float rgb[3];
                    const float w = weight(x, y, rgb) * opacity;
                    const double want = w > 0 ? std::min(std::max(double(w * graphic_luma(rgb, 0) * dst_scale + (1 - w) * below.code(0, x, y, 0)), 0.), double(maxv))
                                              : below.code(0, x, y, 0);
                    if (std::abs(out.code(0, x, y, 0) - want) > 0.5 + 0.05)
                        throw std::runtime_error("rotated key luma at " + std::to_string(x) + "," + std::to_string(y));
                }
            const int bh = 1 << canvas_format.sub_y;
            for (int y = 0; y < out.uv_rows; ++y)
                for (int x = 0; x < 256; ++x) {
                    float sum[3] = {0, 0, 0}, sum_a = 0;
                    int cnt = 0;
                    for (int j = 0; j < bh; ++j)
                        for (int i = 0; i < 2; ++i) {
                            ++cnt;
                            float rgb[3];
                            const float w = weight(2 * x + i, y * bh + j, rgb);
                            for (int c = 0; c < 3; ++c) sum[c] += w * rgb[c];
                            sum_a += w;
                        }
                    for (int c = 0; c < 2; ++c) {
                        double want = below.code(1, x, y, c);
                        if (sum_a > 0) {
                            float cb, cr;
                            graphic_chroma(sum[0] / sum_a, sum[1] / sum_a, sum[2] / sum_a, 0, cb, cr);
                            const float a = sum_a / cnt * opacity;
                            want = std::min(std::max(double(a * (c ? cr : cb) * dst_scale) + (1 - a) * want, 0.), double(maxv));
                        }
                        if (std::abs(out.code(1, x, y, c) - want) > 0.5 + 0.05)
                            throw std::runtime_error("rotated key chroma at " + std::to_string(x) + "," + std::to_string(y));
                    }
                }
        }
    cuMemFree(pixels);
    std::printf("PASS rotated RGBA key over %s: coverage x alpha, straight and premultiplied, opacity 1 and 0.4\n", canvas_format.name);
}

static void timeLayers(const char *label, const Format &format, int cw, int ch, Kernel kernel, const std::vector<AvpRectLayer> &layers) {
    Canvas canvas(format, cw, ch);
    canvas.upload(layers);
    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "event");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "event");
    for (int i = 0; i < 20; ++i) canvas.launch(kernel, int(layers.size()));
    check(cuCtxSynchronize(), "warm up");
    std::vector<float> ms;
    for (int i = 0; i < 200; ++i) {
        check(cuEventRecord(start, nullptr), "record");
        canvas.launch(kernel, int(layers.size()));
        check(cuEventRecord(stop, nullptr), "record");
        check(cuEventSynchronize(stop), "wait");
        float elapsed = 0;
        check(cuEventElapsedTime(&elapsed, start, stop), "elapsed");
        ms.push_back(elapsed);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-58s median %.3f ms  p90 %.3f ms\n", label, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
    cuEventDestroy(start);
    cuEventDestroy(stop);
}

static void timings(std::mt19937 &rng) {
    Frame cam(NV12, 1920, 1080, rng);
    const Draw full{&cam, 0, 0, 1920, 1080, 0, 0, 1920, 1080};
    timeLayers("1920x1080 full canvas, plain bilinear (yuv)", NV12, 1920, 1080, composite_planes_yuv, {entry(full, NV12, false)});
    timeLayers("1920x1080 full canvas, 15 deg bilinear (yuv_affine)", NV12, 1920, 1080, composite_planes_yuv_affine,
               {affineEntry({&cam, {0, 0, 1920, 1080}, {0, 0, 1920, 1080}, 15}, NV12, 1920, 1080, false)});
    const Draw pip{&cam, 0, 0, 1920, 1080, 1200, 640, 640, 360};
    timeLayers("PiP 640x360 over full, plain bilinear (yuv)", NV12, 1920, 1080, composite_planes_yuv,
               {entry(full, NV12, false), entry(pip, NV12, false)});
    timeLayers("PiP 640x360 over full, 15 deg bilinear (yuv_affine)", NV12, 1920, 1080, composite_planes_yuv_affine,
               {entry(full, NV12, false), affineEntry({&cam, {0, 0, 1920, 1080}, {1200, 640, 640, 360}, 15}, NV12, 1920, 1080, false)});
    timeLayers("zoom 1.37x, integer crop, bicubic (filter)", NV12, 1920, 1080, composite_planes_filter,
               {entry({&cam, 280, 158, 1402, 788, 0, 0, 1920, 1080, AVP_RECT_FILTER_BICUBIC, -0.5f}, NV12, false)});
    timeLayers("zoom 1.37x, crop +0.37 px, bicubic (yuv_affine)", NV12, 1920, 1080, composite_planes_yuv_affine,
               {affineEntry({&cam, {280.37, 158.21, 1401.6, 788.4}, {0, 0, 1920, 1080}, 0, AVP_RECT_FILTER_BICUBIC, -0.5f},
                            NV12, 1920, 1080, false)});
}

int main(int argc, char **argv) {
    const bool timing_only = argc > 1 && std::string(argv[1]) == "--timings";
    CUcontext context = nullptr;
    try {
        check(cuInit(0), "initialize CUDA");
        check(cuDevicePrimaryCtxRetain(&context, 0), "retain primary context");
        check(cuCtxSetCurrent(context), "set context");
        std::mt19937 rng(20261106);
        if (!timing_only) {
            CUdeviceptr key = 0;
            std::vector<unsigned char> rgba(32 * 32 * 4);
            for (auto &value : rgba) value = static_cast<unsigned char>(rng());
            check(cuMemAlloc(&key, rgba.size()), "allocate RGBA key");
            check(cuMemcpyHtoD(key, rgba.data(), rgba.size()), "upload RGBA key");
            {
                Frame nv12(NV12, 240, 144, rng), p010(P010, 240, 144, rng), p210(P210, 240, 144, rng);
                const std::pair<const Frame *, const Format *> pairs[] = {
                    {&nv12, &NV12}, {&p010, &P010}, {&p210, &P210}, {&nv12, &P010}, {&p010, &NV12}, {&p210, &NV12}};
                for (const auto &[source, canvas] : pairs) {
                    arbitrary_angles_match_the_reference(*source, *canvas);
                    quarter_turns_are_exact_copies(*source, *canvas);
                    subpixel_offset_is_bilinear_of_the_shift(*source, *canvas);
                    outside_the_quad_is_untouched(*source, *canvas);
                    identity_affine_matches_plain_entries(*source, *canvas);
                }
                for (const Format *format : {&NV12, &P010, &P210}) no_reads_outside_the_crop(rng, *format);
                affine_entry_draws_plain_tables_like_the_filter_entry(nv12, nv12, NV12, key);
                affine_entry_draws_plain_tables_like_the_filter_entry(p010, p010, P010, key);
                affine_entry_draws_plain_tables_like_the_filter_entry(p210, p210, P210, key);
                rotated_rgba_key_matches_reference(rng, nv12, NV12);
                rotated_rgba_key_matches_reference(rng, p210, P210);
            }
            check(cuMemFree(key), "release RGBA key");
        }
        timings(rng);
        check(cuDevicePrimaryCtxRelease(0), "release context");
        std::cout << (timing_only ? "TIMINGS DONE\n" : "ALL PASS\n");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (context) cuDevicePrimaryCtxRelease(0);
        return 1;
    }
}
