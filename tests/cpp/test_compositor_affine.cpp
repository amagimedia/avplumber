// Layer transform geometry: the inverse affine per plane, its bounds and taps, the parser's
// rotation/pivot/sub-pixel crop, and how resolveDrawOps keeps a transformed layer whole.
#include "mixer/primitives/compositor_affine.hpp"
#include "mixer/primitives/compositor_layers.hpp"
#include "nodes/hwaccel/cuda_rect_entries.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

void print_stack_trace() {}

using namespace avp::mixer;

static bool near(double a, double b, double eps = 1e-4) { return std::fabs(a - b) <= eps; }
static bool sameRect(const Rect &a, const Rect &b) { return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h; }

static LayerTransform transform(RectF crop, Rect box, double deg = 0, Rect clip = {0, 0, 1920, 1080}) {
    LayerTransform t;
    t.crop = t.edges = crop;
    t.box = box;
    t.rotation_deg = deg;
    t.pivot_x = box.x + box.w / 2.;
    t.pivot_y = box.y + box.h / 2.;
    t.clip = clip;
    return t;
}
static AffinePlane luma(const LayerTransform &t) {
    auto p = affinePlane(t, 0, 0, 0, 0);
    assert(p);
    return *p;
}
// Source continuous coordinates of the centre of destination sample (X, Y).
static std::pair<double, double> at(const AffinePlane &p, int X, int Y) {
    return {p.m[0] * (X + .5) + p.m[1] * (Y + .5) + p.m[2], p.m[3] * (X + .5) + p.m[4] * (Y + .5) + p.m[5]};
}
static LayerSpec parse(const char *json, LayerSpec out = {}) {
    parseLayerFromJson(Parameters::parse(json), out);
    return out;
}
static bool throws(const char *json) {
    try { parse(json); } catch (const std::exception &) { return true; }
    return false;
}

static void identity_is_the_unit_matrix() {
    const auto p = luma(transform({0, 0, 1920, 1080}, {0, 0, 1920, 1080}));
    const float unit[6] = {1, 0, 0, 0, 1, 0};
    for (int i = 0; i < 6; ++i) assert(p.m[i] == unit[i]);
    assert(p.edge[0] == 0 && p.edge[1] == 0 && p.edge[2] == 1920 && p.edge[3] == 1080);
    assert(p.aa[0] == 1 && p.aa[1] == 1);
    assert(sameRect(p.bounds, {0, 0, 1920, 1080}));
    assert(sameRect(p.taps, {0, 0, 1920, 1080}));
}

static void subpixel_crop_scales_and_offsets() {
    const auto p = luma(transform({100.5, 50.25, 960, 540}, {0, 0, 1920, 1080}));
    assert(p.m[0] == .5f && p.m[4] == .5f && p.m[1] == 0 && p.m[3] == 0);
    assert(p.m[2] == 100.5f && p.m[5] == 50.25f);
    assert(p.aa[0] == 2 && p.aa[1] == 2);
    assert(p.footprint[0] == .5f && p.footprint[1] == .5f);
    assert(sameRect(p.taps, {100, 50, 961, 541}));
}

static void quarter_turns_are_exact() {
    for (double deg : {90., 180., 270.}) {
        const auto p = luma(transform({0, 0, 512, 512}, {100, 100, 512, 512}, deg));
        for (int i : {0, 1, 3, 4}) assert(p.m[i] == 0 || p.m[i] == 1 || p.m[i] == -1);
        assert(sameRect(p.bounds, {99, 99, 514, 514}));
        if (deg == 90) {
            const auto [u, v] = at(p, 100, 100);
            assert(u == .5 && v == 511.5);   // clockwise: the crop's bottom-left lands top-left
        }
    }
}

static void pivot_default_is_the_box_centre() {
    const Rect box{600, 600, 512, 512}, clip{0, 0, 3000, 3000};
    assert(sameRect(luma(transform({0, 0, 512, 512}, box, 180, clip)).bounds, {599, 599, 514, 514}));
    auto t = transform({0, 0, 512, 512}, box, 180, clip);
    t.pivot_x = 600;
    t.pivot_y = 600;
    assert(sameRect(luma(t).bounds, {87, 87, 514, 514}));
}

static void plane_matrices_conjugate_subsampling() {
    const auto t = transform({0, 0, 512, 512}, {100, 100, 512, 512}, 90);
    const auto y = luma(t);
    const auto nv12 = *affinePlane(t, 1, 1, 1, 1);
    for (int i : {0, 1, 3, 4}) assert(nv12.m[i] == y.m[i]);
    assert(near(nv12.m[2], y.m[2] / 2) && near(nv12.m[5], y.m[5] / 2));
    const auto p210 = *affinePlane(t, 1, 0, 1, 0);
    assert(p210.m[0] == y.m[0] && p210.m[4] == y.m[4]);
    assert(near(p210.m[1], y.m[1] / 2) && near(p210.m[3], y.m[3] * 2));
    assert(near(p210.m[2], y.m[2] / 2) && near(p210.m[5], y.m[5]));
    // A P210 source (chroma rows at full height) on an NV12 canvas (half height): both factors.
    const auto mixed = *affinePlane(t, 1, 1, 1, 0);
    assert(near(mixed.m[3], y.m[3] * 2) && near(mixed.m[4], y.m[4] * 2));
    assert(near(mixed.m[0], y.m[0]) && near(mixed.m[1], y.m[1]));
}

static void bounds_clip_to_the_clip_rect() {
    const Rect box{-200, 300, 800, 450};
    const auto t = transform({0, 0, 1920, 1080}, box, 30);
    const auto p = luma(t);
    assert(p.bounds.x >= 0 && p.bounds.y >= 0 && p.bounds.x + p.bounds.w <= 1920 && p.bounds.y + p.bounds.h <= 1080);
    const double c = std::cos(M_PI / 6), s = std::sin(M_PI / 6), px = t.pivot_x, py = t.pivot_y;
    for (auto [x, y] : {std::pair{box.x, box.y}, {box.x + box.w, box.y}, {box.x, box.y + box.h}, {box.x + box.w, box.y + box.h}}) {
        const double X = px + c * (x - px) - s * (y - py), Y = py + s * (x - px) + c * (y - py);
        if (X < 0 || Y < 0 || X > 1920 || Y > 1080) continue;
        assert(X >= p.bounds.x && Y >= p.bounds.y && X <= p.bounds.x + p.bounds.w && Y <= p.bounds.y + p.bounds.h);
    }
}

static void edges_clamp_but_mapping_does_not() {
    auto t = transform({0, -40, 1920, 1080}, {0, 0, 1920, 1080});
    const auto unclamped = luma(t);
    t.edges = {0, 0, 1920, 1040};
    const auto clamped = luma(t);
    assert(clamped.edge[1] == 0 && clamped.edge[3] == 1040);
    for (int i = 0; i < 6; ++i) assert(clamped.m[i] == unclamped.m[i]);
}

static void needs_transform_only_for_rotation_or_fractions() {
    assert(!needsTransform(parse(R"({"crop":{"x":100,"y":0,"w":10,"h":10}})")));
    assert(!needsTransform(parse(R"({"rotation_deg":360})")));
    assert(normalizedRotation(-180) == 180 && normalizedRotation(540) == 180 && normalizedRotation(-90) == -90);
    assert(needsTransform(parse(R"({"rotation_deg":-180})")));
    assert(needsTransform(parse(R"({"rotation_deg":0.001})")));
    assert(needsTransform(parse(R"({"crop":{"x":100.5,"y":0,"w":10,"h":10}})")));
}

static void parse_follows_the_geometry_rule() {
    const auto rotated = parse(R"({"rotation_deg":30,"pivot_x":10,"pivot_y":20})");
    assert(rotated.rotation_deg == 30 && rotated.pivot_set && rotated.pivot_x == 10 && rotated.pivot_y == 20);
    // metadata_without_rotation_resets_it
    const auto again = parse(R"({"dst_x":4})", rotated);
    assert(again.rotation_deg == 0 && !again.pivot_set);
    assert(throws(R"({"pivot_x":10})"));
    assert(throws(R"({"rotation_deg":"x"})"));
    LayerSpec inf;
    inf.rotation_deg = 0;
    bool threw = false;
    try { normalizedRotation(INFINITY); } catch (const std::exception &) { threw = true; }
    assert(threw);
    const auto exact = parse(R"({"crop":{"x":100.5,"y":2,"w":10,"h":4}})");
    assert(exact.crop_x == 100 && exact.crop_w == 11 && exact.crop_y == 2 && exact.crop_h == 4);
    assert(exact.crop_exact && exact.crop_exact->x == 100.5 && exact.crop_exact->w == 10);
    assert(!parse(R"({"crop":{"x":100,"y":2,"w":10,"h":4}})").crop_exact);
    assert(throws(R"({"crop":{"x":100.5,"y":2}})"));   // a fractional crop needs its size
}

static std::vector<DrawOp> resolve(const char *json, const av::VideoFrame &frame, int w = 1920, int h = 1080) {
    return resolveDrawOps({&frame}, parseLayersArray(Parameters::parse(json)), w, h, AV_PIX_FMT_NV12);
}

static void rotated_layer_keeps_its_crop_whole(const av::VideoFrame &frame) {
    const auto ops = resolve(R"([{"dst_x":-480,"dst_y":0,"dst_w":960,"dst_h":540,"rotation_deg":45}])", frame);
    assert(ops[0].src && ops[0].transform);
    const auto &t = *ops[0].transform;
    assert(t.crop.x == 0 && t.crop.y == 0 && t.crop.w == 1920 && t.crop.h == 1080);
    assert(sameRect(t.box, {-480, 0, 960, 540}) && t.rotation_deg == 45);
    const auto &l = ops[0].layer;
    assert(l.dst_x == 0 && l.dst_y >= 0 && l.dst_w > 0 && l.dst_x + l.dst_w <= 1920);
    // The same layer unrotated is cut in proportion as before: half the box, half the source.
    const auto plain = resolve(R"([{"dst_x":-480,"dst_y":0,"dst_w":960,"dst_h":540}])", frame);
    assert(!plain[0].transform);
    const auto &q = plain[0].layer;
    assert(q.dst_x == 0 && q.dst_w == 480 && q.crop_x == 960 && q.crop_w == 960 && q.crop_h == 1080);
}

static void rotated_tile_layer_clips_to_its_tile(const av::VideoFrame &frame) {
    const auto ops = resolve(R"([{"dst_x":0,"dst_y":0,"dst_w":1920,"dst_h":1080,"rotation_deg":20,
        "tile":{"x":960,"y":0,"w":960,"h":540},"scene_canvas":{"w":1920,"h":1080}}])", frame);
    assert(ops[0].src && ops[0].transform);
    const auto &t = *ops[0].transform;
    assert(sameRect(t.clip, {960, 0, 960, 540}));
    assert(sameRect(t.box, {960, 0, 960, 540}));
    assert(near(t.pivot_x, 1440) && near(t.pivot_y, 270));
    const auto &l = ops[0].layer;
    assert(l.dst_x >= 960 && l.dst_y >= 0 && l.dst_x + l.dst_w <= 1920 && l.dst_y + l.dst_h <= 540);
}

static void drawop_equality_sees_the_transform(const av::VideoFrame &frame) {
    const auto a = resolve(R"([{"dst_w":960,"dst_h":540,"rotation_deg":10}])", frame);
    const auto b = resolve(R"([{"dst_w":960,"dst_h":540,"rotation_deg":11}])", frame);
    assert(a[0] == a[0] && !(a[0] == b[0]));
}

static void transformed_op_never_copies_the_whole_frame(const av::VideoFrame &frame) {
    const auto ops = resolve(R"([{"dst_w":1920,"dst_h":1080,"crop":{"x":0.25,"y":0,"w":1920,"h":1080}}])", frame);
    assert(ops[0].transform && !copiesWholeFrame(ops, 1920, 1080));
}

// Today's chain (cuda_rect_draw.cpp launch()), verbatim: tables without an affine layer keep it.
static CompositeEntry before(const TableKinds &k, int planes) {
    return planes == 1 ? CompositeEntry::Packed4
        : k.filter ? (k.rgb || k.wide ? CompositeEntry::Filter
                      : k.array ? CompositeEntry::YuvArrayFilter : CompositeEntry::YuvFilter)
        : k.fade ? (k.array ? CompositeEntry::OpacityArray : CompositeEntry::Opacity)
        : k.rgb ? (k.array ? CompositeEntry::Array : CompositeEntry::Full)
        : k.array ? CompositeEntry::YuvArray : CompositeEntry::Yuv;
}

static void entry_selection() {
    for (int bits = 0; bits < 32; ++bits) {
        TableKinds k;
        k.rgb = bits & 1; k.fade = bits & 2; k.array = bits & 4; k.filter = bits & 8; k.wide = bits & 16;
        for (int planes : {1, 2}) assert(compositeEntry(k, planes) == before(k, planes));
        k.affine = true;
        assert(compositeEntry(k, 1) == CompositeEntry::Packed4);
        assert(compositeEntry(k, 2) == (!k.rgb && !k.array && !k.fade ? CompositeEntry::YuvAffine : CompositeEntry::Affine));
    }
}

int main() {
    entry_selection();
    identity_is_the_unit_matrix();
    subpixel_crop_scales_and_offsets();
    quarter_turns_are_exact();
    pivot_default_is_the_box_centre();
    plane_matrices_conjugate_subsampling();
    bounds_clip_to_the_clip_rect();
    edges_clamp_but_mapping_does_not();
    needs_transform_only_for_rotation_or_fractions();
    parse_follows_the_geometry_rule();
    av::VideoFrame frame(AV_PIX_FMT_NV12, 1920, 1080);
    rotated_layer_keeps_its_crop_whole(frame);
    rotated_tile_layer_clips_to_its_tile(frame);
    drawop_equality_sees_the_transform(frame);
    transformed_op_never_copies_the_whole_frame(frame);
}
