// One decoded picture can provide several independently moving crops. Leaving the
// canvas is a placement change, not an unmatched-layer fade.
#include "mixer/primitives/frame_move.hpp"
#include <cassert>
#include <cmath>

void print_stack_trace() {}

using namespace avp::mixer;

static bool near(double a, double b) { return std::abs(a - b) < 1e-6; }
static RectF box(const LayerSpec &l) {
    return l.dst_exact.value_or(RectF{double(l.dst_x), double(l.dst_y), double(l.dst_w), double(l.dst_h)});
}
static RectF crop(const LayerSpec &l) {
    return l.crop_exact.value_or(RectF{double(l.crop_x), double(l.crop_y), double(l.crop_w), double(l.crop_h)});
}
static const LayerSpec &input(const std::vector<LayerSpec> &layers, int index) {
    const auto it = std::find_if(layers.begin(), layers.end(), [index](const LayerSpec &l) { return l.input == index; });
    assert(it != layers.end());
    return *it;
}

static void exits_keep_their_picture_and_opacity(const av::VideoFrame &frame) {
    const auto start = parseLayersArray(Parameters::parse(R"([
      {"input":0,"crop":{"x":960,"y":0,"w":608,"h":1080},
       "dst_x":270,"dst_y":480,"dst_w":540,"dst_h":960,"fit":"stretch"}
    ])"));
    // Left/right/top/bottom: each destination is wholly outside a 1080x1920 canvas.
    for (const auto curve : {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut})
    for (const auto &end : std::vector<std::pair<int, int>>{{-540,480}, {1080,480}, {270,-960}, {270,1920}}) {
        auto target = start;
        target[0].dst_x = end.first;
        target[0].dst_y = end.second;
        assert(moveLayers(start, target, 0, {&frame}, AV_PIX_FMT_NV12, "fade") == start);
        assert(moveLayers(start, target, 1, {&frame}, AV_PIX_FMT_NV12, "fade") == target);
        for (int tick = 1; tick < 50; ++tick) {
            const double s = fadeCurveAt(curve, tick / 50.);
            const auto out = moveLayers(start, target, s, {&frame}, AV_PIX_FMT_NV12, "fade");
            assert(out.size() == 1 && out[0].opacity == 1);
            const auto b = box(out[0]), c = crop(out[0]);
            assert(near(b.x, 270 + (end.first - 270) * s));
            assert(near(b.y, 480 + (end.second - 480) * s));
            assert(near(b.w, 540) && near(b.h, 960));
            assert(near(c.x, 960) && near(c.y, 0) && near(c.w, 608) && near(c.h, 1080));
            const auto ops = resolveDrawOps({&frame}, out, 1080, 1920, AV_PIX_FMT_NV12);
            assert(ops.size() == 1 && ops[0].src == &frame && ops[0].transform);
            const auto clip = ops[0].transform->clip;
            assert(clip.x == 0 && clip.y == 0 && clip.w == 1080 && clip.h == 1920);
        }
        const auto ops = resolveDrawOps({&frame}, target, 1080, 1920, AV_PIX_FMT_NV12);
        assert(ops.size() == 1 && ops[0].src == nullptr);
    }
}

static void two_crops_of_one_frame_keep_their_identity(const av::VideoFrame &frame) {
    const auto start = parseLayersArray(Parameters::parse(R"([
      {"input":0,"crop":{"x":0,"y":0,"w":608,"h":540},
       "dst_w":1080,"dst_h":960,"fit":"stretch"},
      {"input":1,"crop":{"x":960,"y":0,"w":608,"h":540},
       "dst_y":960,"dst_w":1080,"dst_h":960,"fit":"stretch","z":1}
    ])"));
    // Reverse the list order: the speaker stays input 1 while expanding to full
    // screen, and input 0 exits upward. Both pads refer to the very same frame.
    const auto target = parseLayersArray(Parameters::parse(R"([
      {"input":1,"crop":{"x":960,"y":0,"w":608,"h":1080},
       "dst_w":1080,"dst_h":1920,"fit":"stretch","z":1},
      {"input":0,"crop":{"x":0,"y":0,"w":608,"h":540},
       "dst_y":-960,"dst_w":1080,"dst_h":960,"fit":"stretch"}
    ])"));
    const std::vector<const av::VideoFrame *> sources{&frame, &frame};
    for (const auto curve : {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut}) {
    double previous_y = 960, previous_h = 960;
    for (int tick = 0; tick <= 50; ++tick) {
        const double s = fadeCurveAt(curve, tick / 50.);
        const auto out = moveLayers(start, target, s, sources, AV_PIX_FMT_NV12, "fade");
        assert(out.size() == 2);
        const auto &leaving = input(out, 0), &speaker = input(out, 1);
        assert(leaving.opacity == 1 && speaker.opacity == 1);
        const auto a = box(leaving), b = box(speaker);
        assert(near(a.y, -960 * s) && near(a.h, 960));
        assert(near(crop(leaving).x, 0) && near(crop(speaker).x, 960));
        assert(b.y <= previous_y + 1e-6 && b.h >= previous_h - 1e-6);
        previous_y = b.y;
        previous_h = b.h;
        // Re-evaluation at the same media position cannot advance/restart a move.
        assert(moveLayers(start, target, s, sources, AV_PIX_FMT_NV12, "fade") == out);
    }
    assert(near(previous_y, 0) && near(previous_h, 1920));
    }
}

int main() {
    const av::VideoFrame frame(AV_PIX_FMT_NV12, 1920, 1080);
    exits_keep_their_picture_and_opacity(frame);
    two_crops_of_one_frame_keep_their_identity(frame);
    for (const auto &curve : {"linear", "ease-in", "ease-out", "ease-in-out"}) {
        const Parameters from = {{{"input",0}, {"dst_w",1080}, {"dst_h",1920}}};
        const Parameters to = {{{"input",0}, {"dst_x",-1080}, {"dst_w",1080}, {"dst_h",1920}}};
        for (const double progress : {-1., 0., 0.25, 0.5, 0.75, 1., 2.}) {
            const Parameters request{{"from",from}, {"to",to}, {"progress",progress}, {"curve",curve}};
            assert(frameMoveLayers(request, {&frame}, AV_PIX_FMT_NV12) ==
                   moveLayers(parseLayersArray(from), parseLayersArray(to),
                              fadeCurveAt(parseFadeCurve(curve), progress), {&frame}, AV_PIX_FMT_NV12, "fade"));
        }
    }
}
