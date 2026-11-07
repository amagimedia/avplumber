// Move transition: matched layers glide from scene A's placement to scene B's, unmatched ones fade
// or cut, and the stacking follows the lerped z keys.
#include "mixer/primitives/layer_move.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

void print_stack_trace() {}

using namespace avp::mixer;

static bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }
static bool nearRect(const RectF &a, const RectF &b, double eps = 1e-6) {
    return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.w, b.w, eps) && near(a.h, b.h, eps);
}
static std::vector<LayerSpec> layers(const char *json) { return parseLayersArray(Parameters::parse(json)); }
static RectF box(const LayerSpec &l) { return l.dst_exact ? *l.dst_exact : RectF{double(l.dst_x), double(l.dst_y), double(l.dst_w), double(l.dst_h)}; }
static RectF crop(const LayerSpec &l) { return l.crop_exact ? *l.crop_exact : RectF{double(l.crop_x), double(l.crop_y), double(l.crop_w), double(l.crop_h)}; }
static const LayerSpec *byInput(const std::vector<LayerSpec> &v, int input) {
    for (const auto &l : v) if (l.input == input) return &l;
    return nullptr;
}

struct Sources {
    av::VideoFrame hd0{AV_PIX_FMT_NV12, 1920, 1080}, hd1{AV_PIX_FMT_NV12, 1920, 1080}, sd{AV_PIX_FMT_NV12, 1440, 1080};
    std::vector<const av::VideoFrame *> all() const { return {&hd0, &hd1, &sd, &hd0}; }
};

static std::vector<LayerSpec> move(const std::vector<LayerSpec> &a, const std::vector<LayerSpec> &b, double s,
                                   const Sources &src, const std::string &unmatched = "fade") {
    return moveLayers(a, b, s, src.all(), AV_PIX_FMT_NV12, unmatched);
}

static const char *kFull = R"([{"input":0,"dst_w":1920,"dst_h":1080},{"input":1,"dst_x":1400,"dst_y":60,"dst_w":480,"dst_h":270,"z":1}])";
static const char *kSplit = R"([{"input":0,"dst_w":960,"dst_h":540,"dst_y":270},{"input":3,"dst_x":960,"dst_y":270,"dst_w":960,"dst_h":540}])";

static void endpoints_verbatim(const Sources &src) {
    const auto a = layers(kFull), b = layers(kSplit);
    assert(move(a, b, 0, src) == a && move(a, b, -0.5, src) == a);
    assert(move(a, b, 1, src) == b && move(a, b, 1.5, src) == b);
}

static void matched_box_is_geometric(const Sources &src) {
    const auto a = layers(R"([{"input":0,"dst_w":1920,"dst_h":1080}])");
    const auto b = layers(R"([{"input":0,"dst_x":960,"dst_y":540,"dst_w":480,"dst_h":270}])");
    for (double s : {0.25, 0.5, 0.9}) {
        const auto out = move(a, b, s, src);
        assert(out.size() == 1 && out[0].input == 0 && out[0].dst_exact && !out[0].fit);
        assert(nearRect(box(out[0]), zoomRect({0, 0, 1920, 1080}, {960, 540, 480, 270}, s)));
        assert(near(box(out[0]).w, 1920 * std::pow(0.25, s)));
        assert(nearRect(crop(out[0]), {0, 0, 1920, 1080}));
        // The integer fields enclose the fractional box.
        const RectF f = *out[0].dst_exact;
        assert(out[0].dst_x <= f.x && out[0].dst_x + out[0].dst_w >= f.x + f.w && out[0].dst_x + 1 > f.x);
        assert(needsTransform(out[0]));
    }
}

static void rotation_takes_the_shortest_arc(const Sources &src) {
    const auto a = layers(R"([{"input":0,"dst_w":960,"dst_h":540,"rotation_deg":170}])");
    const auto b = layers(R"([{"input":0,"dst_w":960,"dst_h":540,"rotation_deg":-170}])");
    assert(near(move(a, b, 0.25, src)[0].rotation_deg, 175));
    assert(near(move(a, b, 0.5, src)[0].rotation_deg, 180));
    assert(near(move(b, a, 0.25, src)[0].rotation_deg, -175));
    const auto c = layers(R"([{"input":0,"dst_w":960,"dst_h":540}])");
    assert(near(move(c, a, 0.5, src)[0].rotation_deg, 85));
}

static void stacking_crosses_at_the_keys(const Sources &src) {
    const auto a = layers(R"([{"input":0,"dst_w":960,"dst_h":540,"z":0},{"input":1,"dst_x":480,"dst_w":960,"dst_h":540,"z":2}])");
    const auto b = layers(R"([{"input":0,"dst_x":480,"dst_w":960,"dst_h":540,"z":2},{"input":1,"dst_w":960,"dst_h":540,"z":0}])");
    const auto before = move(a, b, 0.4, src), after = move(a, b, 0.6, src);
    assert(before.size() == 2 && after.size() == 2);
    assert(before.back().input == 1 && after.back().input == 0);   // drawn last = on top
    for (const auto *v : {&before, &after})
        for (size_t i = 0; i < v->size(); ++i) assert((*v)[i].z == int(i));
}

static void contain_is_placed_before_the_glide(const Sources &src) {
    // A 4:3 camera contained in the full frame and in the top-right quarter.
    const auto a = layers(R"([{"input":2,"dst_w":1920,"dst_h":1080,"fit":"contain"}])");
    const auto b = layers(R"([{"input":2,"dst_x":960,"dst_w":960,"dst_h":540,"fit":"contain"}])");
    const RectF from{240, 0, 1440, 1080}, to{1080, 0, 720, 540};
    assert(nearRect(box(move(a, b, 1e-9, src)[0]), from, 1e-4));
    const auto mid = move(a, b, 0.5, src)[0];
    assert(!mid.fit && nearRect(box(mid), zoomRect(from, to, 0.5)) && nearRect(crop(mid), {0, 0, 1440, 1080}));
}

static void unmatched_fade_and_cut(const Sources &src) {
    const auto a = layers(kFull), b = layers(kSplit);   // input 1 only in A, input 3 only in B
    const auto fade = move(a, b, 0.25, src);
    assert(byInput(fade, 1) && near(byInput(fade, 1)->opacity, 0.75) && byInput(fade, 3) && near(byInput(fade, 3)->opacity, 0.25));
    assert(near(byInput(fade, 0)->opacity, 1));
    // The faded layers keep their own placement; a faded camera draws through the affine entry.
    assert(box(*byInput(fade, 1)).x == 1400 && needsTransform(*byInput(fade, 3)));
    const auto early = move(a, b, 0.25, src, "cut"), late = move(a, b, 0.5, src, "cut");
    assert(byInput(early, 1) && near(byInput(early, 1)->opacity, 1) && !byInput(early, 3));
    assert(!byInput(late, 1) && byInput(late, 3) && near(byInput(late, 3)->opacity, 1));
}

static void faded_out_layers_are_dropped(const Sources &src) {
    auto a = layers(kFull);
    a[1].opacity = 0;   // a key already faded out
    const auto out = move(a, layers(kSplit), 0.3, src);
    assert(!byInput(out, 1));
    for (const auto &l : out) {
        assert(l.opacity > 0);
        assert(l.dst_w > 0 && l.dst_h > 0 && box(l).w > 0 && box(l).h > 0);
    }
    // A matched source without a frame is never interpolated against a guess: it is left out.
    const std::vector<const av::VideoFrame *> none{nullptr, &src.hd1, &src.sd, &src.hd0};
    const auto missing = moveLayers(layers(kFull), layers(kSplit), 0.5, none, AV_PIX_FMT_NV12, "fade");
    assert(!byInput(missing, 0));
}

static void json_round_trip() {
    const auto spec = parseMoveSpec(Parameters::parse(
        R"({"generation":7,"unmatched":"cut","from":[{"input":2,"dst_x":10,"dst_w":640,"dst_h":360,"rotation_deg":15}]})"));
    assert(spec.generation == 7 && spec.unmatched == "cut" && spec.from.size() == 1 && spec.from[0].input == 2);
    const auto again = parseMoveSpec(moveSpecJson(spec));
    assert(again.generation == 7 && again.unmatched == "cut" && again.from == spec.from);
    assert(parseMoveSpec(Parameters::parse(R"({"generation":1,"from":[]})")).unmatched == "fade");
    bool threw = false;
    try { parseMoveSpec(Parameters::parse(R"({"generation":1,"from":[],"unmatched":"slide"})")); } catch (const std::exception &) { threw = true; }
    assert(threw);
}

// The slot's move against the published take clock: progress 0 until its take is on air, the
// clock's progress while it runs, dropped once the take ends or a later take is on air.
static TakeClock clockAt(uint64_t generation, int64_t start_ns, int64_t duration_ns) {
    TakeClock c;
    c.generation = generation;
    c.schedule = RampHistory({start_ns, duration_ns});
    c.on_air_ns = start_ns;
    return c;
}

static void slot_move_follows_the_take_clock(const Sources &src) {
    const auto a = layers(kFull), b = layers(kSplit);
    SlotMove m;
    MoveSpec spec;
    spec.generation = 5;
    spec.from = a;
    m.set(spec);
    auto drawn = b;
    assert(m.apply(drawn, nullptr, 0, src.all(), AV_PIX_FMT_NV12) && drawn == a && m.drawn == 0);   // no clock yet
    const TakeClock before = clockAt(4, 0, 1000);
    drawn = b;
    assert(m.apply(drawn, &before, 500, src.all(), AV_PIX_FMT_NV12) && drawn == a);
    const TakeClock on = clockAt(5, 1000, 1000);
    drawn = b;
    assert(m.apply(drawn, &on, 500, src.all(), AV_PIX_FMT_NV12) && drawn == a);   // before its first frame
    drawn = b;
    assert(m.apply(drawn, &on, 1500, src.all(), AV_PIX_FMT_NV12) && near(m.drawn, 0.5));
    assert(drawn == moveLayers(a, b, 0.5, src.all(), AV_PIX_FMT_NV12, "fade"));
    // Progress 1 while the take still runs draws B (a lever at its end), and keeps the move.
    drawn = b;
    assert(m.apply(drawn, &on, 2500, src.all(), AV_PIX_FMT_NV12) && drawn == b && m.spec);
    TakeClock done = on;
    done.end = TakeClock::End::Completed;
    done.end_ns = 2000;
    drawn = b;
    assert(m.apply(drawn, &done, 1900, src.all(), AV_PIX_FMT_NV12) && m.spec);   // a tick before the end
    drawn = b;
    assert(!m.apply(drawn, &done, 2000, src.all(), AV_PIX_FMT_NV12) && !m.spec && drawn == b);
    // A slot at rest does nothing.
    assert(!m.apply(drawn, &done, 2100, src.all(), AV_PIX_FMT_NV12) && drawn == b);
}

static void slot_move_drops_on_cancel_and_later_takes(const Sources &src) {
    const auto a = layers(kFull), b = layers(kSplit);
    MoveSpec spec;
    spec.generation = 5;
    spec.from = a;
    SlotMove m;
    m.set(spec);
    TakeClock cancelled = clockAt(5, 1000, 1000);
    cancelled.end = TakeClock::End::Cancelled;
    cancelled.end_ns = 1400;
    auto drawn = b;
    assert(m.apply(drawn, &cancelled, 1300, src.all(), AV_PIX_FMT_NV12) && m.spec);
    drawn = b;
    assert(!m.apply(drawn, &cancelled, 1400, src.all(), AV_PIX_FMT_NV12) && !m.spec && drawn == b);
    m.set(spec);
    const TakeClock later = clockAt(6, 0, 1000);   // its end was missed: a later take is on air
    drawn = b;
    assert(!m.apply(drawn, &later, 100, src.all(), AV_PIX_FMT_NV12) && !m.spec && drawn == b);
}

// Two backgrounds that differ crossfade as a fade does, never through black: the incoming one stays
// whole under the outgoing one, which alone fades.
static void covered_background_stays_whole(const Sources &src) {
    const auto a = layers(R"([{"input":0,"dst_w":1920,"dst_h":1080},{"input":1,"dst_x":1400,"dst_w":480,"dst_h":270,"z":1}])");
    const auto b = layers(R"([{"input":3,"dst_w":1920,"dst_h":1080},{"input":1,"dst_x":100,"dst_w":480,"dst_h":270,"z":1}])");
    const auto out = move(a, b, 0.5, src);
    const auto at = [&](int input) { for (size_t i = 0; i < out.size(); ++i) if (out[i].input == input) return int(i); return -1; };
    assert(at(3) >= 0 && at(0) > at(3));                      // the outgoing background draws above
    assert(near(byInput(out, 3)->opacity, 1) && near(byInput(out, 0)->opacity, 0.5));
    // A blended key does not cover: the layer under it still fades.
    auto keyed = a;
    keyed[0].blend = true;
    assert(near(byInput(move(keyed, b, 0.5, src), 3)->opacity, 0.5));
}

int main() {
    const Sources src;
    endpoints_verbatim(src);
    matched_box_is_geometric(src);
    rotation_takes_the_shortest_arc(src);
    stacking_crosses_at_the_keys(src);
    contain_is_placed_before_the_glide(src);
    unmatched_fade_and_cut(src);
    faded_out_layers_are_dropped(src);
    covered_background_stays_whole(src);
    json_round_trip();
    slot_move_follows_the_take_clock(src);
    slot_move_drops_on_cancel_and_later_takes(src);
    return 0;
}
