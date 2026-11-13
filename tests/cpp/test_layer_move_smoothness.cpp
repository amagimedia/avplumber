// Behavioural motion checks on the actual inverse maps sent to the CUDA sampler.
// No GPU is needed to detect pixel snapping, direction reversals or zoom breathing.
#include "mixer/primitives/frame_move.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void print_stack_trace() {}
using namespace avp::mixer;

struct Sample { double t, x, y, log_scale; };

static Sample sample(const Parameters &a, const Parameters &b, double t, const char *curve,
                     const av::VideoFrame &frame, double sx, double sy) {
    const Parameters request{{"from", a}, {"to", b}, {"progress", t / .8}, {"curve", curve}};
    const auto layers = frameMoveLayers(request, {&frame}, AV_PIX_FMT_NV12);
    const auto ops = resolveDrawOps({&frame}, layers, 1080, 1920, AV_PIX_FMT_NV12);
    assert(ops.size() == 1 && ops[0].src);
    const auto &op = ops[0];
    double x, y, scale;
    if (op.transform) {
        const auto p = affinePlane(*op.transform, 0, 0, 0, 0);
        assert(p && p->m[1] == 0 && p->m[3] == 0);
        // Forward-map two source landmarks through the FLOAT map used by the
        // rasterizer. Using dst_x/crop_x here would miss the fractional path.
        x = (sx - p->m[2]) / p->m[0];
        y = (sy - p->m[5]) / p->m[4];
        scale = 1. / p->m[0];
    } else {
        const auto &l = op.layer;
        scale = double(l.dst_w) / l.crop_w;
        x = l.dst_x + (sx - l.crop_x) * scale;
        y = l.dst_y + (sy - l.crop_y) * double(l.dst_h) / l.crop_h;
    }
    return {t, x, y, std::log(scale)};
}

static void smooth_path(const std::vector<Sample> &v, double Sample::*field, bool eased) {
    const double start = v.front().*field, end = v.back().*field, travel = end - start;
    if (std::abs(travel) < 1e-5) {
        for (const auto &s : v) assert(std::abs(s.*field - start) < .001);
        return;
    }
    double previous_velocity = 0, previous_dt = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        const double dt = v[i].t - v[i-1].t;
        const double position = (v[i].*field - start) / travel;
        const double step = (v[i].*field - v[i-1].*field) / travel;
        assert(position >= -1e-5 && position <= 1 + 1e-5); // no overshoot
        assert(step >= -1e-5);                            // no reversal
        const double velocity = step / dt;
        assert(velocity < 4.0);                           // no one-frame jump
        if (i > 1) {
            const double acceleration = (velocity - previous_velocity) / ((dt + previous_dt) / 2);
            if (std::abs(acceleration) >= 18.0)
                std::cerr << "motion discontinuity t=" << v[i].t << " accel=" << acceleration
                          << " position=" << position << " travel=" << travel << "\n";
            assert(std::abs(acceleration) < 18.0);        // no staircase/velocity spikes
        }
        previous_velocity = velocity;
        previous_dt = dt;
    }
    if (eased) {
        const double first = std::abs((v[1].*field - start) / travel);
        const double last = std::abs((end - v[v.size()-2].*field) / travel);
        assert(first < .015 && last < .015);              // settle gently at both ends
    }
}

int main() {
    const av::VideoFrame frame(AV_PIX_FMT_NV12, 1920, 1080);
    Parameters lut = Parameters::array();
    for (int p = 0; p < 3; ++p) {
        Parameters codes = Parameters::array();
        for (int i = 0; i < 256; ++i) codes.push_back(255 - i);
        lut.push_back(codes);
    }
    Parameters lut_layer{{"input", 0}, {"dst_w", 1920}, {"dst_h", 1080}, {"yuv_lut", lut}};
    const auto lut_layers = parseLayersArray(Parameters::array({lut_layer}));
    assert(lut_layers[0].yuv_lut && (*lut_layers[0].yuv_lut)[2][42] == 213);
    // A 1:1 transform must not take the no-copy bypass and lose its effect.
    assert(!copiesWholeFrame(resolveDrawOps({&frame}, lut_layers, 1920, 1080, AV_PIX_FMT_NV12), 1920, 1080));
    for (const Parameters invalid : {Parameters(-1), Parameters(256), Parameters(1.5), Parameters(true)}) {
        auto bad = lut_layer;
        bad["yuv_lut"][0][0] = invalid;
        bool rejected = false;
        try { parseLayersArray(Parameters::array({bad})); } catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
    // A frame-driven L1 crop can cross an odd integer without running a move.
    // Its explicit subpixel contract must not switch samplers for that frame.
    for (int i = 0; i <= 40; ++i) {
        const double x = 400 + i / 20.;
        const Parameters layer{{"input", 0}, {"crop_subpixel", true},
            {"crop", {{"x", x}, {"y", 200}, {"w", 600}, {"h", 600}}},
            {"dst_x", 100}, {"dst_y", 100}, {"dst_w", 600}, {"dst_h", 600}};
        const Parameters scene = Parameters::array({layer});
        const auto got = sample(scene, scene, .8, "linear", frame, 600, 400);
        assert(std::abs(got.x - (700 - x)) < .0001);
        assert(std::abs(got.y - 300) < .0001);
    }
    const Parameters a = Parameters::parse(R"([
      {"input":0,"move_id":"speaker","crop":{"x":400,"y":200,"w":600,"h":600},
       "dst_x":100,"dst_y":100,"dst_w":600,"dst_h":600}
    ])");
    // Pan, zoom in and zoom out; include a sub-pixel-size pan to detect snapping.
    for (const auto &b : std::vector<Parameters>{
        Parameters::parse(R"([{"input":0,"move_id":"speaker","crop":{"x":440,"y":220,"w":600,"h":600},"dst_x":100,"dst_y":100,"dst_w":600,"dst_h":600}])"),
        Parameters::parse(R"([{"input":0,"move_id":"speaker","crop":{"x":400,"y":200,"w":400,"h":400},"dst_x":100,"dst_y":100,"dst_w":600,"dst_h":600}])"),
        Parameters::parse(R"([{"input":0,"move_id":"speaker","crop":{"x":400,"y":200,"w":800,"h":800},"dst_x":100,"dst_y":100,"dst_w":600,"dst_h":600}])"),
        Parameters::parse(R"([{"input":0,"move_id":"speaker","crop":{"x":402,"y":200,"w":600,"h":600},"dst_x":100,"dst_y":100,"dst_w":600,"dst_h":600}])")}) {
        for (const double fps : {25., 30000./1001., 50.})
        for (const char *curve : {"linear", "ease-in", "ease-out", "ease-in-out"}) {
            std::vector<Sample> path;
            for (int i = 0; i / fps < .8 - 1e-9; ++i)
                path.push_back(sample(a, b, i / fps, curve, frame, 600, 400));
            path.push_back(sample(a, b, .8, curve, frame, 600, 400));
            const bool eased = std::string(curve) == "ease-in-out";
            for (auto field : {&Sample::x, &Sample::y, &Sample::log_scale}) smooth_path(path, field, eased);
            // The final held frame and the transition endpoint agree exactly.
            const auto held = sample(b, b, .84, curve, frame, 600, 400);
            assert(std::abs(held.x - path.back().x) < 1e-6);
            assert(std::abs(held.log_scale - path.back().log_scale) < 1e-6);
        }
    }
}
