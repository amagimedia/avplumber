#include "mixer/primitives/SourceDelivery.hpp"
#include <cassert>
using namespace avp::mixer;
int main() {
    SourceDelivery d;
    const uint64_t period = 20000000;
    auto show = [&](int tick, std::initializer_list<uint64_t> frames, uint64_t epoch = 1) {
        SourceFrames ids;
        for (auto id : frames) ids["web"].push_back({epoch, id, period});
        d.observe(tick * int64_t(period), period, ids);
    };
    show(0, {100}); // First visible capture, not 100 lost prewarm frames.
    show(1, {101});
    show(2, {101}); // Same pixels with a new PTS (repeat_last_frame, force_fps, snapshot).
    show(3, {103}); // Capture 102 was never shown.
    assert(d.counts().at("web").repeats == 1);
    assert(d.counts().at("web").dropped == 1);
    show(4, {}); show(5, {500}); // Off-air / back on air: no false gap.
    assert(d.counts().at("web").dropped == 1);
    show(6, {501, 502, 502}); // Both fade inputs / duplicate aliases count once.
    assert(d.counts().at("web").dropped == 1);
    show(7, {0}, 2); // Producer restart starts a new sequence.
    assert(d.counts().at("web").dropped == 1);
    show(8, {1}, 2);
    assert(d.counts().at("web").repeats == 1);
    d.reset();
    d.observe(0, period, {{"slow", {{1, 0, 2 * period}}}});
    d.observe(period, period, {{"slow", {{1, 0, 2 * period}}}});
    assert(d.counts().at("slow").repeats == 0); // 25 -> 50 is intentional rate conversion.
    d.observe(2 * period, period, {{"slow", {{1, 0, 2 * period}}}});
    assert(d.counts().at("slow").repeats == 1);
    d.observe(0, period, {{"static", {{1, 0, 0}}}});
    d.observe(period, period, {{"static", {{1, 0, 0}}}});
    assert(d.counts().at("static").repeats == 0);
    assert(d.counts().at("static").static_holds == 1);
    d.observe(2 * period, period, {{"static", {{1, 2, 0}}}});
    assert(d.counts().at("static").dropped == 1); // Static pages can still lose real changes.
    AVFrame* a = av_frame_alloc();
    AVFrame* b = av_frame_alloc();
    AVFrame* out = av_frame_alloc();
    setFrameTag(a, "avp.source.web", SourceFrame{1, 42, period}.encode());
    setFrameTag(b, "avp.source.web", SourceFrame{1, 43, period}.encode());
    mergeSourceFrames(out, a); mergeSourceFrames(out, a); mergeSourceFrames(out, b);
    assert(sourceFrames(out).at("web").size() == 2);
    AVFrame* copy = av_frame_alloc();
    assert(av_frame_copy_props(copy, out) == 0);
    copy->pts = 12345;
    assert(sourceFrames(copy).at("web")[0].number == 42);
    clearSourceFrames(out);
    assert(sourceFrames(out).empty());
    assert(sourceFrames(copy).size() == 1);
    av_frame_free(&copy); av_frame_free(&out); av_frame_free(&b); av_frame_free(&a);
}
