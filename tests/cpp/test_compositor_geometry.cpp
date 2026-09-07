#include "hwaccel/CompositorGeometry.hpp"
#include <cassert>
#include <limits>
using namespace avp::compositor;
int main() {
    const Rect tile{540, 240, 540, 240};
    auto wide = place(1920, 1080, {}, tile, true, 2, 2);
    auto changed = place(1280, 720, {}, tile, true, 2, 2);
    assert(wide && changed);
    assert(wide->destination.w == 426 && wide->destination.h == 240);
    assert(wide->destination.x == 596);
    assert(changed->destination.w == wide->destination.w);
    assert(changed->source.w == 1280 && wide->source.w == 1920);
    auto portrait = place(720, 1280, {}, tile, true, 2, 2);
    assert(portrait->destination.w == 134 && portrait->destination.h == 240);
    auto stretch = place(720, 1280, {}, tile, false, 2, 2);
    assert(stretch->destination.w == 540);
    auto cropped = place(1280, 720, {320, 180, 640, 360}, tile, true, 2, 2);
    assert(cropped->source.x == 320 && cropped->source.w == 640);
    assert(cropped->destination.w == wide->destination.w);
    auto remaining = place(1280, 720, {640, 360}, tile, true, 2, 2);
    assert(remaining->source.w == 640 && remaining->source.h == 360);
    assert(!place(1280, 720, {1300, 0}, tile, true, 2, 2));
    assert(!place(0, 720, {}, tile, true, 2, 2));
    auto negative = place(1280, 720, {-100, -100, 400, 400}, tile, false, 2, 2);
    assert(negative->source.x == 0 && negative->source.w == 300);
    auto overflow = place(1280, 720, {100, 100, std::numeric_limits<int>::max(), 400}, tile, false, 2, 2);
    assert(overflow->source.w == 1180);
    auto direct = placeInCanvas(640, 360, {}, tile, 1920, 1080, 2, 2);
    assert(direct->destination.w == 426 && direct->destination.h == 240);
    assert(direct->destination.x == wide->destination.x);
    auto portrait_canvas = placeInCanvas(720, 1280, {}, {0, 0, 1080, 1920}, 1920, 1080, 2, 2);
    assert(portrait_canvas->destination.w == 340 && portrait_canvas->destination.h == 606);
    assert(portrait_canvas->destination.x == 368 && portrait_canvas->destination.y == 656);
    assert(!placeInCanvas(640, 360, {}, tile, 0, 1080, 2, 2));
    for (int w = 2; w < 2048; w += 17) {
        auto p = place(w, 719, {}, tile, true, 2, 2);
        if (!p) continue;
        assert(p->source.w <= w && p->source.h <= 719);
        assert(p->destination.x >= tile.x && p->destination.y >= tile.y);
        assert(p->destination.x + p->destination.w <= tile.x + tile.w);
        assert(p->destination.y + p->destination.h <= tile.y + tile.h);
        assert(p->source.w % 2 == 0 && p->destination.w % 2 == 0);
    }
}
