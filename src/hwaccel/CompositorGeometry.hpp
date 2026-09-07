#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>

namespace avp::compositor {

struct Rect { int x = 0, y = 0, w = 0, h = 0; };
struct Placement { Rect source, destination; };

// Resolve against this frame, never a cached decoder format. Zero crop extent
// means the remaining frame; destination is a fixed box, optionally letterboxed.
inline std::optional<Placement> place(int width, int height, Rect crop, Rect box,
                                      bool fit, int align_x, int align_y) {
    if (width <= 0 || height <= 0 || box.w <= 0 || box.h <= 0) return {};
    auto aligned = [](int value, int alignment) {
        return value - ((value % alignment + alignment) % alignment);
    };
    int64_t right = crop.w > 0 ? int64_t(crop.x) + crop.w : width;
    int64_t bottom = crop.h > 0 ? int64_t(crop.y) + crop.h : height;
    crop.x = aligned(std::clamp(crop.x, 0, width), align_x);
    crop.y = aligned(std::clamp(crop.y, 0, height), align_y);
    crop.w = aligned(int(std::clamp(right, int64_t(crop.x), int64_t(width))) - crop.x, align_x);
    crop.h = aligned(int(std::clamp(bottom, int64_t(crop.y), int64_t(height))) - crop.y, align_y);
    if (crop.w <= 0 || crop.h <= 0) return {};
    int w = box.w, h = box.h;
    if (fit) {
        if (int64_t(crop.w) * h >= int64_t(crop.h) * w)
            h = int(int64_t(crop.h) * w / crop.w);
        else
            w = int(int64_t(crop.w) * h / crop.h);
    }
    w = aligned(w, align_x); h = aligned(h, align_y);
    if (w <= 0 || h <= 0) return {};
    box.x = aligned(box.x + (box.w - w) / 2, align_x);
    box.y = aligned(box.y + (box.h - h) / 2, align_y);
    box.w = w; box.h = h;
    return Placement{crop, box};
}
// Collapse letterboxing into a virtual source canvas followed by destination
// fitting. This preserves an established source aspect without allocating or
// scaling an intermediate frame.
inline std::optional<Placement> placeInCanvas(int width, int height, Rect crop, Rect box,
                                              int canvas_w, int canvas_h, int ax, int ay) {
    auto inner = place(width, height, crop, {0, 0, canvas_w, canvas_h}, true, ax, ay);
    auto outer = place(canvas_w, canvas_h, {}, box, true, ax, ay);
    if (!inner || !outer) return {};
    auto map = [](int value, int extent, int canvas, int alignment) {
        int result = int(int64_t(value) * extent / canvas);
        return result - result % alignment;
    };
    const auto &i = inner->destination, &o = outer->destination;
    Rect destination{o.x + map(i.x, o.w, canvas_w, ax), o.y + map(i.y, o.h, canvas_h, ay),
                     map(i.w, o.w, canvas_w, ax), map(i.h, o.h, canvas_h, ay)};
    if (destination.w <= 0 || destination.h <= 0) return {};
    return Placement{inner->source, destination};
}

}
