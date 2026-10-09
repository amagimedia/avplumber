#pragma once
// Viewports of the composite kernel: a launch that composes one rectangle of a canvas
// instead of all of it. The kernel is unchanged; it is handed the rectangle as its canvas
// (plane pointers at the rectangle's origin, the rectangle's size) and a rect table whose
// destinations are relative to that origin. Plain C++ without FFmpeg, shared by
// CudaRectDraw and tests/cuda/test_rect_composite.cu.
#include "cuda_rect_table.h"

#include <algorithm>
#include <vector>

namespace avp::mixer {

struct RectViewport {
    int x = 0, y = 0, w = 0, h = 0;   // luma samples of the canvas
    bool empty() const { return w <= 0 || h <= 0; }
    bool operator==(const RectViewport &o) const { return x == o.x && y == o.y && w == o.w && h == o.h; }
};

/// The rectangle grown outward to what a launch may own and clipped to the canvas: columns in
/// fours (a thread stores AVP_RECT_PX luma samples, or half as many chroma pairs, at once, so
/// the origin keeps those stores aligned) and rows in twos (whole chroma rows). Empty when the
/// rectangle misses the canvas.
inline RectViewport alignViewport(int x, int y, int w, int h, int canvas_w, int canvas_h) {
    const int x0 = std::max(0, x) / AVP_RECT_PX * AVP_RECT_PX, y0 = std::max(0, y) / 2 * 2;
    const int x1 = std::min(canvas_w, (x + w + AVP_RECT_PX - 1) / AVP_RECT_PX * AVP_RECT_PX);
    const int y1 = std::min(canvas_h, (y + h + 1) / 2 * 2);
    if (w <= 0 || h <= 0 || x1 <= x0 || y1 <= y0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

/// Replaces viewports that overlap by their bounding box until none do: two launches over
/// the same samples would race on them.
inline void mergeViewports(std::vector<RectViewport> &views) {
    for (bool merged = true; merged;) {
        merged = false;
        for (size_t i = 0; i < views.size() && !merged; ++i)
            for (size_t j = i + 1; j < views.size() && !merged; ++j) {
                RectViewport &a = views[i];
                const RectViewport &b = views[j];
                if (a.x >= b.x + b.w || b.x >= a.x + a.w || a.y >= b.y + b.h || b.y >= a.y + a.h) continue;
                const int x1 = std::max(a.x + a.w, b.x + b.w), y1 = std::max(a.y + a.h, b.y + b.h);
                a.x = std::min(a.x, b.x);
                a.y = std::min(a.y, b.y);
                a.w = x1 - a.x;
                a.h = y1 - a.y;
                views.erase(views.begin() + j);
                merged = true;
            }
    }
}

/// A whole-canvas table entry as the entry of a launch over `view`: the same layer, its
/// destination relative to the viewport's origin on both planes. `sub_x` and `sub_y` are the
/// canvas's chroma shifts; the origin is a multiple of both steps (alignViewport).
inline AvpRectLayer viewportLayer(AvpRectLayer layer, const RectViewport &view, int sub_x, int sub_y) {
    layer.dx[0] -= view.x;
    layer.dy[0] -= view.y;
    layer.dx[1] -= view.x >> sub_x;
    layer.dy[1] -= view.y >> sub_y;
    return layer;
}

}
