#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>

extern "C" {
#include <libavutil/mathematics.h>
}

namespace avp::mixer {

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
            h = int(av_rescale(crop.h, w, crop.w));
        else
            w = int(av_rescale(crop.w, h, crop.h));
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
        int result = int(av_rescale(value, extent, canvas));
        return result - result % alignment;
    };
    const auto &i = inner->destination, &o = outer->destination;
    Rect destination{o.x + map(i.x, o.w, canvas_w, ax), o.y + map(i.y, o.h, canvas_h, ay),
                     map(i.w, o.w, canvas_w, ax), map(i.h, o.h, canvas_h, ay)};
    if (destination.w <= 0 || destination.h <= 0) return {};
    return Placement{inner->source, destination};
}

/// How a layer is resampled. `Auto` chooses among the other three by the layer's scale.
enum class ScaleFilter { Auto, Bilinear, Bicubic, Multisample };
inline constexpr ScaleFilter kScaleFilters[] = {ScaleFilter::Auto, ScaleFilter::Bilinear, ScaleFilter::Bicubic,
                                                ScaleFilter::Multisample};

inline const char *scaleFilterName(ScaleFilter filter) {
    switch (filter) {
    case ScaleFilter::Auto: return "auto";
    case ScaleFilter::Bicubic: return "bicubic";
    case ScaleFilter::Multisample: return "multisample";
    default: return "bilinear";
    }
}

/// A layer's filter settings. The defaults leave a layer bilinear.
struct FilterSpec {
    ScaleFilter mode = ScaleFilter::Bilinear;
    // The cubic's coefficient with the sign of FFmpeg scale_cuda's `param` (the cubic's A is its
    // negative). 0 is scale_cuda's default bicubic: no weight on the outer taps, so a 2x2
    // interpolation with Hermite weights; 0.5 is Catmull-Rom.
    float bicubic_param = 0.f;
    int samples = 4;                // multisample: 4 or 8 bilinear samples per output sample
    double bicubic_above = 1.3;     // auto: bicubic when a layer enlarges by more than this
    double multisample_above = 2.;  // auto: multisample when a layer shrinks by more than this

    bool operator==(const FilterSpec &other) const {
        return mode == other.mode && bicubic_param == other.bicubic_param && samples == other.samples &&
               bicubic_above == other.bicubic_above && multisample_above == other.multisample_above;
    }
};

/// The filter that draws a layer scaling `sw`x`sh` source pixels to `dw`x`dh`: `spec.mode`, or for
/// `Auto` the choice by scale, never `Auto` itself. The axis with the larger factor decides.
/// Shrinking is tested first: a layer that shrinks on one axis and enlarges on the other aliases
/// on the first under a cubic, while multisampling only stays soft on the second.
inline ScaleFilter resolveScaleFilter(const FilterSpec &spec, int sw, int sh, int dw, int dh) {
    if (spec.mode != ScaleFilter::Auto) return spec.mode;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return ScaleFilter::Bilinear;
    if (std::max(double(sw) / dw, double(sh) / dh) > spec.multisample_above) return ScaleFilter::Multisample;
    if (std::max(double(dw) / sw, double(dh) / sh) > spec.bicubic_above) return ScaleFilter::Bicubic;
    return ScaleFilter::Bilinear;
}

}
