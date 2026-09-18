#pragma once
// Layer descriptions for the CUDA compositor: JSON parsing, per-frame metadata
// overrides, and the resolution of layers against the actual source frames into
// an ordered list of draw operations. No CUDA here.
#include "../../util.hpp"
#include "compositor_geometry.hpp"
#include "pixel_layout.hpp"
#include <avcpp/frame.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace avp::mixer {

struct LayerSpec {
    int dst_x = 0;
    int dst_y = 0;
    int crop_x = 0;
    int crop_y = 0;
    int dst_w = 0;
    int dst_h = 0;
    bool fit = false;
    int source_canvas_w = 0;
    int source_canvas_h = 0;
    int crop_w= 0;
    int crop_h = 0;
    int z = 0;   // draw order: lower first, ties by source index
    bool blend = false;   // honour the source's alpha instead of overwriting

    bool operator==(const LayerSpec &other) const {
        return dst_x == other.dst_x && dst_y == other.dst_y && z == other.z && blend == other.blend &&
               crop_x == other.crop_x && crop_y == other.crop_y &&
               crop_w == other.crop_w && crop_h == other.crop_h &&
               dst_w == other.dst_w && dst_h == other.dst_h && fit == other.fit &&
               source_canvas_w == other.source_canvas_w && source_canvas_h == other.source_canvas_h;
    }
};

struct DrawOp {
    const av::VideoFrame *src = nullptr;
    int src_w = 0;
    int src_h = 0;
    LayerSpec layer;

    bool operator==(const DrawOp &other) const {
        return (src != nullptr) == (other.src != nullptr) &&
               src_w == other.src_w && src_h == other.src_h &&
               layer == other.layer;
    }
};

inline void parseLayerFromJson(const Parameters &obj, LayerSpec &out) {
    out.dst_x = obj.value("dst_x", 0);
    out.dst_y = obj.value("dst_y", 0);
    out.dst_w = obj.value("dst_w", 0);
    out.dst_h = obj.value("dst_h", 0);
    out.z = obj.value("z", 0);
    out.blend = obj.value("blend", false);
    const std::string fit = obj.value("fit", std::string("stretch"));
    if (fit != "stretch" && fit != "contain")
        throw Error("cuda_rect_overlay: fit must be stretch or contain");
    out.fit = fit == "contain";
    out.source_canvas_w = out.source_canvas_h = 0;
    if (obj.contains("source_canvas")) {
        const auto &canvas = obj.at("source_canvas");
        out.source_canvas_w = canvas.at("w").get<int>();
        out.source_canvas_h = canvas.at("h").get<int>();
        if (!out.fit || out.dst_w <= 0 || out.dst_h <= 0 || out.source_canvas_w <= 0 || out.source_canvas_h <= 0)
            throw Error("cuda_rect_overlay: source_canvas requires positive dimensions and fit=contain");
    }
    if (out.dst_w < 0 || out.dst_h < 0 || (out.dst_w == 0) != (out.dst_h == 0))
        throw Error("cuda_rect_overlay: dst_w and dst_h must both be positive or both omitted");
#ifndef HAVE_CUDA_RECT_SCALE
    if (out.dst_w || out.dst_h)
        throw Error("cuda_rect_overlay: destination sizing requires HAVE_NVCC=1");
#endif
    if (obj.contains("crop") && obj["crop"].is_object()) {
        const auto &c = obj["crop"];
        out.crop_x = c.value("x", 0);
        out.crop_y = c.value("y", 0);
        // 0 means "use remaining source width/height from crop_x/crop_y" (resolved per-frame in resolveDrawOps).
        out.crop_w = c.value("w", 0);
        out.crop_h = c.value("h", 0);
    }
    // No crop object → crop_x/y/w/h all stay 0 (full source frame from origin).
}

inline std::vector<LayerSpec> parseLayersArray(const Parameters &arr) {
    std::vector<LayerSpec> layers;
    if (!arr.is_array())
        throw Error("cuda_rect_overlay: layers must be an array");
    for (const auto &item : arr) {
        if (!item.is_object())
            throw Error("cuda_rect_overlay: layers entries must be objects");
        LayerSpec s;
        parseLayerFromJson(item, s);
        layers.push_back(s);
    }
    return layers;
}

inline std::vector<LayerSpec> parseLayersParam(const Parameters &params) {
    if (!params.contains("layers") || !params["layers"].is_array())
        throw Error("cuda_rect_overlay: layers array required (one entry per input in src order)");
    return parseLayersArray(params["layers"]);
}

/// Apply a per-frame metadata override (JSON text) to `layers`: either {"layers": [...]} in src
/// order or {"<index>": {...}} per input. Throws on malformed JSON or entries.
inline void applyLayerMetadata(std::vector<LayerSpec> &layers, const char *json) {
    Parameters md = Parameters::parse(json);
    if (md.contains("layers") && md["layers"].is_array()) {
        const auto &arr = md["layers"];
        for (size_t i = 0; i < arr.size() && i < layers.size(); ++i) {
            if (!arr[i].is_object())
                continue;
            parseLayerFromJson(arr[i], layers[i]);
        }
    } else {
        for (size_t i = 0; i < layers.size(); ++i) {
            const std::string k = std::to_string(i);
            if (md.contains(k) && md[k].is_object())
                parseLayerFromJson(md[k], layers[i]);
        }
    }
}

/// Resolve each layer against its source frame: crop/destination clipping, chroma alignment,
/// fit/contain placement, then z-order. Missing sources and empty rectangles yield ops with
/// src == nullptr; a placement rejected by geometry keeps a negative src_w so the log can name it.
inline std::vector<DrawOp> resolveDrawOps(const std::vector<const av::VideoFrame *> &sources,
                                          const std::vector<LayerSpec> &layers,
                                          int canvas_w, int canvas_h, AVPixelFormat canvas_fmt) {
    std::vector<DrawOp> ops;
    ops.reserve(std::min(sources.size(), layers.size()));
    for (size_t i = 0; i < sources.size() && i < layers.size(); ++i) {
        const av::VideoFrame *srcp = sources[i];
        if (!srcp || !srcp->raw()) {
            ops.push_back({});
            continue;
        }
        LayerSpec L = layers[i];
        if (L.dst_w > 0) {
            const Rect crop{L.crop_x, L.crop_y, L.crop_w, L.crop_h};
            const Rect box{L.dst_x, L.dst_y, L.dst_w, L.dst_h};
            const int ax = chromaXAlign(canvas_fmt), ay = chromaYAlign(canvas_fmt);
            auto placement = L.source_canvas_w > 0
                ? placeInCanvas(srcp->width(), srcp->height(), crop, box,
                                L.source_canvas_w, L.source_canvas_h, ax, ay)
                : place(srcp->width(), srcp->height(), crop, box, L.fit, ax, ay);
            if (!placement || placement->destination.x >= canvas_w ||
                placement->destination.y >= canvas_h ||
                int64_t(placement->destination.x) + placement->destination.w <= 0 ||
                int64_t(placement->destination.y) + placement->destination.h <= 0) {
                DrawOp rejected;
                rejected.src_w = -srcp->width(); rejected.src_h = srcp->height();   // marks "rejected" in the log
                rejected.layer = L;
                ops.push_back(rejected);
                continue;
            }
            const auto &p = *placement;
            L.crop_x = p.source.x; L.crop_y = p.source.y;
            L.crop_w = p.source.w; L.crop_h = p.source.h;
            L.dst_x = p.destination.x; L.dst_y = p.destination.y;
            L.dst_w = p.destination.w; L.dst_h = p.destination.h;
            ops.push_back({srcp, srcp->width(), srcp->height(), L});
            continue;
        }
        // 0 means "remaining source extent from the crop origin".
        if (L.crop_w <= 0) L.crop_w = srcp->width()  - L.crop_x;
        if (L.crop_h <= 0) L.crop_h = srcp->height() - L.crop_y;
        if (!clipRect(L.crop_x, L.crop_y, L.crop_w, L.crop_h, srcp->width(), srcp->height()) ||
            !clipRect(L.dst_x, L.dst_y, L.crop_w, L.crop_h, canvas_w, canvas_h)) {
            ops.push_back({});
            continue;
        }
        const int ax = chromaXAlign(canvas_fmt);
        const int ay = chromaYAlign(canvas_fmt);
        L.crop_x = alignCoord(L.crop_x, ax);
        L.crop_y = alignCoord(L.crop_y, ay);
        L.dst_x = alignCoord(L.dst_x, ax);
        L.dst_y = alignCoord(L.dst_y, ay);
        if (!clipRect(L.crop_x, L.crop_y, L.crop_w, L.crop_h, srcp->width(), srcp->height()) ||
            !clipRect(L.dst_x, L.dst_y, L.crop_w, L.crop_h, canvas_w, canvas_h)) {
            ops.push_back({});
            continue;
        }
        ops.push_back({srcp, srcp->width(), srcp->height(), L});
    }
    // z decides who draws on top; equal z keeps source order (stable).
    std::stable_sort(ops.begin(), ops.end(),
                     [](const DrawOp &a, const DrawOp &b) { return a.layer.z < b.layer.z; });
    return ops;
}

/// One-line description of a resolved op list, for change-triggered logging.
inline std::string describeDrawOps(const std::vector<DrawOp> &ops) {
    std::ostringstream desc;
    for (size_t i = 0; i < ops.size(); ++i) {
        const DrawOp &op = ops[i];
        const LayerSpec &L = op.layer;
        if (!op.src) {
            if (op.src_w < 0)
                desc << " [" << i << ":REJECTED src " << -op.src_w << "x" << op.src_h << " crop " << L.crop_x
                     << "," << L.crop_y << " " << L.crop_w << "x" << L.crop_h << " box " << L.dst_x << ","
                     << L.dst_y << " " << L.dst_w << "x" << L.dst_h << "]";
            continue;
        }
        desc << " [" << i << ":" << op.src_w << "x" << op.src_h << " crop " << L.crop_x << "," << L.crop_y
             << " " << L.crop_w << "x" << L.crop_h << " -> " << L.dst_x << "," << L.dst_y << " "
             << L.dst_w << "x" << L.dst_h << " z" << L.z << "]";
    }
    return desc.str();
}

}
