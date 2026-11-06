#pragma once
// Picture-mapped layer zoom: one scene layer glides between two framings of the same camera.
//
// A tracked camera arrives as a sheet: its whole picture 1:1 in a reference rect (the raw tile)
// and framed tiles drawn from it, each described per frame by a layer (crop in picture pixels, dst
// on the sheet) under the source's metadata key. A scene layer's sheet crop therefore names a rect
// of the picture. During a zoom the layer draws, from the reference rect, the rect gliding between
// the from and the to framing's picture rects, both re-read every tick, so tracking continues.
// Scale is geometric about the fixed point of the two rects: every edge moves monotonically and
// the apparent zoom speed is uniform. Interior frames are sub-pixel (LayerSpec::crop_exact, drawn
// by the affine entry); at progress 0 and 1 the rects are the tiles' own. No AI here: the engine
// knows only the layer schema it parses and the metadata key it is given.
#include "compositor_layers.hpp"
#include "fade_curve.hpp"
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/dict.h>
}

namespace avp::mixer {

using SourceRect = RectF;

/// One mapped per-frame layer of a sheet: the picture rect `crop` drawn into the sheet rect `dst`.
struct PictureMap {
    Rect dst;
    SourceRect crop;
    FilterSpec filter;
};

/// What a source declares for zooms: where its picture sits 1:1 on its frames, which metadata key
/// carries the framings, and which of those layers (by input) are framings of that picture.
struct SourcePicture {
    Rect reference;
    std::string metadata_key;
    std::vector<int> mapped;   // empty: all
};

/// One zoom of one layer: from a framing (or a running zoom) to the layer's target framing.
struct ZoomSpec {
    int input = -1;
    SourcePicture picture;
    Rect from_crop, from_dst;                  // the from scene layer: sheet crop and canvas dst
    std::shared_ptr<const ZoomSpec> from_zoom;  // or a zoom still running when this one began
    Rect to_crop, to_dst;                      // the target scene layer
    int64_t duration_ns = 0;
    FadeCurve curve = FadeCurve::EaseInOut;
    uint64_t generation = 0;
    std::optional<int64_t> start_ns;           // the first on-air frame's pts; progress 0 until set
};

/// The last framings read, held across ticks whose frame lacks the metadata.
struct ZoomFrameState {
    std::vector<PictureMap> maps;
    uint64_t held = 0;
};

/// Where a zoom is at one tick: the picture rect, the canvas dst and the filter it draws with.
struct ZoomPose {
    SourceRect picture;
    RectF dst;
    FilterSpec filter;
};

inline std::vector<PictureMap> parsePictureMaps(const char *json, const SourcePicture &picture) {
    const Parameters md = Parameters::parse(json);
    std::vector<PictureMap> maps;
    if (!md.contains("layers") || !md["layers"].is_array()) return maps;
    const auto &layers = md["layers"];
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i].is_object()) continue;
        const int input = layers[i].value("input", int(i));
        if (!picture.mapped.empty() && std::find(picture.mapped.begin(), picture.mapped.end(), input) == picture.mapped.end())
            continue;
        LayerSpec l;
        parseLayerFromJson(layers[i], l);
        if (l.dst_w <= 0 || l.dst_h <= 0 || l.crop_w <= 0 || l.crop_h <= 0) continue;
        const SourceRect crop = l.crop_exact ? *l.crop_exact
            : SourceRect{double(l.crop_x), double(l.crop_y), double(l.crop_w), double(l.crop_h)};
        maps.push_back({{l.dst_x, l.dst_y, l.dst_w, l.dst_h}, crop, l.filter});
    }
    return maps;
}

/// The picture rect a sheet crop shows, through the mapped layer it overlaps most (it may reach
/// outside the picture where the tile is letterboxed); nullopt when it overlaps none.
inline std::optional<std::pair<SourceRect, const PictureMap *>> pictureRect(const std::vector<PictureMap> &maps,
                                                                            Rect sheet_crop) {
    const PictureMap *best = nullptr;
    int64_t best_area = 0;
    for (const PictureMap &m : maps) {
        const int64_t w = std::min(sheet_crop.x + sheet_crop.w, m.dst.x + m.dst.w) - std::max(sheet_crop.x, m.dst.x);
        const int64_t h = std::min(sheet_crop.y + sheet_crop.h, m.dst.y + m.dst.h) - std::max(sheet_crop.y, m.dst.y);
        if (w > 0 && h > 0 && w * h > best_area) { best = &m; best_area = w * h; }
    }
    if (!best) return std::nullopt;
    const double kx = best->crop.w / best->dst.w, ky = best->crop.h / best->dst.h;
    return std::make_pair(SourceRect{best->crop.x + (sheet_crop.x - best->dst.x) * kx,
                                     best->crop.y + (sheet_crop.y - best->dst.y) * ky,
                                     sheet_crop.w * kx, sheet_crop.h * ky}, best);
}

/// The rect at `s` in [0, 1] from `from` to `to`: geometric scale about the two rects' fixed point
/// per axis, linear when the sizes are within 0.1 %.
inline SourceRect zoomRect(const SourceRect &from, const SourceRect &to, double s) {
    auto axis = [s](double a, double aw, double b, double bw, double &x, double &w) {
        const double r = bw / aw;
        if (std::fabs(std::log(r)) < 1e-3) {
            x = a + (b - a) * s;
            w = aw + (bw - aw) * s;
            return;
        }
        const double fixed = (b - r * a) / (1 - r), k = std::pow(r, s);
        x = fixed + (a - fixed) * k;
        w = aw * k;
    };
    SourceRect out;
    axis(from.x, from.w, to.x, to.w, out.x, out.w);
    axis(from.y, from.h, to.y, to.h, out.y, out.h);
    return out;
}

/// Eased progress at `pts_ns`: 0 until the zoom is on air (start_ns) and before its start, 1 from
/// its end.
inline double zoomProgress(const ZoomSpec &z, int64_t pts_ns) {
    if (!z.start_ns || pts_ns <= *z.start_ns) return 0.;
    if (z.duration_ns <= 0 || pts_ns - *z.start_ns >= z.duration_ns) return 1.;
    return fadeCurveAt(z.curve, double(pts_ns - *z.start_ns) / double(z.duration_ns));
}

/// Where `z` is at `pts_ns` with these framings; a nested from-zoom follows its own schedule and,
/// once ended, stands at its target. Nullopt when a sheet crop shows no mapped framing.
inline std::optional<ZoomPose> zoomPose(const ZoomSpec &z, const std::vector<PictureMap> &maps, int64_t pts_ns) {
    const auto to = pictureRect(maps, z.to_crop);
    if (!to) return std::nullopt;
    std::optional<ZoomPose> from;
    if (z.from_zoom) from = zoomPose(*z.from_zoom, maps, pts_ns);
    else if (const auto f = pictureRect(maps, z.from_crop))
        from = ZoomPose{f->first, {double(z.from_dst.x), double(z.from_dst.y), double(z.from_dst.w), double(z.from_dst.h)},
                        f->second->filter};
    if (!from) return std::nullopt;
    const double s = zoomProgress(z, pts_ns);
    const RectF to_dst{double(z.to_dst.x), double(z.to_dst.y), double(z.to_dst.w), double(z.to_dst.h)};
    const auto lerp = [s](double a, double b) { return a + (b - a) * s; };
    return ZoomPose{zoomRect(from->picture, to->first, s),
                    {lerp(from->dst.x, to_dst.x), lerp(from->dst.y, to_dst.y), lerp(from->dst.w, to_dst.w),
                     lerp(from->dst.h, to_dst.h)},
                    to->second->filter};
}

/// Writes `r` as the layer's crop: whole pixels as today, a fraction as crop_exact (the affine entry).
inline void setLayerCrop(LayerSpec &l, const SourceRect &r) {
    const bool whole = r.x == std::floor(r.x) && r.y == std::floor(r.y) && r.w == std::floor(r.w) && r.h == std::floor(r.h);
    if (whole) l.crop_exact.reset();
    else l.crop_exact = r;
    l.crop_x = int(std::floor(r.x));
    l.crop_y = int(std::floor(r.y));
    l.crop_w = int(std::ceil(r.x + r.w)) - l.crop_x;
    l.crop_h = int(std::ceil(r.y + r.h)) - l.crop_y;
}

/// Rewrites the layer drawing `z.input` for the tick at `pts_ns` from `source`'s framings. False,
/// with the layers unchanged, once the zoom is over or when there is nothing to draw it from (then
/// `why` says so). A tick without the metadata holds the last framings read (state.held counts it).
inline bool applyLayerZoom(std::vector<LayerSpec> &layers, const ZoomSpec &z, const av::VideoFrame *source, int64_t pts_ns,
                           ZoomFrameState &state, std::string *why) {
    auto layer = std::find_if(layers.begin(), layers.end(), [&](const LayerSpec &l) { return l.input == z.input; });
    if (layer == layers.end()) {
        if (why) *why = "no layer draws input " + std::to_string(z.input);
        return false;
    }
    if (zoomProgress(z, pts_ns) >= 1.) return false;
    const AVDictionaryEntry *e = source && source->raw()
        ? av_dict_get(source->raw()->metadata, z.picture.metadata_key.c_str(), nullptr, 0) : nullptr;
    std::vector<PictureMap> maps = e && e->value ? parsePictureMaps(e->value, z.picture) : std::vector<PictureMap>{};
    if (!maps.empty()) state.maps = std::move(maps);
    else if (!state.maps.empty()) ++state.held;
    else {
        if (why) *why = "no " + z.picture.metadata_key + " framings on the source frame";
        return false;
    }
    const auto pose = zoomPose(z, state.maps, pts_ns);
    if (!pose) {
        if (why) *why = "a zoom endpoint shows no mapped tile";
        return false;
    }
    // Picture to sheet, kept inside the reference (never a neighbouring tile); the destination
    // loses the same share.
    const Rect &ref = z.picture.reference;
    const SourceRect crop{ref.x + pose->picture.x, ref.y + pose->picture.y, pose->picture.w, pose->picture.h};
    const double x0 = std::max(crop.x, double(ref.x)), y0 = std::max(crop.y, double(ref.y));
    const double x1 = std::min(crop.x + crop.w, double(ref.x) + ref.w), y1 = std::min(crop.y + crop.h, double(ref.y) + ref.h);
    if (!(x1 > x0) || !(y1 > y0)) {
        if (why) *why = "the zoom left the picture";
        return false;
    }
    const RectF &d = pose->dst;
    const double kx = d.w / crop.w, ky = d.h / crop.h;
    LayerSpec &l = *layer;
    setLayerCrop(l, {x0, y0, x1 - x0, y1 - y0});
    l.dst_x = int(std::lround(d.x + (x0 - crop.x) * kx));
    l.dst_y = int(std::lround(d.y + (y0 - crop.y) * ky));
    l.dst_w = int(std::lround(d.x + (x1 - crop.x) * kx)) - l.dst_x;
    l.dst_h = int(std::lround(d.y + (y1 - crop.y) * ky)) - l.dst_y;
    l.fit = false;
    l.source_canvas_w = l.source_canvas_h = 0;
    l.filter = pose->filter;
    if (l.filter.mode == ScaleFilter::Auto) l.filter.mode = ScaleFilter::Bicubic;   // never switch kernels mid-glide
    l.rotation_deg = 0;
    l.pivot_set = false;
    return true;
}

/// A scene slot's zoom, on its render thread: set with the composition that carries it, started by
/// the take's on-air message, applied every tick, dropped at its end.
struct SlotZoom {
    std::optional<ZoomSpec> spec;
    ZoomFrameState state;
    std::string error;   // why the last tick drew the scene's own layer instead

    void set(std::optional<ZoomSpec> z) {
        spec = std::move(z);
        state = {};
        error.clear();
    }
    /// The take of `generation` went on air with its first frame at `start_ns`.
    void onAir(uint64_t generation, int64_t start_ns) {
        if (spec && spec->generation == generation && !spec->start_ns) spec->start_ns = start_ns;
    }
    double progress(int64_t pts_ns) const { return spec ? zoomProgress(*spec, pts_ns) : 1.; }
    /// Rewrites the zooming layer for this tick; false (and the zoom dropped once it is over) when
    /// the layers stay the scene's.
    bool apply(std::vector<LayerSpec> &layers, const std::vector<const av::VideoFrame *> &sources, int64_t pts_ns) {
        if (!spec) return false;
        if (zoomProgress(*spec, pts_ns) >= 1.) {
            spec.reset();
            return false;
        }
        const av::VideoFrame *source = spec->input >= 0 && size_t(spec->input) < sources.size() ? sources[spec->input] : nullptr;
        std::string why;
        const bool drawn = applyLayerZoom(layers, *spec, source, pts_ns, state, &why);
        if (!why.empty()) error = why;
        return drawn;
    }
};

inline Parameters zoomSpecJson(const ZoomSpec &z) {
    const auto rect = [](const Rect &r) { return Parameters{{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}}; };
    Parameters out{{"input", z.input},
                   {"picture", {{"reference", rect(z.picture.reference)}, {"metadata_key", z.picture.metadata_key},
                                {"mapped", z.picture.mapped}}},
                   {"to", {{"crop", rect(z.to_crop)}, {"dst", rect(z.to_dst)}}},
                   {"duration_ns", z.duration_ns}, {"curve", fadeCurveName(z.curve)}, {"generation", z.generation}};
    if (z.from_zoom) out["from_zoom"] = zoomSpecJson(*z.from_zoom);
    else out["from"] = {{"crop", rect(z.from_crop)}, {"dst", rect(z.from_dst)}};
    if (z.start_ns) out["start_ns"] = *z.start_ns;
    return out;
}

inline ZoomSpec parseZoomSpec(const Parameters &p, int depth = 0) {
    if (depth > 1) throw Error("zoom: a glide may nest one running glide, not more");
    const auto rect = [](const Parameters &r) { return Rect{r.at("x"), r.at("y"), r.at("w"), r.at("h")}; };
    ZoomSpec z;
    z.input = p.at("input").get<int>();
    const auto &pic = p.at("picture");
    z.picture = {rect(pic.at("reference")), pic.at("metadata_key").get<std::string>(),
                 pic.value("mapped", std::vector<int>{})};
    z.to_crop = rect(p.at("to").at("crop"));
    z.to_dst = rect(p.at("to").at("dst"));
    if (p.contains("from_zoom")) z.from_zoom = std::make_shared<ZoomSpec>(parseZoomSpec(p.at("from_zoom"), depth + 1));
    else {
        z.from_crop = rect(p.at("from").at("crop"));
        z.from_dst = rect(p.at("from").at("dst"));
    }
    z.duration_ns = p.at("duration_ns").get<int64_t>();
    z.curve = parseFadeCurve(p.value("curve", std::string("ease-in-out")));
    z.generation = p.value("generation", uint64_t(0));
    if (p.contains("start_ns")) z.start_ns = p.at("start_ns").get<int64_t>();
    if (z.duration_ns <= 0 || z.picture.reference.w <= 0 || z.picture.reference.h <= 0)
        throw Error("zoom: a positive duration and a picture reference are required");
    return z;
}

/// A zoom from the scene layer `from_layer` to `to_layer` (scene layer JSON: crop and dst_*), or
/// nullopt with the reason: another source ("source"), no declared picture ("picture"), a layer
/// without a sheet crop ("tile").
inline std::optional<ZoomSpec> planZoom(const Parameters &from_layer, const Parameters &to_layer, int from_input, int to_input,
                                        const SourcePicture *picture, std::string &why) {
    if (from_input != to_input) { why = "the scenes draw another source"; return std::nullopt; }
    if (!picture) { why = "the source declares no picture"; return std::nullopt; }
    const auto sheet = [](const Parameters &l, Rect &crop, Rect &dst) {
        if (!l.contains("crop") || !l["crop"].is_object()) return false;
        const auto &c = l["crop"];
        crop = {c.value("x", 0), c.value("y", 0), c.value("w", 0), c.value("h", 0)};
        dst = {l.value("dst_x", 0), l.value("dst_y", 0), l.value("dst_w", 0), l.value("dst_h", 0)};
        return crop.w > 0 && crop.h > 0 && dst.w > 0 && dst.h > 0;
    };
    ZoomSpec z;
    if (!sheet(from_layer, z.from_crop, z.from_dst) || !sheet(to_layer, z.to_crop, z.to_dst)) {
        why = "a scene layer has no tile crop";
        return std::nullopt;
    }
    z.input = to_input;
    z.picture = *picture;
    return z;
}

}
