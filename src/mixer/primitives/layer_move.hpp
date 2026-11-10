#pragma once
// Move transition: Program's scene A glides into scene B on B's slot, in the slot's one launch.
//
// Layers are matched by source (the k-th layer of a source in A with the k-th in B). A matched
// layer is placed at both ends against its frame (contain and source_canvas resolved there). Its
// camera picture then moves as one picture: the whole picture's place on the canvas glides
// geometrically from A's to B's (zoomRect) and the box glides on its own, clipping it, so nothing
// pans or zooms inside the box. A face sheet's tiles are framings of one picture: through the
// frame's per-tile layers (layer_zoom.hpp PictureMap) both ends are read in the camera picture and
// interior frames draw from its reference tile, never a neighbouring one; without those layers the
// pair fades. Its rotation takes the shortest arc and its z is a float key: overlapping layers swap
// where their keys cross. Interior frames are fractional (crop_exact, dst_exact) and draw through
// the affine entry. A layer in only one scene
// fades (its opacity, the affine entry's transparency for a camera) or cuts at the half, stacked
// right above the gliding layer under it in its own scene (at the bottom without one), so it never
// jumps over or under a gliding layer half-way through its fade. Pure: the
// slot reads progress from the take clock and calls moveLayers per tick.
#include "TakeClock.hpp"
#include "layer_zoom.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace avp::mixer {

struct MoveSpec {
    uint64_t generation = 0;        // the take that runs it
    std::vector<LayerSpec> from;    // scene A's layers
    Parameters from_json;           // as received: the round trip without a LayerSpec serializer
    std::string unmatched = "fade"; // "fade" | "cut"
    std::map<int, SourcePicture> pictures;   // by input: sources that are face sheets of one picture
};

/// The face sheets a move draws from: what each declares, and the per-tile layers last read from
/// its frames.
struct MovePictures {
    std::map<int, SourcePicture> declared;
    std::map<int, std::vector<PictureMap>> maps;
};

inline MoveSpec parseMoveSpec(const Parameters &p, ScaleFilter default_filter = ScaleFilter::Bilinear) {
    MoveSpec m;
    m.generation = p.value("generation", uint64_t(0));
    m.from_json = p.at("from");
    m.from = parseLayersArray(m.from_json, default_filter);
    m.unmatched = p.value("unmatched", std::string("fade"));
    if (m.unmatched != "fade" && m.unmatched != "cut") throw Error("move: unmatched must be fade or cut");
    if (p.contains("pictures"))
        for (const auto &e : p.at("pictures")) {
            const auto &pic = e.at("picture"), &r = pic.at("reference");
            m.pictures[e.at("input").get<int>()] = SourcePicture{{r.at("x"), r.at("y"), r.at("w"), r.at("h")},
                                                                 pic.at("metadata_key").get<std::string>(),
                                                                 pic.value("mapped", std::vector<int>{})};
        }
    return m;
}

inline Parameters moveSpecJson(const MoveSpec &m) {
    Parameters out{{"generation", m.generation}, {"from", m.from_json}, {"unmatched", m.unmatched}};
    if (!m.pictures.empty()) {
        Parameters list = Parameters::array();
        for (const auto &[input, pic] : m.pictures) {
            const Rect &r = pic.reference;
            list.push_back({{"input", input}, {"picture", {{"reference", {{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}}},
                                                           {"metadata_key", pic.metadata_key}, {"mapped", pic.mapped}}}});
        }
        out["pictures"] = std::move(list);
    }
    return out;
}

/// A layer placed against its frame: the visible-picture crop drawn into the canvas box.
struct MovePose {
    RectF crop, box;
    double rotation = 0;
    std::optional<std::pair<double, double>> pivot;
};

/// Where `l` draws from `src`, as resolveDrawOps places it; nullopt for a tile layer (AUX
/// multiview tiles do not glide), a missing frame or an empty placement.
inline std::optional<MovePose> placedPose(const LayerSpec &l, const av::VideoFrame *src, AVPixelFormat fmt) {
    if (!src || !src->raw() || l.tile.w) return std::nullopt;
    const Rect visible = visibleFrameRect(*src->raw());
    const auto f = [](const Rect &r) { return RectF{double(r.x), double(r.y), double(r.w), double(r.h)}; };
    MovePose p;
    p.rotation = normalizedRotation(l.rotation_deg);
    if (l.pivot_set) p.pivot = std::make_pair(l.pivot_x, l.pivot_y);
    if (l.dst_w > 0) {
        const Rect crop{l.crop_x, l.crop_y, l.crop_w, l.crop_h}, box{l.dst_x, l.dst_y, l.dst_w, l.dst_h};
        const int ax = chromaXAlign(fmt), ay = chromaYAlign(fmt);
        const auto placed = l.source_canvas_w > 0
            ? placeInCanvas(visible.w, visible.h, crop, box, l.source_canvas_w, l.source_canvas_h, ax, ay)
            : place(visible.w, visible.h, crop, box, l.fit, ax, ay);
        if (!placed) return std::nullopt;
        const bool exact = l.crop_exact && !l.fit && !l.source_canvas_w;   // a stretched fractional crop draws as given
        p.crop = exact ? *l.crop_exact : f(placed->source);
        p.box = f(placed->destination);
    } else {
        const double w = l.crop_w > 0 ? l.crop_w : visible.w - l.crop_x, h = l.crop_h > 0 ? l.crop_h : visible.h - l.crop_y;
        p.crop = l.crop_exact ? *l.crop_exact : RectF{double(l.crop_x), double(l.crop_y), w, h};
        p.box = {double(l.dst_x), double(l.dst_y), p.crop.w, p.crop.h};
    }
    if (!(p.box.w > 0 && p.box.h > 0 && p.crop.w > 0 && p.crop.h > 0)) return std::nullopt;
    return p;
}

/// The layers drawn at progress `s` of the move from `from` to `to`; `s <= 0` is `from` and
/// `s >= 1` is `to`, verbatim. Every layer out names its input and has z = its position.
inline std::vector<LayerSpec> moveLayers(const std::vector<LayerSpec> &from, const std::vector<LayerSpec> &to, double s,
                                         const std::vector<const av::VideoFrame *> &sources, AVPixelFormat fmt,
                                         const std::string &unmatched, const MovePictures *pictures = nullptr) {
    if (!(s > 0)) return from;
    if (s >= 1) return to;
    const auto frame = [&](size_t input) { return input < sources.size() ? sources[input] : nullptr; };
    // Stacked by (key, tier, own): a gliding layer is (its lerped z, 0, 0); a layer in one scene only
    // is (the key of the gliding layer under it in its scene, or -inf, 1, its z). `weight` < 0: drawn
    // as is; else faded by its scene's share (1 - s for A, s for B) once the stacking is known; `box`
    // its placed box (none: unplaced).
    struct Item {
        double key; LayerSpec layer; double weight = -1; bool from_a = false; std::optional<RectF> box;
        int tier = 0; double own = 0;
    };
    std::vector<Item> items;
    const bool cut = unmatched == "cut";
    std::vector<int> glide_a(from.size(), -1), glide_b(to.size(), -1);   // the item gliding each layer
    // The key of the gliding layer right under scene[k] in its scene's stacking (z, then position).
    const auto under = [&](const std::vector<LayerSpec> &scene, const std::vector<int> &glide, size_t k) {
        double key = -std::numeric_limits<double>::infinity();
        std::optional<std::pair<int, size_t>> best;
        for (size_t m = 0; m < scene.size(); ++m) {
            const auto pos = std::make_pair(scene[m].z, m);
            if (glide[m] >= 0 && pos < std::make_pair(scene[k].z, k) && (!best || pos > *best)) {
                best = pos;
                key = items[size_t(glide[m])].key;
            }
        }
        return key;
    };
    const auto alone = [&](size_t k, bool from_a) {
        const auto &scene = from_a ? from : to;
        LayerSpec l = scene[k];
        const size_t input = l.sourceIndex(k);
        if (cut ? (from_a ? s >= 0.5 : s < 0.5) : false) return;
        l.input = int(input);
        std::optional<RectF> box;
        if (const auto pose = placedPose(l, frame(input), fmt)) box = pose->box;
        const double key = under(scene, from_a ? glide_a : glide_b, k), own = l.z;
        items.push_back({key, std::move(l), cut ? -1. : (from_a ? 1 - s : s), from_a, box, 1, own});
    };
    std::vector<bool> used(from.size(), false);
    std::vector<bool> placed_b(to.size(), false), alone_pair(to.size(), false);
    for (size_t j = 0; j < to.size(); ++j) {
        const size_t input = to[j].sourceIndex(j);
        size_t i = 0;
        while (i < from.size() && (used[i] || from[i].sourceIndex(i) != input)) ++i;
        if (i == from.size()) continue;
        used[i] = placed_b[j] = true;
        const av::VideoFrame *src = frame(input);
        if (!src || !src->raw()) continue;   // nothing to draw either end from
        const auto a = placedPose(from[i], src, fmt), b = placedPose(to[j], src, fmt);
        // The picture both ends show, and each end's crop of it in frame pixels: the visible frame,
        // or a face sheet's reference tile with the ends read through the frame's tile layers.
        const Rect visible = visibleFrameRect(*src->raw());
        RectF ref{double(visible.x), double(visible.y), double(visible.w), double(visible.h)};
        std::optional<RectF> ca, cb;
        if (a && b) { ca = a->crop; cb = b->crop; }
        const SourcePicture *declared = nullptr;
        if (pictures)
            if (const auto it = pictures->declared.find(int(input)); it != pictures->declared.end()) declared = &it->second;
        if (ca && declared) {
            const Rect &r = declared->reference;
            const auto maps = pictures->maps.find(int(input));
            const auto whole = [](const RectF &c) {
                return Rect{int(std::lround(c.x)), int(std::lround(c.y)), int(std::lround(c.w)), int(std::lround(c.h))};
            };
            const auto pa = maps != pictures->maps.end() ? pictureRect(maps->second, whole(*ca)) : std::nullopt;
            const auto pb = maps != pictures->maps.end() ? pictureRect(maps->second, whole(*cb)) : std::nullopt;
            if (pa && pb) {
                ref = {double(r.x), double(r.y), double(r.w), double(r.h)};
                ca = RectF{r.x + pa->first.x, r.y + pa->first.y, pa->first.w, pa->first.h};
                cb = RectF{r.x + pb->first.x, r.y + pb->first.y, pb->first.w, pb->first.h};
            } else {
                ca.reset();   // tiles without their framings: never slide across the sheet
            }
        }
        if (!ca) {   // a tile or an unplaceable end, or a sheet without framings: the two fade (or cut)
            used[i] = false;
            alone_pair[j] = true;
            continue;
        }
        // Where the whole picture sits on the canvas at each end, gliding as one; the box clips it.
        const auto onCanvas = [&](const RectF &c, const RectF &box) {
            const double kx = box.w / c.w, ky = box.h / c.h;
            return RectF{box.x - (c.x - ref.x) * kx, box.y - (c.y - ref.y) * ky, ref.w * kx, ref.h * ky};
        };
        const RectF pic = zoomRect(onCanvas(*ca, a->box), onCanvas(*cb, b->box), s);
        RectF bx = zoomRect(a->box, b->box, s);
        const double kx = pic.w / ref.w, ky = pic.h / ref.h;
        // The box's crop of the picture, kept inside it; the box loses the same share.
        const double x0 = std::max(ref.x + (bx.x - pic.x) / kx, ref.x), y0 = std::max(ref.y + (bx.y - pic.y) / ky, ref.y);
        const double x1 = std::min(ref.x + (bx.x + bx.w - pic.x) / kx, ref.x + ref.w);
        const double y1 = std::min(ref.y + (bx.y + bx.h - pic.y) / ky, ref.y + ref.h);
        if (!(x1 > x0) || !(y1 > y0)) {   // the box left the picture: nothing of it to draw this tick
            used[i] = false;
            alone_pair[j] = true;
            continue;
        }
        bx = {pic.x + (x0 - ref.x) * kx, pic.y + (y0 - ref.y) * ky, (x1 - x0) * kx, (y1 - y0) * ky};
        LayerSpec l = to[j];
        l.input = int(input);
        l.fit = false;
        l.source_canvas_w = l.source_canvas_h = 0;
        setLayerCrop(l, {x0, y0, x1 - x0, y1 - y0});
        l.dst_exact = bx;
        l.dst_x = int(std::floor(bx.x));
        l.dst_y = int(std::floor(bx.y));
        l.dst_w = int(std::ceil(bx.x + bx.w)) - l.dst_x;
        l.dst_h = int(std::ceil(bx.y + bx.h)) - l.dst_y;
        double turn = b->rotation - a->rotation;
        if (turn > 180) turn -= 360;
        else if (turn <= -180) turn += 360;
        l.rotation_deg = normalizedRotation(a->rotation + turn * s);
        l.pivot_set = a->pivot || b->pivot;
        if (l.pivot_set) {
            const auto centre = [](const MovePose &p) {
                return p.pivot ? *p.pivot : std::make_pair(p.box.x + p.box.w / 2, p.box.y + p.box.h / 2);
            };
            const auto pa = centre(*a), pb = centre(*b);
            l.pivot_x = pa.first + (pb.first - pa.first) * s;
            l.pivot_y = pa.second + (pb.second - pa.second) * s;
        }
        l.opacity = float(from[i].opacity + (to[j].opacity - from[i].opacity) * s);
        glide_a[i] = glide_b[j] = int(items.size());
        items.push_back({from[i].z + (to[j].z - from[i].z) * s, std::move(l), -1., false, std::nullopt});
    }
    for (size_t j = 0; j < to.size(); ++j)
        if (!placed_b[j] || alone_pair[j]) alone(j, false);
    for (size_t i = 0; i < from.size(); ++i)
        if (!used[i]) alone(i, true);
    std::stable_sort(items.begin(), items.end(), [](const Item &x, const Item &y) {
        return std::tie(x.key, x.tier, x.own) < std::tie(y.key, y.tier, y.own);
    });
    // A fading layer wholly under an opaque fading layer of the other scene stays whole: the upper
    // one alone carries the mix, as a fade's (1-s)A + sB, never through black.
    const auto covers = [](const RectF &o, const RectF &u) {
        return o.x <= u.x && o.y <= u.y && o.x + o.w >= u.x + u.w && o.y + o.h >= u.y + u.h;
    };
    for (size_t i = 0; i < items.size(); ++i) {
        Item &u = items[i];
        if (u.weight < 0) continue;
        bool whole = false;
        for (size_t j = i + 1; j < items.size() && !whole && u.box; ++j) {
            const Item &o = items[j];
            whole = o.weight >= 0 && o.from_a != u.from_a && !o.layer.blend && o.layer.opacity >= 1.f && !o.layer.rotation_deg &&
                    o.box && covers(*o.box, *u.box);
        }
        if (!whole) u.layer.opacity = float(u.layer.opacity * u.weight);
    }
    items.erase(std::remove_if(items.begin(), items.end(), [](const Item &it) {
        return !(it.layer.opacity > 0) || (it.layer.dst_exact && !(it.layer.dst_exact->w > 0 && it.layer.dst_exact->h > 0));
    }), items.end());
    std::vector<LayerSpec> out;
    out.reserve(items.size());
    for (auto &it : items) {
        it.layer.z = int(out.size());
        out.push_back(std::move(it.layer));
    }
    return out;
}

/// A scene slot's move, on its render thread: set with the composition that carries it. Each tick
/// draws scene A at progress 0 until the move's take is on air, then at the published take clock's
/// progress; the move is dropped at the take's end (completed or cancelled) or once a later take
/// is on air, and the slot draws its own scene again.
struct SlotMove {
    std::optional<MoveSpec> spec;
    double drawn = 0;     // the progress the last tick drew
    std::string error;    // why the last tick drew the scene's own layers
    MovePictures pictures;   // the sheets' framings, held across ticks whose frame lacks them

    void set(std::optional<MoveSpec> m) {
        spec = std::move(m);
        drawn = 0;
        error.clear();
        pictures = {};
        if (spec) pictures.declared = spec->pictures;
    }
    /// Rewrites `layers` (scene B's) for the tick at `pts_ns`; false, with them unchanged, when no
    /// move runs. `clock`: the mixer's published take clock, nullptr before the first take.
    bool apply(std::vector<LayerSpec> &layers, const TakeClock *clock, int64_t pts_ns,
               const std::vector<const av::VideoFrame *> &sources, AVPixelFormat fmt) {
        if (!spec) return false;
        double s = 0;
        if (clock && clock->generation > spec->generation) {
            spec.reset();
            return false;
        }
        if (clock && clock->generation == spec->generation) {
            if (clock->end != TakeClock::End::Running && pts_ns >= clock->end_ns) {
                spec.reset();
                return false;
            }
            s = clock->progress(pts_ns);
        }
        drawn = s;
        for (const auto &[input, pic] : pictures.declared) {
            const av::VideoFrame *f = input >= 0 && size_t(input) < sources.size() ? sources[size_t(input)] : nullptr;
            const AVDictionaryEntry *e = f && f->raw() ? av_dict_get(f->raw()->metadata, pic.metadata_key.c_str(), nullptr, 0) : nullptr;
            if (!e || !e->value) continue;
            auto maps = parsePictureMaps(e->value, pic);
            if (!maps.empty()) pictures.maps[input] = std::move(maps);
        }
        layers = moveLayers(spec->from, layers, s, sources, fmt, spec->unmatched, &pictures);
        return true;
    }
};

}
