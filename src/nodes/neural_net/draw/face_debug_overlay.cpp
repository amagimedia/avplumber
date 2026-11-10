// The compositor's debug overlay (mixer/primitives/debug_overlay.hpp) for face cameras: per
// layer that asks for it, the face box and landmarks, the pose skeleton of each person, the look
// arrow, the reaction meter and a few short labels, read from the source frame's `face_debug_v1`
// metadata and drawn on the composed frame at the size the layer is shown, with ml_debug's
// renderer and kernel: every layer of the frame in one launch over the 16x16 tiles it touches.
//
// `face_debug_v1` (written by the face sheet's join) is in source pixels of the camera picture:
//   {"size": [w, h], "layers": "<metadata key>", "face": [x0, y0, x1, y1] | null,
//    "points": [x, y, ...], "people": [{"k": [x, y, v] * 17, "box": [x0, y0, x1, y1, conf] | null}],
//    "pose_min": 0.4, "look": -1..1 | null, "reaction": 0..100 | null, "labels": ["...", ...]}
// The frame may be a sheet of several framings of that picture. The key `layers` names says where
// each is: {"layers": [{"crop": {x, y, w, h}, "dst_x", "dst_y", "dst_w", "dst_h"}, ...]}, the
// crop of the picture and its place on the frame. A point goes through the sheet layer whose
// place holds the centre of what the compositor's layer shows, then through that layer. Without
// the key the frame is the picture itself.
//
// The model payload changes with the face sample (5-15 a second) and is parsed once per string;
// the framing moves every frame, so the items are placed anew for each frame, on the CPU.
#include "ml_debug_renderer.hpp"

#include "../../../mixer/primitives/debug_overlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include "../../../../objs/src/nodes/neural_net/draw/ml_debug.ptx.h"

namespace {

using namespace cuda_overlay;

constexpr const char* kModelKey = "face_debug_v1";
constexpr int kPosePoints = 17;
constexpr int kPoseHeadPoints = 5;      // nose, eyes, ears: the landmarks show the head better
constexpr int kMeterSegments = 10;
constexpr size_t kMaxLabels = 6;
constexpr size_t kMaxLabelChars = 40;
constexpr size_t kMaxPoints = 512;      // landmarks of one face
constexpr size_t kMaxPeople = 4;

// The pose tile's bones (COCO order): legs and hips, torso, arms, head. No neck: an ear-to-shoulder
// diagonal looks like an extra arm when a head turns.
constexpr int kBones[][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6},
                             {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}};

const DrawColor kGreen{173, 42, 26}, kRed{81, 90, 240}, kYellow{210, 16, 146}, kLightBlue{169, 166, 16},
    kCyan{188, 154, 16}, kWhite{235, 128, 128}, kGray{128, 128, 128}, kOrange{156, 44, 200};

struct Person {
    std::array<float, kPosePoints * 3> k{};
    bool has_box = false;
    double box[4] = {0, 0, 0, 0};
};

/// One camera's `face_debug_v1`, in source pixels.
struct Model {
    std::string layers_key;
    bool has_face = false;
    bool speaking = false;
    double face[4] = {0, 0, 0, 0};
    std::vector<float> points;      // x, y, ...
    std::vector<Person> people;
    double pose_min = .4;
    bool has_look = false, has_reaction = false;
    double look = 0, reaction = 0;
    std::vector<std::string> labels;
};

/// One framing of the picture on a sheet frame: the picture's crop and where it is on the frame.
struct Framing {
    double cx = 0, cy = 0, cw = 0, ch = 0;
    double dx = 0, dy = 0, dw = 0, dh = 0;
};

bool numbers(const Parameters& array, size_t count) {
    if (!array.is_array() || array.size() < count) return false;
    for (size_t i = 0; i < count; ++i)
        if (!array[i].is_number()) return false;
    return true;
}

Model parseModel(const char* text) {
    Model m;
    try {
        const Parameters j = Parameters::parse(text);
        if (!j.is_object()) return m;
        if (j.contains("layers") && j["layers"].is_string()) m.layers_key = j["layers"].get<std::string>();
        if (j.contains("speaking") && j["speaking"].is_boolean()) m.speaking = j["speaking"].get<bool>();
        if (j.contains("face") && numbers(j["face"], 4)) {
            for (size_t i = 0; i < 4; ++i) m.face[i] = j["face"][i].get<double>();
            m.has_face = m.face[2] > m.face[0] && m.face[3] > m.face[1];
        }
        if (j.contains("points") && j["points"].is_array()) {
            const Parameters& points = j["points"];
            const size_t count = std::min(points.size() / 2, kMaxPoints) * 2;
            if (numbers(points, count))
                for (size_t i = 0; i < count; ++i) m.points.push_back(points[i].get<float>());
        }
        if (j.contains("people") && j["people"].is_array()) {
            for (const Parameters& p : j["people"]) {
                if (m.people.size() >= kMaxPeople) break;
                if (!p.is_object() || !p.contains("k") || !numbers(p["k"], kPosePoints * 3)) continue;
                Person person;
                for (size_t i = 0; i < person.k.size(); ++i) person.k[i] = p["k"][i].get<float>();
                if (p.contains("box") && numbers(p["box"], 4)) {
                    for (size_t i = 0; i < 4; ++i) person.box[i] = p["box"][i].get<double>();
                    person.has_box = true;
                }
                m.people.push_back(person);
            }
        }
        if (j.contains("pose_min") && j["pose_min"].is_number()) m.pose_min = j["pose_min"].get<double>();
        if (j.contains("look") && j["look"].is_number()) {
            m.look = std::max(-1., std::min(1., j["look"].get<double>()));
            m.has_look = true;
        }
        if (j.contains("reaction") && j["reaction"].is_number()) {
            m.reaction = std::max(0., std::min(100., j["reaction"].get<double>()));
            m.has_reaction = true;
        }
        if (j.contains("labels") && j["labels"].is_array()) {
            for (const Parameters& label : j["labels"]) {
                if (m.labels.size() >= kMaxLabels) break;
                if (!label.is_string()) continue;
                std::string line = label.get<std::string>();
                if (line.size() > kMaxLabelChars) line.resize(kMaxLabelChars);
                if (!line.empty()) m.labels.push_back(std::move(line));
            }
        }
    } catch (const std::exception&) {
        return Model();     // a malformed payload draws nothing
    }
    return m;
}

std::vector<Framing> parseFramings(const char* text) {
    std::vector<Framing> framings;
    try {
        const Parameters j = Parameters::parse(text);
        if (!j.is_object() || !j.contains("layers") || !j["layers"].is_array()) return framings;
        for (const Parameters& layer : j["layers"]) {
            if (!layer.is_object() || !layer.contains("crop") || !layer["crop"].is_object()) continue;
            const Parameters& crop = layer["crop"];
            const auto number = [](const Parameters& object, const char* key) {
                return object.contains(key) && object[key].is_number() ? object[key].get<double>() : 0.;
            };
            Framing f;
            f.cx = number(crop, "x");
            f.cy = number(crop, "y");
            f.cw = number(crop, "w");
            f.ch = number(crop, "h");
            f.dx = number(layer, "dst_x");
            f.dy = number(layer, "dst_y");
            f.dw = number(layer, "dst_w");
            f.dh = number(layer, "dst_h");
            if (f.cw > 0 && f.ch > 0 && f.dw > 0 && f.dh > 0) framings.push_back(f);
        }
    } catch (const std::exception&) {
        framings.clear();
    }
    return framings;
}

/// What was parsed from the last metadata strings, by string: a camera repeats its model payload
/// until the next face sample, and several layers may show the same frame.
template <typename T, size_t N>
class ParsedStrings {
    struct Entry {
        std::string text;
        T value;
        uint64_t used = 0;
    };
    std::vector<Entry> entries_;
    uint64_t clock_ = 0;

public:
    ParsedStrings() { entries_.reserve(N); }    // references handed out stay valid: never reallocated

    template <typename Parse>
    const T& get(const char* text, Parse&& parse) {
        ++clock_;
        Entry* oldest = nullptr;
        for (Entry& entry : entries_) {
            if (entry.text == text) {
                entry.used = clock_;
                return entry.value;
            }
            if (!oldest || entry.used < oldest->used) oldest = &entry;
        }
        if (entries_.size() < N) {
            entries_.emplace_back();
            oldest = &entries_.back();
        }
        oldest->text = text;
        oldest->value = parse(text);
        oldest->used = clock_;
        return oldest->value;
    }

    void clear() { entries_.clear(); }
};

/// Source pixels of the picture to canvas pixels, and the canvas rectangle the layer may paint.
struct Placement {
    double kx = 1, ky = 1, bx = 0, by = 0;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;      // [x0, x1) x [y0, y1)

    double x(double v) const { return kx * v + bx; }
    double y(double v) const { return ky * v + by; }
    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
};

class FaceDebugOverlay : public avp::mixer::DebugOverlay {
    MlDebugRenderer renderer_;
    CUcontext ctx_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction kernel_ = nullptr;
    ParsedStrings<Model, 32> models_;
    ParsedStrings<std::vector<Framing>, 32> framings_;
    size_t last_items_ = 0, last_tiles_ = 0;

    void unload() {
        if (!ctx_) return;
        cuCtxSetCurrent(ctx_);
        renderer_.release(ctx_);
        if (module_) CUDA_OVERLAY_CHECK_CU(cuModuleUnload(module_));
        module_ = nullptr;
        kernel_ = nullptr;
        ctx_ = nullptr;
    }

    /// Loads the kernel under the context of `canvas`, which the compositor has made current.
    void ensureKernel(const AVFrame* canvas) {
        const AVHWFramesContext* frames =
            canvas->hw_frames_ctx ? (const AVHWFramesContext*)canvas->hw_frames_ctx->data : nullptr;
        const AVCUDADeviceContext* device =
            frames && frames->device_ctx ? (const AVCUDADeviceContext*)frames->device_ctx->hwctx : nullptr;
        CUcontext current = device ? device->cuda_ctx : nullptr;
        if (!current) throw Error("debug overlay: the canvas frame has no CUDA context");
        if (kernel_ && current == ctx_) return;
        if (ctx_ && current != ctx_) {
            unload();                       // under the context it was loaded in
            if (CUDA_OVERLAY_CHECK_CU(cuCtxSetCurrent(current))) throw Error("debug overlay: cuCtxSetCurrent failed");
        }
        ctx_ = current;
        if (!module_) {
            const std::string image(avpl_ml_debug_ptx, avpl_ml_debug_ptx + avpl_ml_debug_ptx_len);
            if (CUDA_OVERLAY_CHECK_CU(cuModuleLoadDataEx(&module_, (const void*)image.c_str(), 0, nullptr, nullptr))) {
                module_ = nullptr;
                throw Error("debug overlay: cannot load the ml_debug kernel module");
            }
        }
        if (CUDA_OVERLAY_CHECK_CU(cuModuleGetFunction(&kernel_, module_, "kMlDebugNV12"))) {
            kernel_ = nullptr;
            throw Error("debug overlay: the module has no kMlDebugNV12");
        }
    }

    // ---- items, clipped on the host to the layer's rectangle: the kernel clips to the frame only.

    /// A line of `radius` around (ax, ay)-(bx, by), cut where its capsule would leave the rectangle.
    void segment(const Placement& p, double ax, double ay, double bx, double by, int radius, const DrawColor& color) {
        const double left = p.x0 + radius, right = p.x1 - 1 - radius, top = p.y0 + radius, bottom = p.y1 - 1 - radius;
        if (right < left || bottom < top) return;
        if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) || !std::isfinite(by)) return;
        // Liang-Barsky.
        double t0 = 0, t1 = 1;
        const double dx = bx - ax, dy = by - ay;
        const double edges[4][2] = {{-dx, ax - left}, {dx, right - ax}, {-dy, ay - top}, {dy, bottom - ay}};
        for (const auto& edge : edges) {
            const double step = edge[0], room = edge[1];
            if (step == 0) {
                if (room < 0) return;
                continue;
            }
            const double t = room / step;
            if (step < 0) {
                if (t > t1) return;
                t0 = std::max(t0, t);
            } else {
                if (t < t0) return;
                t1 = std::min(t1, t);
            }
        }
        const LineSegment cut{(int)std::lround(ax + t0 * dx), (int)std::lround(ay + t0 * dy),
                              (int)std::lround(ax + t1 * dx), (int)std::lround(ay + t1 * dy)};
        renderer_.addSegment(cut, radius, color);
    }

    void dot(const Placement& p, double x, double y, int radius, const DrawColor& color) {
        if (!std::isfinite(x) || !std::isfinite(y)) return;
        if (x - radius < p.x0 || y - radius < p.y0 || x + radius > p.x1 - 1 || y + radius > p.y1 - 1) return;
        renderer_.addDot(KeypointPos{(float)x, (float)y}, radius, color);
    }

    /// A box outline: the kernel's box when it is inside the rectangle, else the sides that are.
    void box(const Placement& p, double x0, double y0, double x1, double y1, int thickness, const DrawColor& color) {
        if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) return;
        if (x1 - x0 < 2 || y1 - y0 < 2) return;
        if (x0 >= p.x0 && y0 >= p.y0 && x1 <= p.x1 && y1 <= p.y1) {
            BatchedBBox b;
            b.x1 = (int)std::lround(x0);
            b.y1 = (int)std::lround(y0);
            b.x2 = (int)std::lround(x1);
            b.y2 = (int)std::lround(y1);
            b.thickness = thickness;
            b.y_color = color.y;
            b.u_color = color.u;
            b.v_color = color.v;
            renderer_.addBox(b);
            return;
        }
        const int radius = thickness > 1 ? 1 : 0;
        segment(p, x0, y0, x1, y0, radius, color);
        segment(p, x0, y1, x1, y1, radius, color);
        segment(p, x0, y0, x0, y1, radius, color);
        segment(p, x1, y0, x1, y1, radius, color);
    }

    /// The lines of `labels` stacked from `top` down, centred on `centre`, kept inside the rectangle.
    void labels(const Placement& p, const std::vector<std::string>& lines, double centre, double top, bool large, bool speaking = false) {
        if (lines.empty()) return;
        const GlyphPreset preset = large ? GlyphPreset::k10x14 : GlyphPreset::k5x7;
        const int advance = glyphAdvance(preset), glyph_h = glyphBaseHeight(preset);
        const int pad_x = 4, pad_y = 3;
        const int line_h = glyph_h + 2 * pad_y, step = line_h + 1;
        const int max_chars = (p.width() - 2 * pad_x) / advance;
        if (max_chars < 3 || p.height() < line_h) return;
        const int count = (int)std::min<size_t>(lines.size(), (size_t)std::max(1, p.height() / step));
        int y = (int)std::lround(top);
        y = std::max(p.y0, std::min(y, p.y1 - count * step));
        const size_t first = renderer_.labels().size();
        for (int i = 0; i < count; ++i, y += step) {
            const int len = std::min((int)lines[(size_t)i].size(), max_chars);
            const int bg_w = len * advance + 2 * pad_x;
            int bg_x = (int)std::lround(centre - bg_w / 2.);
            bg_x = std::max(p.x0, std::min(bg_x, p.x1 - bg_w));
            BatchedTextLabel label;
            label.line1_len = len;
            label.origin_x = bg_x + pad_x;
            label.origin_y = y + pad_y;
            label.font_scale = 1;
            label.line_spacing = 0;
            label.glyph_preset = (int)preset;
            label.bg_x = bg_x;
            label.bg_y = y;
            label.bg_w = bg_w;
            label.bg_h = line_h;
            label.draw_background = 1;
            label.background_opacity = .7f;
            const DrawColor& text_color = speaking && i == 0 ? kRed : kWhite;
            label.text_y = text_color.y;
            label.text_u = text_color.u;
            label.text_v = text_color.v;
            label.bg_y_color = 16;
            label.bg_u = 128;
            label.bg_v = 128;
            label.line1_offset = appendTextBlob(renderer_.text(), lines[(size_t)i].data(), len);
            renderer_.labels().push_back(label);
        }
        renderer_.addLabels(first);
    }

    /// Everything of one layer, in paint order: people, face, look arrow, meter, labels.
    void collect(const Model& m, const Placement& p, bool large) {
        const int line = large ? 2 : 1;
        // The landmarks show the head; the pose model's nose, eyes and ears only without them.
        const int first_point = m.points.empty() ? 0 : kPoseHeadPoints;
        for (const Person& person : m.people) {
            if (person.has_box)
                box(p, p.x(person.box[0]), p.y(person.box[1]), p.x(person.box[2]), p.y(person.box[3]), line, kLightBlue);
            const auto seen = [&](int k) { return k >= first_point && person.k[(size_t)k * 3 + 2] >= m.pose_min; };
            for (const auto& bone : kBones) {
                if (!seen(bone[0]) || !seen(bone[1])) continue;
                segment(p, p.x(person.k[(size_t)bone[0] * 3]), p.y(person.k[(size_t)bone[0] * 3 + 1]),
                        p.x(person.k[(size_t)bone[1] * 3]), p.y(person.k[(size_t)bone[1] * 3 + 1]), line, kYellow);
            }
            for (int k = first_point; k < kPosePoints; ++k)
                if (seen(k)) dot(p, p.x(person.k[(size_t)k * 3]), p.y(person.k[(size_t)k * 3 + 1]), line + 1, kRed);
        }
        for (size_t i = 0; i + 1 < m.points.size(); i += 2) dot(p, p.x(m.points[i]), p.y(m.points[i + 1]), line, kGreen);

        if (!m.has_face) {
            // No face to hang them on: the labels at the top left of the layer.
            labels(p, m.labels, p.x0 + p.width() / 2., p.y0 + 4, large, m.speaking);
            return;
        }
        const double x0 = p.x(m.face[0]), y0 = p.y(m.face[1]), x1 = p.x(m.face[2]), y1 = p.y(m.face[3]);
        box(p, x0, y0, x1, y1, m.speaking ? line + 1 : line, m.speaking ? kRed : kGreen);
        const double unit = std::max(x1 - x0, 24.);        // the face width, as shown
        const double centre = (x0 + x1) / 2;

        // Look: a rail under the chin with a centre tick, and an arrow that grows with the score.
        double below = y1 + 4;
        if (m.has_look) {
            const double rail = std::min(y1 + .18 * unit, p.y1 - 2. - 2 * line), reach = .5 * unit, tick = .05 * unit + 1;
            segment(p, centre - reach, rail, centre + reach, rail, line - 1, kWhite);
            segment(p, centre, rail - tick, centre, rail + tick, line - 1, kWhite);
            const double length = std::abs(m.look) * reach;
            if (std::abs(m.look) >= .05) {
                const double side = m.look > 0 ? 1 : -1, tip = centre + side * length;
                const double head = std::max(3., std::min(.12 * unit, .6 * length));
                segment(p, centre, rail, tip, rail, line, kCyan);
                segment(p, tip, rail, tip - side * head, rail - .75 * head, line, kCyan);
                segment(p, tip, rail, tip - side * head, rail + .75 * head, line, kCyan);
            }
            below = rail + tick + 3;
        }

        // Reaction: a level meter of ten segments beside the face, lit from the bottom.
        if (m.has_reaction) {
            const int radius = std::max(1, std::min(4, (int)std::lround(.03 * unit)));
            const double x = std::min(x1 + .12 * unit + radius, p.x1 - 2. - radius), step = (y1 - y0) / kMeterSegments;
            const int lit = (int)(m.reaction / 100. * kMeterSegments + .5);
            for (int i = 0; i < kMeterSegments; ++i) {
                const double bottom = y1 - i * step - .12 * step;
                const DrawColor& color = i >= lit ? kGray : i < 5 ? kGreen : i < 7 ? kYellow : i < 9 ? kOrange : kRed;
                segment(p, x, bottom, x, bottom - .76 * step, radius, color);
            }
        }
        labels(p, m.labels, centre, below, large, m.speaking);
    }

    void collect(const avp::mixer::DebugOverlayTile& tile, int canvas_w, int canvas_h) {
        if (!tile.source || !tile.source->metadata || tile.crop_w <= 0 || tile.crop_h <= 0 || tile.dst_w <= 0 ||
            tile.dst_h <= 0) return;
        const AVDictionaryEntry* entry = av_dict_get(tile.source->metadata, kModelKey, nullptr, 0);
        if (!entry || !entry->value) return;
        const Model& m = models_.get(entry->value, parseModel);

        // The framing of the picture this layer shows: the sheet layer that holds its centre.
        Framing shown;
        shown.cw = shown.dw = 1;
        shown.ch = shown.dh = 1;       // the frame is the picture
        if (!m.layers_key.empty()) {
            const AVDictionaryEntry* layers = av_dict_get(tile.source->metadata, m.layers_key.c_str(), nullptr, 0);
            if (!layers || !layers->value) return;
            const double centre_x = tile.crop_x + tile.crop_w / 2, centre_y = tile.crop_y + tile.crop_h / 2;
            const Framing* found = nullptr;
            for (const Framing& f : framings_.get(layers->value, parseFramings)) {
                if (centre_x >= f.dx && centre_x < f.dx + f.dw && centre_y >= f.dy && centre_y < f.dy + f.dh) {
                    found = &f;
                    break;
                }
            }
            if (!found) return;
            shown = *found;
        }
        Placement p;
        const double sheet_x = shown.dw / shown.cw, sheet_y = shown.dh / shown.ch;      // picture to frame
        const double out_x = tile.dst_w / tile.crop_w, out_y = tile.dst_h / tile.crop_h;  // frame to canvas
        p.kx = sheet_x * out_x;
        p.ky = sheet_y * out_y;
        p.bx = tile.dst_x + (shown.dx - shown.cx * sheet_x - tile.crop_x) * out_x;
        p.by = tile.dst_y + (shown.dy - shown.cy * sheet_y - tile.crop_y) * out_y;
        p.x0 = std::max(0, tile.dst_x);
        p.y0 = std::max(0, tile.dst_y);
        p.x1 = std::min(canvas_w, tile.dst_x + tile.dst_w);
        p.y1 = std::min(canvas_h, tile.dst_y + tile.dst_h);
        if (p.width() < 32 || p.height() < 32) return;      // a thumbnail shows no overlay
        collect(m, p, tile.dst_w >= 1200);
    }

public:
    ~FaceDebugOverlay() override = default;     // the owner calls release() under its context

    void draw(void* stream, AVFrame* canvas, const std::vector<avp::mixer::DebugOverlayTile>& tiles) override {
        last_items_ = last_tiles_ = 0;
        if (!canvas || canvas->format != AV_PIX_FMT_CUDA || !canvas->data[0] || !canvas->data[1]) return;
        renderer_.begin(canvas->width, canvas->height);
        for (const avp::mixer::DebugOverlayTile& tile : tiles) collect(tile, canvas->width, canvas->height);
        last_items_ = renderer_.itemCount();
        if (!last_items_) return;
        ensureKernel(canvas);
        if (!renderer_.draw(ctx_, (CUstream)stream, kernel_, canvas, "debug overlay", false))
            throw Error("debug overlay: failed to draw");
        last_tiles_ = renderer_.tileCount();
    }

    void release() override {
        unload();
        models_.clear();
        framings_.clear();
    }

    size_t lastItems() const override { return last_items_; }
    size_t lastTiles() const override { return last_tiles_; }
};

const struct RegisterFaceDebugOverlay {
    RegisterFaceDebugOverlay() {
        avp::mixer::debugOverlayFactory() = []() -> std::unique_ptr<avp::mixer::DebugOverlay> {
            return std::make_unique<FaceDebugOverlay>();
        };
    }
} register_face_debug_overlay;

}   // namespace
