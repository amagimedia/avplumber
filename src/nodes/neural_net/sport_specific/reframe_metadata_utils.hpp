#pragma once

#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace avp_sport_reframe {

struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    int track_id = -1;
    bool has_track_id = false;
    int model_index = -1;
};

inline double boxWidth(const DetectionBox& b) { return b.x2 - b.x1; }
inline double boxHeight(const DetectionBox& b) { return b.y2 - b.y1; }
inline double centerX(const DetectionBox& b) { return (b.x1 + b.x2) * 0.5; }
inline double centerY(const DetectionBox& b) { return (b.y1 + b.y2) * 0.5; }

inline bool finiteBox(const DetectionBox& b) {
    return std::isfinite(b.x1) && std::isfinite(b.y1) &&
           std::isfinite(b.x2) && std::isfinite(b.y2) &&
           b.x2 > b.x1 && b.y2 > b.y1;
}

inline double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

inline bool labelMatches(const DetectionBox& det, const std::string& label) {
    return label.empty() || (det.has_label && det.label == label);
}

inline Parameters tryParseMetadata(const AVFrame* raw, const std::string& key) {
    if (!raw || !raw->metadata || key.empty()) return {};
    AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
    if (!entry || !entry->value) return {};
    try {
        return Parameters::parse(entry->value);
    } catch (...) {
        return {};
    }
}

inline bool parseDetections(const AVFrame* raw,
                            const std::string& key,
                            std::vector<DetectionBox>& out,
                            double& model_w,
                            double& model_h) {
    out.clear();
    Parameters md = tryParseMetadata(raw, key);
    if (!md.is_object()) return false;
    model_w = md.value("model_width", model_w);
    model_h = md.value("model_height", model_h);
    if (!md.contains("detections") || !md["detections"].is_array()) return false;

    for (const auto& item : md["detections"]) {
        if (!item.is_object()) continue;
        if (!item.contains("xyxy") || !item["xyxy"].is_array() || item["xyxy"].size() < 4) continue;
        DetectionBox d;
        d.cls = item.value("cls", -1);
        d.conf = item.value("conf", 0.0);
        d.model_index = item.value("model_index", -1);
        d.x1 = item["xyxy"][0].get<double>();
        d.y1 = item["xyxy"][1].get<double>();
        d.x2 = item["xyxy"][2].get<double>();
        d.y2 = item["xyxy"][3].get<double>();
        if (item.contains("label") && item["label"].is_string()) {
            d.label = item["label"].get<std::string>();
            d.has_label = true;
        }
        if (item.contains("track_id") && !item["track_id"].is_null()) {
            try {
                d.track_id = item["track_id"].get<int>();
                d.has_track_id = d.track_id >= 0;
            } catch (...) {
                d.track_id = -1;
                d.has_track_id = false;
            }
        }
        if (d.x2 < d.x1) std::swap(d.x1, d.x2);
        if (d.y2 < d.y1) std::swap(d.y1, d.y2);
        if (finiteBox(d)) out.push_back(d);
    }
    return true;
}

inline double pointToBoxDistance(double px, double py, const DetectionBox& b) {
    const double dx = std::max({0.0, b.x1 - px, px - b.x2});
    const double dy = std::max({0.0, b.y1 - py, py - b.y2});
    return std::sqrt(dx * dx + dy * dy);
}

inline double boxAreaFractionInsideCircle(const DetectionBox& b,
                                          double cx,
                                          double cy,
                                          double r,
                                          int y_samples = 129) {
    const double w = boxWidth(b);
    const double h = boxHeight(b);
    if (!(w > 0.0) || !(h > 0.0) || !(r > 0.0) || y_samples < 2) return 0.0;

    const double r2 = r * r;
    const double step = h / (double)(y_samples - 1);
    double area = 0.0;
    double prev_width = 0.0;
    bool have_prev = false;
    for (int i = 0; i < y_samples; ++i) {
        const double y = b.y1 + step * (double)i;
        const double dy = y - cy;
        double width = 0.0;
        if (dy * dy <= r2) {
            const double sx = std::sqrt(std::max(0.0, r2 - dy * dy));
            const double lo = std::max(b.x1, cx - sx);
            const double hi = std::min(b.x2, cx + sx);
            width = std::max(0.0, hi - lo);
        }
        if (have_prev) area += 0.5 * (prev_width + width) * step;
        prev_width = width;
        have_prev = true;
    }
    return area / (w * h);
}

inline bool cameraShotIsWide(const AVFrame* raw, const std::string& key, bool default_value = true) {
    if (key.empty()) return default_value;
    Parameters md = tryParseMetadata(raw, key);
    if (!md.is_object()) return default_value;
    const std::string shot = md.value("camera_shot_type", std::string());
    if (shot.empty()) return default_value;
    return shot == "wide";
}

inline bool sceneResetRequested(const AVFrame* raw,
                                const std::string& key,
                                double mean_abs_threshold) {
    if (key.empty()) return false;
    Parameters md = tryParseMetadata(raw, key);
    if (!md.is_object()) return false;
    if (md.value("scene_cut", false) || md.value("is_cut", false) || md.value("reset", false)) return true;
    if (mean_abs_threshold > 0.0) {
        const double mean_abs = md.value("mean_abs", 0.0);
        if (mean_abs >= mean_abs_threshold) return true;
    }
    return false;
}

inline void addDetection(Parameters& arr, const DetectionBox& d, const std::string& label_override = std::string()) {
    Parameters det;
    det["label"] = label_override.empty() ? d.label : label_override;
    det["conf"] = d.conf;
    det["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
    if (d.cls >= 0) det["cls"] = d.cls;
    if (d.has_track_id) det["track_id"] = d.track_id;
    if (d.model_index >= 0) det["model_index"] = d.model_index;
    arr.push_back(det);
}

inline Parameters emptyYoloMetadata(double model_w, double model_h) {
    Parameters out;
    out["coord_space"] = "model";
    out["model_width"] = model_w;
    out["model_height"] = model_h;
    out["detections"] = Parameters::array();
    return out;
}

} // namespace avp_sport_reframe
