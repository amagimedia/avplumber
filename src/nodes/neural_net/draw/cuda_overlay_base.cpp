#include "cuda_overlay_base.hpp"

namespace cuda_overlay {

bool tryParseGlyphPreset(const std::string& preset_name, GlyphPreset& preset_out) {
    const std::string normalized = normalizeColorName(preset_name);
    if (normalized == "5x7") {
        preset_out = GlyphPreset::k5x7;
        return true;
    }
    if (normalized == "10x14") {
        preset_out = GlyphPreset::k10x14;
        return true;
    }
    return false;
}

int glyphBaseWidth(GlyphPreset preset) {
    switch (preset) {
    case GlyphPreset::k10x14: return 10;
    case GlyphPreset::k5x7:
    default: return 5;
    }
}

int glyphBaseHeight(GlyphPreset preset) {
    switch (preset) {
    case GlyphPreset::k10x14: return 14;
    case GlyphPreset::k5x7:
    default: return 7;
    }
}

int glyphAdvance(GlyphPreset preset) {
    return glyphBaseWidth(preset) + 1;
}

bool scaleAndClampBBox(double x1, double y1, double x2, double y2, int frame_width, int frame_height,
                       int& out_x1, int& out_y1, int& out_x2, int& out_y2) {
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
        return false;
    }
    const auto clamp = [](int value, int lo, int hi) { return std::max(lo, std::min(hi, value)); };
    out_x1 = clamp((int)std::lround(x1), 0, frame_width);
    out_y1 = clamp((int)std::lround(y1), 0, frame_height);
    out_x2 = clamp((int)std::lround(x2), 0, frame_width);
    out_y2 = clamp((int)std::lround(y2), 0, frame_height);
    if (out_x2 < out_x1) std::swap(out_x1, out_x2);
    if (out_y2 < out_y1) std::swap(out_y1, out_y2);
    return out_x2 > out_x1 && out_y2 > out_y1;
}

bool remapModelCoord(const YoloParseConfig& cfg, double x, double y,
                     double model_w, double model_h, double& out_x, double& out_y) {
    if (cfg.model_content_width > 0.0 && cfg.model_content_height > 0.0) {
        const double content_x = std::max(0.0, std::min(x - cfg.model_content_offset_x, cfg.model_content_width));
        const double content_y = std::max(0.0, std::min(y - cfg.model_content_offset_y, cfg.model_content_height));
        out_x = content_x * ((double)cfg.frame_width / cfg.model_content_width);
        out_y = content_y * ((double)cfg.frame_height / cfg.model_content_height);
        return true;
    }
    const double sx = model_w > 0.0 ? (double)cfg.frame_width / model_w : 1.0;
    const double sy = model_h > 0.0 ? (double)cfg.frame_height / model_h : 1.0;
    out_x = x * sx;
    out_y = y * sy;
    return true;
}

bool yoloDetectionAllowed(const Parameters& det, const YoloParseConfig& cfg) {
    const double conf = det.value("conf", 0.0);
    if (conf < cfg.min_conf) return false;

    const bool has_class_filter = cfg.allowed_classes && !cfg.allowed_classes->empty();
    const bool has_label_filter = cfg.allowed_labels && !cfg.allowed_labels->empty();
    if (!has_class_filter && !has_label_filter) return true;

    bool class_match = false;
    bool label_match = false;
    if (has_class_filter && det.contains("cls")) {
        class_match = cfg.allowed_classes->count(det["cls"].get<int>()) > 0;
    }
    if (has_label_filter && det.contains("label") && det["label"].is_string()) {
        label_match = cfg.allowed_labels->count(det["label"].get<std::string>()) > 0;
    }
    return class_match || label_match;
}

void parseYoloDetections(const Parameters& md, const YoloParseConfig& cfg,
                         std::vector<ParsedYoloDetection>& detections_out) {
    if (!md.contains("detections") || !md["detections"].is_array()) return;

    const std::string coord_space = md.value("coord_space", std::string("model"));
    const double model_w = md.value("model_width", (double)cfg.frame_width);
    const double model_h = md.value("model_height", (double)cfg.frame_height);

    for (const auto& det : md["detections"]) {
        if (!det.is_object()) continue;
        if (!yoloDetectionAllowed(det, cfg)) continue;
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;

        double x1 = det["xyxy"][0].get<double>();
        double y1 = det["xyxy"][1].get<double>();
        double x2 = det["xyxy"][2].get<double>();
        double y2 = det["xyxy"][3].get<double>();
        if (coord_space == "model") {
            if (!remapModelCoord(cfg, x1, y1, model_w, model_h, x1, y1)) continue;
            if (!remapModelCoord(cfg, x2, y2, model_w, model_h, x2, y2)) continue;
        }

        ParsedYoloDetection parsed;
        if (!scaleAndClampBBox(x1, y1, x2, y2, cfg.frame_width, cfg.frame_height,
                               parsed.x1, parsed.y1, parsed.x2, parsed.y2)) {
            continue;
        }

        parsed.conf = det.value("conf", 0.0);
        if (det.contains("team")) {
            parsed.team = det["team"].get<int>();
            parsed.has_team = true;
        }
        if (det.contains("cls")) {
            parsed.cls = det["cls"].get<int>();
            parsed.has_cls = true;
        }
        if (det.contains("label") && det["label"].is_string()) {
            parsed.label = det["label"].get<std::string>();
            parsed.has_label = true;
        }
        if (det.contains("model_index")) {
            parsed.model_index = det["model_index"].get<int>();
            parsed.has_model_index = true;
        }
        if (det.contains("track_id")) {
            parsed.track_id = det["track_id"].get<int>();
            parsed.has_track_id = true;
        }
        if (det.contains("predicted")) {
            parsed.predicted = det["predicted"].get<bool>();
            parsed.has_predicted = true;
        }
        if (det.contains("velocity") && det["velocity"].is_array() && det["velocity"].size() >= 2) {
            parsed.velocity_x = det["velocity"][0].get<double>();
            parsed.velocity_y = det["velocity"][1].get<double>();
            parsed.has_velocity = true;
        } else if (det.contains("velocity_x") && det.contains("velocity_y")) {
            parsed.velocity_x = det["velocity_x"].get<double>();
            parsed.velocity_y = det["velocity_y"].get<double>();
            parsed.has_velocity = true;
        }
        if (det.contains("jersey_mode_ratio")) {
            parsed.jersey_mode_ratio = det["jersey_mode_ratio"].get<double>();
            parsed.has_jersey_mode_ratio = true;
        }
        detections_out.push_back(std::move(parsed));
    }
}

} // namespace cuda_overlay
