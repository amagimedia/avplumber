#pragma once

#include "../../node_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

namespace ball_tracker_detail {

struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    int model_index = -1;
    std::string engine_name;
    bool has_engine_name = false;
};

struct TrailPoint {
    int x = 0, y = 0;
    uint64_t frame = 0;
};

struct GateDecision {
    bool hard_cap_ok = true;
    bool distance_ok = true;
    bool prediction_ok = false;
    bool iou_ok = true;
    bool speed_ok = true;
    bool continuity_ok = true;
    bool accept = true;
    double dist_prev = 0.0;
    double dist_real = 0.0;
    double d_pred = 0.0;
    double hard_cap = 0.0;
    double base_gate = 0.0;
    double speed_gate = 0.0;
    uint64_t real_gap_frames = 0;
};

struct ParsedFrameMetadata {
    Parameters md;
    Parameters other_det_items = Parameters::array();
    std::vector<DetectionBox> ball_dets;
    double model_w = 0.0;
    double model_h = 0.0;
    bool have_camera_shot_info = false;
    std::string shot_type;
    double court_coverage = 0.0;
    bool have_court_mask_data = false;
    std::vector<int> court_mask_indices;
    const float* court_mask_data = nullptr;
    uint32_t court_num_masks = 0;
    uint32_t court_mask_w = 0;
    uint32_t court_mask_h = 0;
};

struct CourtBoundsCheckConfig {
    bool enabled = true;
    bool require_wide_shot = true;
    double min_court_coverage = 0.1;
    double mask_threshold = 0.5;
    int sample_rows = 3;
    double min_horizontal_overlap_ratio = 0.35;
    double max_horizontal_outside_px = 2.0;
};

struct CourtBoundsCheckResult {
    bool usable = false;
    bool veto = false;
    int sample_rows_used = 0;
    double best_overlap_ratio = 0.0;
    double worst_overlap_ratio = 1.0;
    double max_left_out_px = 0.0;
    double max_right_out_px = 0.0;
    std::string reason = "disabled";
};

struct CandidateSelection {
    int best_idx = -1;
    int best_gated_idx = -1;
    double best_score = -1e18;
    double best_gated_score = -1e18;
    std::vector<double> scores;
    std::vector<GateDecision> gates;
};

struct TrackingDecision {
    bool accepted = false;
    std::string source;
    double ball_score = 0.0;
    DetectionBox tracked_det;
};

inline double boxWidth(const DetectionBox& b) { return b.x2 - b.x1; }
inline double boxHeight(const DetectionBox& b) { return b.y2 - b.y1; }
inline double centerX(const DetectionBox& b) { return (b.x1 + b.x2) * 0.5; }
inline double centerY(const DetectionBox& b) { return (b.y1 + b.y2) * 0.5; }

inline bool finiteBox(const DetectionBox& b) {
    return std::isfinite(b.x1) && std::isfinite(b.y1)
        && std::isfinite(b.x2) && std::isfinite(b.y2)
        && b.x2 > b.x1 && b.y2 > b.y1;
}

inline double iou(const DetectionBox& a, const DetectionBox& b) {
    const double ix1 = std::max(a.x1, b.x1);
    const double iy1 = std::max(a.y1, b.y1);
    const double ix2 = std::min(a.x2, b.x2);
    const double iy2 = std::min(a.y2, b.y2);
    const double iw = std::max(0.0, ix2 - ix1);
    const double ih = std::max(0.0, iy2 - iy1);
    const double inter = iw * ih;
    if (inter <= 0.0) return 0.0;
    const double aa = std::max(0.0, boxWidth(a)) * std::max(0.0, boxHeight(a));
    const double ab = std::max(0.0, boxWidth(b)) * std::max(0.0, boxHeight(b));
    const double uni = aa + ab - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

inline double hypot2(double dx, double dy) { return std::sqrt(dx * dx + dy * dy); }

inline double sizePenalty(double w, double h, double model_w, double model_h, double target_rel) {
    const double s = std::max(w, h);
    const double tgt = target_rel * std::min(model_w, model_h);
    if (tgt <= 0.0) return 0.0;
    return std::clamp(std::abs(s - tgt) / tgt, 0.0, 2.0) * 0.5;
}

inline double roundnessPenalty(double w, double h) {
    const double ar = w / std::max(h, 1e-6);
    return 1.0 - std::exp(-((ar - 1.0) * (ar - 1.0)) / 0.15);
}

inline void addTrailPoint(std::deque<TrailPoint>& trail, int x, int y, uint64_t frame,
                          bool densify, int max_gap, size_t max_len) {
    if (densify && !trail.empty()) {
        const auto& last = trail.back();
        int gap = (int)(frame - last.frame);
        if (gap > 1 && gap <= max_gap) {
            for (int t = 1; t < gap; ++t) {
                double alpha = (double)t / (double)gap;
                int xi = (int)std::round((1.0 - alpha) * last.x + alpha * x);
                int yi = (int)std::round((1.0 - alpha) * last.y + alpha * y);
                trail.push_back({xi, yi, last.frame + (uint64_t)t});
                while (trail.size() > max_len) trail.pop_front();
            }
        }
    }
    trail.push_back({x, y, frame});
    while (trail.size() > max_len) trail.pop_front();
}

inline bool courtMaskRowBounds(const ParsedFrameMetadata& parsed,
                               int row,
                               double mask_threshold,
                               int& out_left,
                               int& out_right) {
    if (!parsed.have_court_mask_data || !parsed.court_mask_data
        || parsed.court_mask_w == 0 || parsed.court_mask_h == 0
        || parsed.court_mask_indices.empty()
        || row < 0 || row >= (int)parsed.court_mask_h) {
        return false;
    }

    const size_t pixels_per_mask = (size_t)parsed.court_mask_w * (size_t)parsed.court_mask_h;
    int left = -1;
    int right = -1;

    for (int x = 0; x < (int)parsed.court_mask_w; ++x) {
        const size_t pix = (size_t)row * parsed.court_mask_w + (size_t)x;
        bool on_court = false;
        for (int mi : parsed.court_mask_indices) {
            if (mi < 0 || (uint32_t)mi >= parsed.court_num_masks) continue;
            const float* mask = parsed.court_mask_data + (size_t)mi * pixels_per_mask;
            if (mask[pix] >= (float)mask_threshold) {
                on_court = true;
                break;
            }
        }
        if (!on_court) continue;
        if (left < 0) left = x;
        right = x;
    }

    if (left < 0 || right < left) return false;
    out_left = left;
    out_right = right;
    return true;
}

inline std::vector<int> sampleCourtMaskRows(const DetectionBox& det,
                                            const ParsedFrameMetadata& parsed,
                                            int sample_rows) {
    std::vector<int> rows;
    if (!finiteBox(det) || parsed.model_h <= 0.0 || parsed.court_mask_h == 0 || sample_rows <= 0) {
        return rows;
    }

    const double y1 = std::clamp(det.y1 / parsed.model_h * (double)parsed.court_mask_h,
                                 0.0, std::max(0.0, (double)parsed.court_mask_h - 1.0));
    const double y2 = std::clamp(det.y2 / parsed.model_h * (double)parsed.court_mask_h,
                                 0.0, std::max(0.0, (double)parsed.court_mask_h - 1.0));
    if (y2 <= y1) {
        rows.push_back((int)std::lround(y1));
        return rows;
    }

    rows.reserve((size_t)sample_rows);
    for (int i = 0; i < sample_rows; ++i) {
        const double alpha = (double)(i + 1) / (double)(sample_rows + 1);
        const int row = (int)std::lround(y1 + alpha * (y2 - y1));
        if (rows.empty() || rows.back() != row) rows.push_back(row);
    }
    return rows;
}

inline CourtBoundsCheckResult checkCourtBoundsHorizontalOverlap(const DetectionBox& det,
                                                                const ParsedFrameMetadata& parsed,
                                                                const CourtBoundsCheckConfig& cfg) {
    CourtBoundsCheckResult result;
    if (!cfg.enabled) {
        result.reason = "disabled";
        return result;
    }
    if (cfg.require_wide_shot) {
        if (!parsed.have_camera_shot_info) {
            result.reason = "no_camera_shot_info";
            return result;
        }
        if (parsed.shot_type != "wide") {
            result.reason = "not_wide";
            return result;
        }
        if (parsed.court_coverage < cfg.min_court_coverage) {
            result.reason = "low_court_coverage";
            return result;
        }
    }
    if (!finiteBox(det) || !parsed.have_court_mask_data || !parsed.court_mask_data
        || parsed.model_w <= 0.0 || parsed.model_h <= 0.0
        || parsed.court_mask_w == 0 || parsed.court_mask_h == 0
        || parsed.court_mask_indices.empty()) {
        result.reason = "no_mask";
        return result;
    }

    const double bbox_left = std::clamp(det.x1 / parsed.model_w * (double)parsed.court_mask_w,
                                        0.0, (double)parsed.court_mask_w);
    const double bbox_right = std::clamp(det.x2 / parsed.model_w * (double)parsed.court_mask_w,
                                         0.0, (double)parsed.court_mask_w);
    const double bbox_width = std::max(1e-6, bbox_right - bbox_left);
    const auto rows = sampleCourtMaskRows(det, parsed, std::max(1, cfg.sample_rows));
    if (rows.empty()) {
        result.reason = "no_court_rows";
        return result;
    }

    for (int row : rows) {
        int court_left = -1;
        int court_right = -1;
        if (!courtMaskRowBounds(parsed, row, cfg.mask_threshold, court_left, court_right)) continue;

        const double court_left_edge = (double)court_left;
        const double court_right_edge = (double)(court_right + 1);
        const double overlap =
            std::max(0.0, std::min(bbox_right, court_right_edge) - std::max(bbox_left, court_left_edge));
        const double overlap_ratio = overlap / bbox_width;
        const double left_out = std::max(0.0, court_left_edge - bbox_left);
        const double right_out = std::max(0.0, bbox_right - court_right_edge);

        result.usable = true;
        result.sample_rows_used++;
        result.best_overlap_ratio = std::max(result.best_overlap_ratio, overlap_ratio);
        result.worst_overlap_ratio = std::min(result.worst_overlap_ratio, overlap_ratio);
        result.max_left_out_px = std::max(result.max_left_out_px, left_out);
        result.max_right_out_px = std::max(result.max_right_out_px, right_out);
    }

    if (!result.usable) {
        result.reason = "no_court_rows";
        return result;
    }

    const double max_outside = std::max(result.max_left_out_px, result.max_right_out_px);
    if (result.best_overlap_ratio >= cfg.min_horizontal_overlap_ratio
        || max_outside <= cfg.max_horizontal_outside_px) {
        result.reason = "inside_bounds";
        return result;
    }

    result.veto = true;
    if (result.max_left_out_px > result.max_right_out_px) {
        result.reason = "outside_left";
    } else if (result.max_right_out_px > result.max_left_out_px) {
        result.reason = "outside_right";
    } else {
        result.reason = "low_overlap";
    }
    return result;
}

} // namespace ball_tracker_detail
