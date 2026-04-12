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
    bool have_court_mask_data = false;
    std::vector<int> court_mask_indices;
    const float* court_mask_data = nullptr;
    uint32_t court_num_masks = 0;
    uint32_t court_mask_w = 0;
    uint32_t court_mask_h = 0;
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

} // namespace ball_tracker_detail
