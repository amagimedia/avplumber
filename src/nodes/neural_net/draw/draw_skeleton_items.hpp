#pragma once
// ml_debug's skeleton layer: the bones of a pose payload (the one draw_keypoints reads) as
// line segments in frame coordinates. A bone is drawn when both of its points are confident.
#include "draw_batch_shared.hpp"
#include "draw_payloads.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace cuda_overlay {

class SkeletonItems {
    std::string metadata_key_ = "yolo_pose";
    DrawColor color_{};
    int thickness_ = 3;
    double min_conf_ = 0.3;
    // Pairs of keypoint indices; the default is the COCO 17-point skeleton Ultralytics draws.
    std::vector<std::pair<int, int>> bones_;
    DetectionFilter content_;
    int debug_log_every_n_ = 0;
    std::string node_;

public:
    static const std::vector<std::pair<int, int>>& coco17() {
        static const std::vector<std::pair<int, int>> bones = {
            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},                 // legs, hips
            {5, 11}, {6, 12}, {5, 6},                                         // torso
            {5, 7}, {6, 8}, {7, 9}, {8, 10},                                  // arms
            {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6},           // head to shoulders
        };
        return bones;
    }

    static SkeletonItems fromParams(const Parameters& params, const std::string& node) {
        SkeletonItems items;
        items.node_ = node;
        items.metadata_key_ = params.value("metadata_key", std::string("yolo_pose"));
        items.thickness_ = params.value("thickness", 3);
        items.min_conf_ = params.value("min_conf", 0.3);
        items.debug_log_every_n_ = params.value("debug_log_every_n", 0);
        items.content_ = DetectionFilter::fromParams(params);
        const std::string color_name = params.value("color", std::string("white"));
        if (!tryParseNamedColor(color_name, items.color_)) {
            throw Error(node + ": unknown color: " + color_name);
        }
        if (params.contains("bones")) {
            if (!params["bones"].is_array()) throw Error(node + ": bones must be a list of [a, b] index pairs");
            for (const auto& bone : params["bones"]) {
                if (!bone.is_array() || bone.size() != 2 || !bone[0].is_number_integer() ||
                    !bone[1].is_number_integer() || bone[0].get<int>() < 0 || bone[1].get<int>() < 0) {
                    throw Error(node + ": bones must be a list of [a, b] index pairs");
                }
                items.bones_.emplace_back(bone[0].get<int>(), bone[1].get<int>());
            }
        } else {
            items.bones_ = coco17();
        }
        return items;
    }

    int thickness() const { return thickness_; }
    const DrawColor& color() const { return color_; }

    /// Appends the frame's bones to `segments`.
    void collect(FramePayloads& payloads, const FrameGeometry& frame, uint64_t frame_counter,
                 std::vector<LineSegment>& segments) const {
        if (!payloads.hasMetadata()) return;
        const Parameters* md = payloads.get(metadata_key_);
        if (!md) return;
        const size_t before = segments.size();
        try {
            append(*md, frame, segments);
        } catch (const std::exception&) {
            segments.resize(before);    // a malformed payload draws none of its bones
            return;
        }
        if (segments.size() > before && debug_log_every_n_ > 0 && (frame_counter % (uint64_t)debug_log_every_n_) == 0) {
            logstream << node_ << ": frame=" << frame_counter << " bones=" << (segments.size() - before);
        }
    }

private:
    void append(const Parameters& md, const FrameGeometry& frame, std::vector<LineSegment>& segments) const {
        if (md.value("num_keypoints", 0) <= 0) return;
        if (!md.contains("poses") || !md["poses"].is_array()) return;
        const double model_w = md.value("model_width", 0.0);
        const double model_h = md.value("model_height", 0.0);
        const YoloParseConfig cfg = content_.config(frame);
        for (const auto& pose : md["poses"]) {
            if (!pose.contains("keypoints") || !pose["keypoints"].is_array()) continue;
            const auto& kpts = pose["keypoints"];
            const int n = (int)kpts.size() / 3;
            for (const auto& [a, b] : bones_) {
                if (a >= n || b >= n) continue;
                const double ca = kpts[(size_t)(a * 3 + 2)].get<double>();
                const double cb = kpts[(size_t)(b * 3 + 2)].get<double>();
                if (!(ca >= min_conf_) || !(cb >= min_conf_)) continue;
                double ax, ay, bx, by;
                remapModelCoord(cfg, kpts[(size_t)(a * 3)].get<double>(), kpts[(size_t)(a * 3 + 1)].get<double>(),
                                model_w, model_h, ax, ay);
                remapModelCoord(cfg, kpts[(size_t)(b * 3)].get<double>(), kpts[(size_t)(b * 3 + 1)].get<double>(),
                                model_w, model_h, bx, by);
                if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) || !std::isfinite(by)) continue;
                // A bone with an end outside the frame is left out, like a dot whose centre is.
                if (ax < 0 || ay < 0 || bx < 0 || by < 0 || ax >= frame.width || bx >= frame.width ||
                    ay >= frame.height || by >= frame.height) continue;
                segments.push_back({(int)std::lround(ax), (int)std::lround(ay), (int)std::lround(bx), (int)std::lround(by)});
            }
        }
    }
};

} // namespace cuda_overlay
