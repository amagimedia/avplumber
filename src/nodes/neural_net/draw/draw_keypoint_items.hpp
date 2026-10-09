#pragma once
// draw_keypoints' reading of a frame: the dots of a pose payload, in frame coordinates.
#include "draw_batch_shared.hpp"
#include "draw_payloads.hpp"

namespace cuda_overlay {

class KeypointItems {
    std::string metadata_key_ = "yolo_pose";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    DrawColor color_{};
    int radius_ = 3;
    double min_conf_ = 0.0;
    bool require_wide_shot_ = false;
    DetectionFilter content_;
    int debug_log_every_n_ = 0;
    std::string node_;

public:
    static KeypointItems fromParams(const Parameters& params, const std::string& node) {
        KeypointItems items;
        items.node_ = node;
        items.metadata_key_ = params.value("metadata_key", std::string("yolo_pose"));
        items.camera_shot_metadata_key_ = params.value("camera_shot_metadata_key", std::string("camera_shot_info"));
        items.radius_ = params.value("radius", 3);
        items.min_conf_ = params.value("min_conf", 0.0);
        items.require_wide_shot_ = params.value("require_wide_shot", false);
        items.debug_log_every_n_ = params.value("debug_log_every_n", 0);
        items.content_ = DetectionFilter::fromParams(params);
        const std::string color_name = params.value("color", std::string("white"));
        if (!tryParseNamedColor(color_name, items.color_)) {
            throw Error(node + ": unknown color: " + color_name);
        }
        return items;
    }

    int radius() const { return radius_; }
    const DrawColor& color() const { return color_; }

    /// Appends the frame's dots to `points`.
    void collect(FramePayloads& payloads, const FrameGeometry& frame, uint64_t frame_counter,
                 std::vector<KeypointPos>& points) const {
        if (!payloads.hasMetadata()) return;
        if (require_wide_shot_ && payloads.shotType(camera_shot_metadata_key_) != "wide") return;
        const Parameters* md = payloads.get(metadata_key_);
        if (!md) return;

        const size_t before = points.size();
        const YoloParseConfig cfg = content_.config(frame);
        try {
            const double model_w = md->value("model_width", 0.0);
            const double model_h = md->value("model_height", 0.0);
            if (md->value("num_keypoints", 0) <= 0) return;
            if (!md->contains("poses") || !(*md)["poses"].is_array()) return;

            for (const auto& pose : (*md)["poses"]) {
                if (!pose.contains("keypoints") || !pose["keypoints"].is_array()) continue;
                const auto& kpts = pose["keypoints"];
                const int n = (int)kpts.size() / 3;
                for (int k = 0; k < n; ++k) {
                    const float kx = kpts[(size_t)(k * 3 + 0)].get<float>();
                    const float ky = kpts[(size_t)(k * 3 + 1)].get<float>();
                    const float kc = kpts[(size_t)(k * 3 + 2)].get<float>();
                    if ((double)kc < min_conf_) continue;

                    double fx, fy;
                    remapModelCoord(cfg, (double)kx, (double)ky, model_w, model_h, fx, fy);
                    // Only dots whose centre is in the frame are drawn.
                    if (fx < 0 || fy < 0 || fx >= frame.width || fy >= frame.height) continue;
                    points.push_back({(float)fx, (float)fy});
                }
            }
        } catch (const std::exception&) {
            // A malformed payload draws nothing, as before it draws none of its dots.
            points.resize(before);
            return;
        }

        if (points.size() > before && debug_log_every_n_ > 0 && (frame_counter % (uint64_t)debug_log_every_n_) == 0) {
            logstream << node_ << ": frame=" << frame_counter << " points=" << (points.size() - before);
        }
    }
};

} // namespace cuda_overlay
