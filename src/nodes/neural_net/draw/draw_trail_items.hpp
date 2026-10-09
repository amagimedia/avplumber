#pragma once
// draw_trail's reading of a frame: the segments joining a payload's trail points.
#include "draw_batch_shared.hpp"
#include "draw_payloads.hpp"

#include <cmath>

namespace cuda_overlay {

class TrailItems {
    std::string metadata_key_ = "yolo_detections";
    DrawColor color_{81, 90, 240};
    int thickness_ = 2;
    DetectionFilter content_;
    int debug_log_every_n_ = 0;
    std::string node_;

public:
    static TrailItems fromParams(const Parameters& params, const std::string& node) {
        TrailItems items;
        items.node_ = node;
        items.metadata_key_ = params.value("metadata_key", std::string("yolo_detections"));
        items.thickness_ = params.value("thickness", 2);
        // An unknown colour name is red, as it always was for a trail.
        if (!tryParseNamedColor(params.value("color", std::string("red")), items.color_)) {
            items.color_ = DrawColor{81, 90, 240};
        }
        items.content_ = DetectionFilter::fromParams(params);
        items.debug_log_every_n_ = params.value("debug_log_every_n", 0);
        return items;
    }

    int thickness() const { return thickness_; }
    const DrawColor& color() const { return color_; }

    /// Appends the frame's trail segments to `segments`; false when the frame has no trail.
    bool collect(FramePayloads& payloads, const FrameGeometry& frame, uint64_t frame_counter,
                 std::vector<LineSegment>& segments) const {
        const size_t before = segments.size();
        const Parameters* md = payloads.get(metadata_key_);
        bool found = false;
        if (md) {
            try {
                found = append(*md, frame, segments);
            } catch (...) {
                segments.resize(before);
                found = false;
            }
        }
        if (debug_log_every_n_ > 0 && (frame_counter % (uint64_t)debug_log_every_n_) == 0) {
            if (found) logstream << node_ << ": frame=" << frame_counter << " segments=" << (segments.size() - before);
            else logstream << node_ << ": frame=" << frame_counter << " no trail data";
        }
        return found;
    }

private:
    bool append(const Parameters& md, const FrameGeometry& frame, std::vector<LineSegment>& segments) const {
        if (!md.contains("trail") || !md["trail"].is_array()) return false;
        const double model_w = md.value("model_width", (double)frame.width);
        const double model_h = md.value("model_height", (double)frame.height);
        const auto& trail = md["trail"];
        if (trail.size() < 2) return false;

        const YoloParseConfig cfg = content_.config(frame);
        const size_t before = segments.size();
        double prev_x = 0.0, prev_y = 0.0;
        bool have_prev = false;
        for (const auto& pt : trail) {
            if (!pt.is_array() || pt.size() < 2) continue;
            double fx = 0.0, fy = 0.0;
            remapModelCoord(cfg, pt[0].get<double>(), pt[1].get<double>(), model_w, model_h, fx, fy);
            if (have_prev) {
                segments.push_back({(int)std::round(prev_x), (int)std::round(prev_y),
                                    (int)std::round(fx), (int)std::round(fy)});
            }
            prev_x = fx;
            prev_y = fy;
            have_prev = true;
        }
        return segments.size() > before;
    }
};

} // namespace cuda_overlay
