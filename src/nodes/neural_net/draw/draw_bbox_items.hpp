#pragma once
// draw_bbox's reading of a frame: the reframer's viewport rectangle and the detection boxes of the
// node's metadata keys, in the order they are painted.
#include "draw_batch_shared.hpp"
#include "draw_payloads.hpp"

#include <cmath>
#include <optional>
#include <tuple>
#include <unordered_map>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace cuda_overlay {

class BBoxItems {
    std::vector<std::string> metadata_keys_;
    bool have_last_viewport_crop_ = false;
    int last_viewport_crop_x_ = 0;
    int last_viewport_crop_y_ = 0;
    int last_viewport_input_w_ = 0;
    int last_viewport_input_h_ = 0;
    int last_viewport_dst_w_ = 0;
    int last_viewport_dst_h_ = 0;
    int bbox_thickness_ = 2;
    int debug_log_every_n_ = 0;
    DetectionFilter filter_;
    DrawColor default_color_{};
    std::optional<DrawColor> predicted_color_;
    bool predicted_only_ = false;
    bool hide_predicted_ = false;
    std::unordered_map<int, DrawColor> model_colors_;
    std::unordered_map<std::string, DrawColor> label_colors_;
    std::string node_;
    // The frame being read; set by collect().
    FrameGeometry frame_;

    static int clampInt(int value, int lo, int hi) {
        return std::max(lo, std::min(hi, value));
    }

    DrawColor resolveModelColor(const ParsedYoloDetection& det) const {
        if (predicted_color_ && det.has_predicted && det.predicted) {
            return *predicted_color_;
        }
        if (!label_colors_.empty() && det.has_label) {
            const auto it = label_colors_.find(det.label);
            if (it != label_colors_.end()) {
                return it->second;
            }
        }
        if (det.has_model_index) {
            const auto it = model_colors_.find(det.model_index);
            if (it != model_colors_.end()) {
                return it->second;
            }
        }
        return default_color_;
    }

    BatchedBBox box(int x1, int y1, int x2, int y2, const DrawColor& color) const {
        BatchedBBox entry;
        entry.x1 = x1;
        entry.y1 = y1;
        entry.x2 = x2;
        entry.y2 = y2;
        entry.thickness = bbox_thickness_;
        entry.y_color = color.y;
        entry.u_color = color.u;
        entry.v_color = color.v;
        return entry;
    }

    int viewportChromaAlign(bool horizontal) const {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(frame_.sw_format);
        const int log2 = !desc ? -1 : horizontal ? desc->log2_chroma_w : desc->log2_chroma_h;
        return log2 < 0 ? 1 : 1 << log2;
    }

    static int alignViewportCropCoord(int value, int align) {
        if (align <= 1) return value;
        return value & ~(align - 1);
    }

    int clampViewportCropX(int x, int dst_w) const {
        const int max_x = std::max(0, frame_.width - dst_w);
        const int clamped = std::max(0, std::min(x, max_x));
        return alignViewportCropCoord(clamped, viewportChromaAlign(true));
    }

    int clampViewportCropY(int y, int dst_h) const {
        const int max_y = std::max(0, frame_.height - dst_h);
        const int clamped = std::max(0, std::min(y, max_y));
        return alignViewportCropCoord(clamped, viewportChromaAlign(false));
    }

    std::pair<int, int> centerViewportCrop(int dst_w, int dst_h) const {
        return {
            clampViewportCropX((frame_.width - dst_w) / 2, dst_w),
            clampViewportCropY((frame_.height - dst_h) / 2, dst_h)
        };
    }

    static bool metadataHasViewportDstDims(const Parameters& md) {
        return md.contains("viewport_dst_width") && md["viewport_dst_width"].is_number()
            && md.contains("viewport_dst_height") && md["viewport_dst_height"].is_number();
    }

    // Same interpretation as crop_metadata_cuda::parseCropPosition (metadata + crop size).
    bool parseViewportCropPositionFromMd(const Parameters& md, int dst_w, int dst_h, int& x_out, int& y_out) const {
        double center_x = NAN;
        double center_y = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
            const auto& bbox = md["viewport_bbox"];
            const double x1 = bbox[0].get<double>();
            const double y1 = bbox[1].get<double>();
            const double x2 = bbox[2].get<double>();
            const double y2 = bbox[3].get<double>();
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else if (md.contains("viewport_center_x")) {
            center_x = md["viewport_center_x"].get<double>();
            center_y = frame_.height * 0.5;
        } else if (md.contains("bbox_norm") && md["bbox_norm"].is_array() && md["bbox_norm"].size() >= 4) {
            const auto& bbox = md["bbox_norm"];
            const double fw = md.value("full_frame_width", frame_.width);
            const double fh = md.value("full_frame_height", frame_.height);
            const double x1 = bbox[0].get<double>() * fw;
            const double y1 = bbox[1].get<double>() * fh;
            const double x2 = bbox[2].get<double>() * fw;
            const double y2 = bbox[3].get<double>() * fh;
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else {
            return false;
        }

        if (!std::isfinite(center_x) || !std::isfinite(center_y)) return false;

        x_out = clampViewportCropX((int)std::lround(center_x - (double)dst_w * 0.5), dst_w);
        y_out = clampViewportCropY((int)std::lround(center_y - (double)dst_h * 0.5), dst_h);
        return true;
    }

    bool tryParseViewportCrop(const std::vector<const Parameters*>& payloads, int dst_w, int dst_h,
                              int& x_out, int& y_out) const {
        for (const Parameters* md : payloads) {
            try {
                if (parseViewportCropPositionFromMd(*md, dst_w, dst_h, x_out, y_out)) return true;
            } catch (const std::exception&) {
                continue;
            }
        }
        return false;
    }

    bool tryReadViewportDstDims(const std::vector<const Parameters*>& payloads, int& w_out, int& h_out) const {
        for (const Parameters* md : payloads) {
            try {
                if (!metadataHasViewportDstDims(*md)) continue;
                const int w = (*md)["viewport_dst_width"].get<int>();
                const int h = (*md)["viewport_dst_height"].get<int>();
                if (w <= 0 || h <= 0) continue;
                if ((w & 1) || (h & 1)) continue;
                if (w > frame_.width || h > frame_.height) continue;
                w_out = w;
                h_out = h;
                return true;
            } catch (const std::exception&) {
                continue;
            }
        }
        return false;
    }

    void updateViewportCropPosition(const std::vector<const Parameters*>& payloads, int dst_w, int dst_h,
                                    int& x_out, int& y_out) {
        int next_x = 0;
        int next_y = 0;
        const bool parsed = tryParseViewportCrop(payloads, dst_w, dst_h, next_x, next_y);
        if (!parsed) {
            if (have_last_viewport_crop_) {
                next_x = last_viewport_crop_x_;
                next_y = last_viewport_crop_y_;
            } else {
                std::tie(next_x, next_y) = centerViewportCrop(dst_w, dst_h);
            }
        }
        last_viewport_crop_x_ = next_x;
        last_viewport_crop_y_ = next_y;
        have_last_viewport_crop_ = true;
        x_out = next_x;
        y_out = next_y;
    }

    bool parseSingleBBoxMetadata(const Parameters& md, std::vector<BatchedBBox>& boxes_out) const {
        double x1 = NAN;
        double y1 = NAN;
        double x2 = NAN;
        double y2 = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
            if (metadataHasViewportDstDims(md)) {
                return false;
            }
            const auto& bbox = md["viewport_bbox"];
            const double fw = md.value("full_frame_width", (double)frame_.width);
            const double fh = md.value("full_frame_height", (double)frame_.height);
            const double sx = fw > 0.0 ? (double)frame_.width / fw : 1.0;
            const double sy = fh > 0.0 ? (double)frame_.height / fh : 1.0;
            x1 = bbox[0].get<double>() * sx;
            y1 = bbox[1].get<double>() * sy;
            x2 = bbox[2].get<double>() * sx;
            y2 = bbox[3].get<double>() * sy;
        } else if (md.contains("bbox_norm") && md["bbox_norm"].is_array() && md["bbox_norm"].size() >= 4) {
            const auto& bbox = md["bbox_norm"];
            x1 = bbox[0].get<double>() * (double)frame_.width;
            y1 = bbox[1].get<double>() * (double)frame_.height;
            x2 = bbox[2].get<double>() * (double)frame_.width;
            y2 = bbox[3].get<double>() * (double)frame_.height;
        } else {
            return false;
        }

        int bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
        if (!scaleAndClampBBox(x1, y1, x2, y2, frame_.width, frame_.height, bx1, by1, bx2, by2)) return false;
        boxes_out.push_back(box(bx1, by1, bx2, by2, DrawColor{}));
        return true;
    }

    void parseDetections(const Parameters& md, std::vector<BatchedBBox>& boxes_out) const {
        std::vector<ParsedYoloDetection> detections;
        parseYoloDetections(md, filter_.config(frame_), detections);
        for (const auto& det : detections) {
            const bool is_predicted = det.has_predicted && det.predicted;
            if (predicted_only_ && !is_predicted) continue;
            if (hide_predicted_ && is_predicted) continue;
            boxes_out.push_back(box(det.x1, det.y1, det.x2, det.y2, resolveModelColor(det)));
        }
    }

public:
    static BBoxItems fromParams(const Parameters& params, const std::string& node) {
        BBoxItems items;
        items.node_ = node;
        items.metadata_keys_ = metadataKeysParam(params, "reframer_bbox", node);
        items.bbox_thickness_ = params.value("bbox_thickness", 2);
        items.debug_log_every_n_ = params.value("debug_log_every_n", 0);
        items.filter_ = DetectionFilter::fromParams(params);
        if (params.count("model_colors") && params["model_colors"].is_object()) {
            for (auto it = params["model_colors"].begin(); it != params["model_colors"].end(); ++it) {
                DrawColor color;
                if (!it.value().is_string() || !tryParseNamedColor(it.value().get<std::string>(), color)) {
                    throw Error(node + ": model_colors values must be a named color");
                }
                items.model_colors_[std::stoi(it.key())] = color;
            }
        }
        if (params.count("label_colors") && params["label_colors"].is_object()) {
            for (auto it = params["label_colors"].begin(); it != params["label_colors"].end(); ++it) {
                DrawColor color;
                if (!it.value().is_string() || !tryParseNamedColor(it.value().get<std::string>(), color)) {
                    throw Error(node + ": label_colors values must be a named color");
                }
                items.label_colors_[it.key()] = color;
            }
        }
        if (params.count("predicted_color") && params["predicted_color"].is_string()) {
            DrawColor color;
            if (!tryParseNamedColor(params["predicted_color"].get<std::string>(), color)) {
                throw Error(node + ": predicted_color must be a named color");
            }
            items.predicted_color_ = color;
        }
        items.predicted_only_ = params.value("predicted_only", false);
        items.hide_predicted_ = params.value("hide_predicted", false);
        if (items.predicted_only_ && items.hide_predicted_) {
            throw Error(node + ": predicted_only and hide_predicted cannot both be true");
        }
        if (items.metadata_keys_.empty()) {
            throw Error(node + ": metadata_keys must be non-empty (or pass metadata_key)");
        }
        if (items.bbox_thickness_ <= 0) {
            throw Error(node + ": bbox_thickness must be positive");
        }
        const DetectionFilter& f = items.filter_;
        if ((f.model_content_width > 0.0 || f.model_content_height > 0.0)
                && !(f.model_content_width > 0.0 && f.model_content_height > 0.0)) {
            throw Error(node + ": model_content_width and model_content_height must both be positive when set");
        }
        if (f.model_content_offset_x < 0.0 || f.model_content_offset_y < 0.0) {
            throw Error(node + ": model_content offsets must be non-negative");
        }
        return items;
    }

    /// Appends the frame's boxes to `boxes`: the viewport rectangle first, then the boxes of each
    /// metadata key in key order. Not const: the viewport rectangle remembers its last position.
    void collect(FramePayloads& frame_payloads, const FrameGeometry& frame, uint64_t frame_counter,
                 std::vector<BatchedBBox>& boxes) {
        frame_ = frame;
        const bool log = debug_log_every_n_ > 0 && (frame_counter % (uint64_t)debug_log_every_n_) == 0;
        std::vector<const Parameters*> payloads;
        for (const std::string& key : metadata_keys_) {
            if (const Parameters* md = frame_payloads.get(key)) payloads.push_back(md);
        }

        int vdw = 0;
        int vdh = 0;
        if (tryReadViewportDstDims(payloads, vdw, vdh)) {
            if (frame_.width != last_viewport_input_w_ || frame_.height != last_viewport_input_h_) {
                have_last_viewport_crop_ = false;
                last_viewport_input_w_ = frame_.width;
                last_viewport_input_h_ = frame_.height;
            }
            if (vdw != last_viewport_dst_w_ || vdh != last_viewport_dst_h_) {
                have_last_viewport_crop_ = false;
                last_viewport_dst_w_ = vdw;
                last_viewport_dst_h_ = vdh;
            }
            int vx = 0;
            int vy = 0;
            updateViewportCropPosition(payloads, vdw, vdh, vx, vy);
            DrawColor white{};
            if (!tryParseNamedColor("white", white)) {
                white = DrawColor{235, 128, 128};
            }
            const BatchedBBox viewport = box(clampInt(vx, 0, frame_.width), clampInt(vy, 0, frame_.height),
                                             clampInt(vx + vdw, 0, frame_.width), clampInt(vy + vdh, 0, frame_.height),
                                             white);
            if (viewport.x2 > viewport.x1 && viewport.y2 > viewport.y1) {
                boxes.push_back(viewport);
                if (log) {
                    logstream << node_ << ": frame=" << frame_counter
                              << " viewport=[" << viewport.x1 << "," << viewport.y1
                              << "," << viewport.x2 << "," << viewport.y2 << "]";
                }
            }
        } else if (log) {
            logstream << node_ << ": frame=" << frame_counter << " no viewport dims found";
        }

        const size_t before = boxes.size();
        for (const Parameters* md : payloads) {
            try {
                if (!parseSingleBBoxMetadata(*md, boxes)) parseDetections(*md, boxes);
            } catch (const std::exception&) {
                continue;      // a malformed payload gives no boxes; the other keys still draw
            }
        }
        if (!log) return;
        if (boxes.size() == before) {
            logstream << node_ << ": frame=" << frame_counter << " no bbox metadata";
            return;
        }
        const BatchedBBox& first = boxes[before];
        logstream << node_ << ": frame=" << frame_counter
                  << " boxes=" << (boxes.size() - before)
                  << " first_bbox=[" << first.x1 << "," << first.y1 << "," << first.x2 << "," << first.y2 << "]"
                  << " thickness=" << bbox_thickness_;
    }
};

} // namespace cuda_overlay
