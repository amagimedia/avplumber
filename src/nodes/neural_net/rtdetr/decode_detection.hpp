#pragma once

#include "../common/infer_trt_base.hpp"

namespace yolo_base {

class RTDetrDetectionDecoder {
public:
    // End-to-end contract: separate boxes[N,4], scores[N], labels[N] tensors
    DetectionResult decode(
        const float* boxes,
        const float* scores,
        const float* labels,
        int det_count,
        const DecodeParams& params
    ) const {
        DetectionResult result;
        if (!boxes || !scores || !labels || det_count <= 0) return result;
        result.detections.reserve((size_t)det_count);

        for (int i = 0; i < det_count; ++i) {
            Detection det;
            det.model_index = params.model_index;
            det.x1 = boxes[(size_t)i * 4 + 0];
            det.y1 = boxes[(size_t)i * 4 + 1];
            det.x2 = boxes[(size_t)i * 4 + 2];
            det.y2 = boxes[(size_t)i * 4 + 3];
            det.conf = scores[i];
            det.cls = (int)std::round(labels[i]);

            if (!std::isfinite(det.conf) || det.conf < params.conf_thresh) continue;
            if (!std::isfinite(det.x1) || !std::isfinite(det.y1) ||
                !std::isfinite(det.x2) || !std::isfinite(det.y2)) {
                continue;
            }

            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }
            result.detections.push_back(det);
        }
        return result;
    }

    // Combined contract: single output[N, 4+num_classes] with cxcywh boxes and class scores.
    // Boxes are in normalized 0-1 coords; scale_w/scale_h convert to pixel coords in model space.
    DetectionResult decodeCombined(
        const float* data,
        int det_count,
        int num_classes,
        float scale_w,
        float scale_h,
        const DecodeParams& params
    ) const {
        DetectionResult result;
        if (!data || det_count <= 0 || num_classes <= 0) return result;
        result.detections.reserve((size_t)det_count);
        int stride = 4 + num_classes;

        for (int i = 0; i < det_count; ++i) {
            const float* row = data + (size_t)i * stride;
            float cx = row[0], cy = row[1], w = row[2], h = row[3];

            // Find best class
            int best_cls = 0;
            float best_score = row[4];
            for (int c = 1; c < num_classes; ++c) {
                if (row[4 + c] > best_score) {
                    best_score = row[4 + c];
                    best_cls = c;
                }
            }

            if (!std::isfinite(best_score) || best_score < params.conf_thresh) continue;
            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(w) || !std::isfinite(h)) {
                continue;
            }

            Detection det;
            det.model_index = params.model_index;
            det.x1 = (cx - w * 0.5f) * scale_w;
            det.y1 = (cy - h * 0.5f) * scale_h;
            det.x2 = (cx + w * 0.5f) * scale_w;
            det.y2 = (cy + h * 0.5f) * scale_h;
            det.conf = best_score;
            det.cls = best_cls;

            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }
            result.detections.push_back(det);
        }
        return result;
    }
};

} // namespace yolo_base
