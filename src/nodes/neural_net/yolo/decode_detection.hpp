#pragma once
#include <NvInfer.h>  // for nvinfer1::Dims
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "../common/decode_types.hpp"

namespace yolo_base {

class DetectionDecoder {
    static float iou(const Detection& a, const Detection& b) {
        float x1 = std::max(a.x1, b.x1);
        float y1 = std::max(a.y1, b.y1);
        float x2 = std::min(a.x2, b.x2);
        float y2 = std::min(a.y2, b.y2);
        float w = std::max(0.0f, x2 - x1);
        float h = std::max(0.0f, y2 - y1);
        float inter = w * h;
        float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
        float uni = area_a + area_b - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }

    static void applyNms(std::vector<Detection>& dets, float iou_thresh, bool class_agnostic) {
        if (iou_thresh <= 0.0f || dets.size() < 2) return;

        std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

        std::vector<uint8_t> suppressed(dets.size(), 0);
        std::vector<Detection> kept;
        kept.reserve(dets.size());

        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            kept.push_back(dets[i]);
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;
                if (!class_agnostic && dets[i].cls != dets[j].cls) continue;
                if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
            }
        }

        dets = std::move(kept);
    }

public:
    DetectionResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params
    ) {
        DetectionResult result;
        if (host_outputs.empty() || !host_outputs[0]) return result;

        const float* out = host_outputs[0];
        const nvinfer1::Dims& d = output_dims[0];

        int count = 0, attrs = 0;
        bool attrs_first = false;

        // Determine layout from dims.
        // In YOLO outputs the attrs dimension (6, 84, etc.) is always smaller
        // than the count dimension (100, 8400, etc.), so pick the smaller dim as attrs.
        if (d.nbDims == 2) {
            int a = d.d[0], b = d.d[1];
            if (a <= b && a >= 6) { attrs = a; count = b; attrs_first = true; }
            else if (b >= 6) { attrs = b; count = a; attrs_first = false; }
            else if (a >= 6) { attrs = a; count = b; attrs_first = true; }
            else return result;
        } else if (d.nbDims == 3 && d.d[0] == 1) {
            int d1 = d.d[1], d2 = d.d[2];
            if (d1 <= d2 && d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else if (d2 >= 6) { attrs = d2; count = d1; attrs_first = false; }
            else if (d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else return result;
        } else {
            return result;
        }

        if (count <= 0 || attrs < 6) return result;

        // Unified accessor: handles both memory layouts
        auto at = [&](int det, int attr) -> float {
            return attrs_first ? out[attr * count + det] : out[det * attrs + attr];
        };

        for (int i = 0; i < count; ++i) {
            Detection det;
            det.model_index = params.model_index;

            if (params.box_format == OutputBoxFormat::EndToEndXYXY) {
                det.x1 = at(i, 0); det.y1 = at(i, 1);
                det.x2 = at(i, 2); det.y2 = at(i, 3);
                det.conf = at(i, 4);
                det.cls = (int)std::round(at(i, 5));
            } else { // RawCXCYWH
                float cx = at(i, 0), cy = at(i, 1);
                float w = at(i, 2), h = at(i, 3);
                int best_cls = 0;
                float best = 0.0f;
                for (int c = 4; c < attrs; ++c) {
                    float s = at(i, c);
                    if (s > best) { best = s; best_cls = c - 4; }
                }
                det.x1 = cx - w * 0.5f; det.y1 = cy - h * 0.5f;
                det.x2 = cx + w * 0.5f; det.y2 = cy + h * 0.5f;
                det.conf = best;
                det.cls = best_cls;
            }

            // Rescale normalized [0,1] coordinates to model-space pixels.
            if (params.boxes_normalized && params.model_w > 0 && params.model_h > 0) {
                det.x1 *= (float)params.model_w;
                det.x2 *= (float)params.model_w;
                det.y1 *= (float)params.model_h;
                det.y2 *= (float)params.model_h;
            }

            if (det.conf < params.conf_thresh) continue;

            // Apply class index remapping
            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
                if (det.cls < 0) continue;
            }

            result.detections.push_back(det);
        }

        applyNms(result.detections, params.nms_iou_thresh, params.nms_class_agnostic);

        return result;
    }
};

} // namespace yolo_base
