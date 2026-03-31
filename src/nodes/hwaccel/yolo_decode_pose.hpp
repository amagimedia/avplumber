#pragma once
#include "cuda_infer_yolo_base.hpp"

namespace yolo_base {

class PoseDecoder {
private:
    int num_classes_ = 1;
    float nms_iou_thresh_ = 0.45f;

    static float iou(const Detection& a, const Detection& b) {
        float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
        float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
        float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float uni = area_a + area_b - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }

public:
    explicit PoseDecoder(int num_classes, float nms_iou_thresh = 0.45f)
        : num_classes_(num_classes), nms_iou_thresh_(nms_iou_thresh) {}

    PoseResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params
    ) {
        PoseResult result;
        if (host_outputs.empty() || !host_outputs[0]) return result;

        const float* out = host_outputs[0];
        const nvinfer1::Dims& d = output_dims[0];

        int count = 0, attrs = 0;
        bool attrs_first = false;

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

        auto at = [&](int det, int attr) -> float {
            return attrs_first ? out[attr * count + det] : out[det * attrs + attr];
        };

        // Layout depends on box format:
        //   EndToEndXYXY: [x1, y1, x2, y2, conf, cls] + [num_keypoints * 3]  (kpt_start = 6)
        //   RawCXCYWH:    [cx, cy, w, h] + [num_classes scores] + [num_keypoints * 3]
        int kpt_start = (params.box_format == OutputBoxFormat::EndToEndXYXY) ? 6 : 4 + num_classes_;
        int kpt_total_attrs = attrs - kpt_start;
        if (kpt_total_attrs < 3 || kpt_total_attrs % 3 != 0) return result;
        result.num_keypoints = kpt_total_attrs / 3;

        // Collect all candidates (pre-NMS)
        struct Candidate {
            Detection det;
            int raw_index;  // index into the raw output for keypoint extraction
        };
        std::vector<Candidate> candidates;

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
                for (int c = 4; c < kpt_start; ++c) {
                    float s = at(i, c);
                    if (s > best) { best = s; best_cls = c - 4; }
                }
                det.x1 = cx - w * 0.5f; det.y1 = cy - h * 0.5f;
                det.x2 = cx + w * 0.5f; det.y2 = cy + h * 0.5f;
                det.conf = best;
                det.cls = best_cls;
            }

            if (det.conf < params.conf_thresh) continue;

            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }

            candidates.push_back({det, i});
        }

        // Sort by confidence descending
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.det.conf > b.det.conf; });

        // NMS
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (suppressed[i]) continue;
            result.detections.push_back(candidates[i].det);

            // Extract keypoints for this surviving detection
            for (int k = 0; k < kpt_total_attrs; ++k) {
                result.keypoints.push_back(at(candidates[i].raw_index, kpt_start + k));
            }

            // Suppress overlapping detections
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (suppressed[j]) continue;
                if (candidates[i].det.cls == candidates[j].det.cls &&
                    iou(candidates[i].det, candidates[j].det) > nms_iou_thresh_) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }
};

} // namespace yolo_base
