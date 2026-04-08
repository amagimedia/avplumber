#pragma once
#include "../common/infer_trt_base.hpp"

namespace yolo_base {

class DetectionDecoder {
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

            if (det.conf < params.conf_thresh) continue;

            // Apply class index remapping
            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }

            result.detections.push_back(det);
        }

        return result;
    }
};

} // namespace yolo_base
