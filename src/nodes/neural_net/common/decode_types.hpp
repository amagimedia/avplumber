#pragma once

// Dependency-light POD types shared by the YOLO detection decoder and the
// TensorRT inference base. Deliberately free of TensorRT/CUDA/FFmpeg includes
// so the pure decoding logic in decode_detection.hpp can be unit-tested on the
// CPU without a GPU toolchain.

#include <vector>

namespace yolo_base {

// --- Enums ---
enum class OutputBoxFormat { EndToEndXYXY, RawCXCYWH };

// --- Detection struct ---
struct Detection {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float conf = 0.0f;
    int cls = -1;
    int model_index = -1;
};

// --- Decode results ---
struct DetectionResult {
    std::vector<Detection> detections;
};

struct DecodeParams {
    int model_index;
    float conf_thresh;
    OutputBoxFormat box_format;
    const std::vector<int>& class_index_remap;
    float nms_iou_thresh = 0.0f;
    bool nms_class_agnostic = false;
    // When true, decoded box coordinates are normalized to [0,1] (relative to the
    // model input) and are rescaled to model-space pixels using model_w/model_h.
    bool boxes_normalized = false;
    int model_w = 0;
    int model_h = 0;
};

} // namespace yolo_base
