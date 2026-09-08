#pragma once
// Minimal TensorRT stub for CPU unit tests of the pure YOLO decode logic.
// Provides only what decode_detection.hpp references (nvinfer1::Dims). The real
// <NvInfer.h> is used in production builds; this stub is picked up only when
// tests/cpp/stubs is placed ahead on the include path.
#include <cstdint>

namespace nvinfer1 {

struct Dims {
    static constexpr int32_t MAX_DIMS = 8;
    int32_t nbDims = 0;
    int32_t d[MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
};

} // namespace nvinfer1
