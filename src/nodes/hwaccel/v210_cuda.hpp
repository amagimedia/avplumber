#pragma once
// Shared by the two halves of the packed-v210 GPU bridge, v210_to_cuda and
// cuda_to_v210: both talk to the driver API directly, and both have to agree
// on what a v210 frame looks like in memory.
#include "../node_common.hpp"
#include "../../cuda.hpp"

extern "C" {
#include <libavutil/imgutils.h>
}

#include <limits>
#include <string>

namespace v210cuda {

// The node name is part of the message because a round trip runs both nodes in
// one process, and a driver error otherwise says nothing about which end failed.
inline void check(CUresult result, const char* node, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* description = nullptr;
    if (cuGetErrorString) cuGetErrorString(result, &description);
    throw Error(std::string(node) + ": " + operation + ": " +
                (description ? description : std::to_string(result)));
}

class CurrentContext {
public:
    CurrentContext(CUcontext context, const char* node) {
        check(cuCtxPushCurrent(context), node, "push context");
    }
    ~CurrentContext() {
        CUcontext previous;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
};

// Packed v210 carries no header, so a frame is exactly stride * height bytes.
struct Layout {
    int width;
    int height;
    int stride;
    size_t size;
};

// Six pixels per 16 bytes, rows padded to 128 bytes: the pitch that SDI-style
// hardware and the libavcodec v210 codecs use.
inline int defaultStride(int width) {
    return ((width + 47) / 48) * 128;
}

// A non-positive stride means "use the default". The minimum is the packed
// width alone, which is what libavcodec writes before it pads the row out.
inline Layout makeLayout(const char* node, int width, int height, int64_t stride) {
    if (width <= 0 || height <= 0 || width % 2 ||
        av_image_check_size(width, height, 0, nullptr) < 0)
        throw Error(std::string(node) + ": requires valid dimensions and an even width");
    if (stride <= 0) stride = defaultStride(width);
    const int64_t minimum_stride = ((int64_t(width) * 2 + 2) / 3) * 4;
    if (stride < minimum_stride || stride % 4 ||
        stride > std::numeric_limits<int>::max() / height)
        throw Error(std::string(node) + ": stride must be a multiple of four, fit a v210 row and an AVPacket");
    return Layout{width, height, static_cast<int>(stride), size_t(stride) * height};
}

// Frame rates, time bases and aspect ratios all come in as "num/den" strings,
// and a zero in either half would silently poison every timestamp downstream.
inline av::Rational positiveRatio(const std::string& value, const char* node) {
    auto ratio = parseRatio(value);
    if (ratio.getNumerator() <= 0 || ratio.getDenominator() <= 0)
        throw Error(std::string(node) + ": ratios must be positive");
    return ratio;
}

// The two ways a 10-bit 4:2:2 frame is stored on the GPU without resampling:
// semiplanar p210le, which keeps samples in the high bits, and planar
// yuv422p10le. Both kernels take the choice as a flag.
inline bool isSupportedFormat(AVPixelFormat format) {
    return format == AV_PIX_FMT_P210LE || format == AV_PIX_FMT_YUV422P10LE;
}

inline bool isSemiplanar(AVPixelFormat format) {
    return format == AV_PIX_FMT_P210LE;
}

} // namespace v210cuda
