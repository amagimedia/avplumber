#pragma once
// Whether a pixel format is FFmpeg's opaque CUDA array (CUarray) storage, without CUDA headers.

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>
}

namespace avp::cuda {

inline bool isArrayFormat(AVPixelFormat format) {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(61, 5, 100)
    return format == AV_PIX_FMT_CUARRAY;
#else
    (void)format;
    return false;
#endif
}

}
