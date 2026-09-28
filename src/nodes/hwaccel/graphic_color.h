#pragma once
// RGB graphics to canvas codes, shared by the compositor kernel (cuda_rect_scale.cu) and the
// host (a mixer dip colour, compositor_color.hpp), so both land on the same codes. Luma and
// chroma come out at 8-bit limited-range scale; callers multiply by 1 << (depth - 8).
#include <math.h>

#ifdef __CUDACC__
#define AVP_HD __host__ __device__
#else
#define AVP_HD
#endif

// Embed SDR RGB graphics at the same display white as tonemap_cuda. Transfer
// codes match that filter: HLG=0, PQ=1, SDR=2. Alpha stays separate.
AVP_HD static inline float pq_code(float nits) {
    const float p = powf(fmaxf(nits, 0.f) / 10000.f, 0.1593017578125f);
    return powf((0.8359375f + 18.8515625f * p) / (1.f + 18.6875f * p), 78.84375f);
}

AVP_HD static inline void convert_graphic_rgb(float *rgb, int transfer, float white, float peak) {
    if (transfer == 2) return;
    float r = powf(fmaxf(rgb[0] / 255.f, 0.f), 2.4f) * white;
    float g = powf(fmaxf(rgb[1] / 255.f, 0.f), 2.4f) * white;
    float b = powf(fmaxf(rgb[2] / 255.f, 0.f), 2.4f) * white;
    rgb[0] = 0.627404f * r + 0.329283f * g + 0.043313f * b;
    rgb[1] = 0.069097f * r + 0.919540f * g + 0.011362f * b;
    rgb[2] = 0.016391f * r + 0.088013f * g + 0.895595f * b;
    if (transfer == 1) {
        for (int i = 0; i < 3; ++i) rgb[i] = 255.f * pq_code(rgb[i]);
        return;
    }
    const float luma = 0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2];
    const float gamma = fmaxf(1.f, 1.2f + 0.42f * log10f(peak / 1000.f));
    const float gain = luma > 0.f ? 12.f * powf(luma / peak, 1.f / gamma) / luma : 0.f;
    for (int i = 0; i < 3; ++i) {
        const float c = fmaxf(rgb[i] * gain, 0.f);
        rgb[i] = 255.f * (c <= 1.f ? 0.5f * sqrtf(c) : 0.17883277f * logf(c - 0.28466892f) + 0.55991073f);
    }
}

AVP_HD static inline float graphic_luma(const float *rgb, int transfer) {
    return transfer == 2 ? 16.f + 0.1826f * rgb[0] + 0.6142f * rgb[1] + 0.0620f * rgb[2]
        : 16.f + (219.f / 255.f) * (0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2]);
}

AVP_HD static inline void graphic_chroma(float r, float g, float b, int transfer, float &cb, float &cr) {
    if (transfer == 2) {
        cb = 128.f - 0.1006f * r - 0.3386f * g + 0.4392f * b;
        cr = 128.f + 0.4392f * r - 0.3989f * g - 0.0403f * b;
    } else {
        const float y = 0.2627f * r + 0.6780f * g + 0.0593f * b;
        cb = 128.f + (224.f / 255.f) * (b - y) / 1.8814f;
        cr = 128.f + (224.f / 255.f) * (r - y) / 1.4746f;
    }
}

#ifndef __CUDACC__
// Host side only: the kernel image is built without FFmpeg include paths.
extern "C" {
#include <libavutil/pixfmt.h>
}
#include <string>

namespace avp::mixer {

/// Compositor defaults in nits: where SDR graphics white sits on an HDR canvas, and the HDR peak.
constexpr float kGraphicSdrWhite = 203.f;
constexpr float kGraphicHdrPeak = 1000.f;

/// Transfer id the RGB conversion takes: 0 = HLG, 1 = PQ, 2 = SDR (BT.709).
inline int kernelTransfer(AVColorTransferCharacteristic trc) {
    return trc == AVCOL_TRC_ARIB_STD_B67 ? 0 : trc == AVCOL_TRC_SMPTE2084 ? 1 : 2;
}

/// A canvas `color` parameter (sdr, hlg or pq) as its transfer; unspecified for anything else.
inline AVColorTransferCharacteristic graphicTransfer(const std::string &color) {
    return color == "sdr" ? AVCOL_TRC_BT709 :
        color == "hlg" ? AVCOL_TRC_ARIB_STD_B67 : color == "pq" ? AVCOL_TRC_SMPTE2084 : AVCOL_TRC_UNSPECIFIED;
}

}  // namespace avp::mixer
#endif
