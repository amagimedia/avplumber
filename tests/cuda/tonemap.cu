// CUDA HDR->SDR tone mapper, ported from FFmpeg n8.1 libavfilter/opencl/
// tonemap.cl + colorspace_common.cl. HLG or PQ BT.2020 -> BT.709 SDR, static
// peak (no per-frame detection). The device functions here are the body of the
// planned tonemap_cuda FFmpeg filter; tonemap_pixels is the offline test entry
// validated against tonemap_reference.py.
#include <cuda_runtime.h>

#define REFERENCE_WHITE 100.0f
#define ST2084_MAX_LUMINANCE 10000.0f
__device__ __constant__ float ST2084_M1 = 0.1593017578125f;
__device__ __constant__ float ST2084_M2 = 78.84375f;
__device__ __constant__ float ST2084_C1 = 0.8359375f;
__device__ __constant__ float ST2084_C2 = 18.8515625f;
__device__ __constant__ float ST2084_C3 = 18.6875f;
__device__ __constant__ float HLG_A = 0.17883277f;
__device__ __constant__ float HLG_B = 0.28466892f;
__device__ __constant__ float HLG_C = 0.55991073f;
#define SDR_AVG 0.25f

// Tone-map operators (transfer=0 direct,1 linear,2 clip,3 reinhard,4 hable,5 mobius).
__device__ static inline float hable_f(float in) {
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}
__device__ static float tone_op(int op, float s, float peak, float p) {
    switch (op) {
        case 1: return s * p / peak;                                   // linear
        case 2: return fminf(fmaxf(s * p, 0.0f), 1.0f);               // clip
        case 3: return s / (s + p) * (peak + p) / peak;              // reinhard
        case 4: return hable_f(s) / hable_f(peak);                    // hable
        case 5: {                                                      // mobius
            float j = p;
            if (s <= j) return s;
            float a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
            float b = (j * j - 2.0f * j * peak + peak) / fmaxf(peak - 1.0f, 1e-6f);
            return (b * b + 2.0f * b * j + j * j) / (b - a) * (s + a) / (s + b);
        }
        default: return s;                                            // direct
    }
}

// fast-math intrinsics: ~precision for speed on the SM-bound tonemap path
__device__ static inline float eotf_st2084(float x) {
    float p = __powf(fmaxf(x, 0.0f), 1.0f / ST2084_M2);
    float a = fmaxf(p - ST2084_C1, 0.0f);
    float b = fmaxf(ST2084_C2 - ST2084_C3 * p, 1e-6f);
    float c = __powf(a / b, 1.0f / ST2084_M1);
    return x > 0.0f ? c * ST2084_MAX_LUMINANCE / REFERENCE_WHITE : 0.0f;
}
// fast-math intrinsics: ~precision for speed on the SM-bound tonemap path
__device__ static inline float inverse_oetf_hlg(float x) {
    float a = 4.0f * x * x;
    float b = __expf((x - HLG_C) / HLG_A) + HLG_B;
    return x < 0.5f ? a : b;
}
// fast-math intrinsics: ~precision for speed on the SM-bound tonemap path
__device__ static inline float oetf_bt709(float c) {
    c = fmaxf(c, 0.0f);
    return c < 0.018f ? 4.5f * c : 1.099f * __powf(c, 0.45f) - 0.099f;
}
// BT.2020 luma for the HLG OOTF and BT.709 luma for desaturation.
__device__ static inline float luma_2020(float3 c) { return 0.2627f * c.x + 0.6780f * c.y + 0.0593f * c.z; }
__device__ static inline float luma_709(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// fast-math intrinsics: ~precision for speed on the SM-bound tonemap path
__device__ static inline float3 ootf_hlg(float3 c, float peak) {
    float luma = luma_2020(c);
    float gamma = fmaxf(1.0f, 1.2f + 0.42f * __log10f(peak * REFERENCE_WHITE / 1000.0f));
    float factor = peak * __powf(fmaxf(luma, 1e-6f), gamma - 1.0f) / __powf(12.0f, gamma);
    return make_float3(c.x * factor, c.y * factor, c.z * factor);
}

// BT.2020 limited-range 10-bit YCbCr code -> non-linear RGB in [0,1].
__device__ static inline float3 yuv2rgb_2020(float y10, float cb10, float cr10) {
    float y = (y10 / 1023.0f * 255.0f - 16.0f) / 219.0f;
    float u = (cb10 / 1023.0f * 255.0f - 128.0f) / 224.0f;
    float v = (cr10 / 1023.0f * 255.0f - 128.0f) / 224.0f;
    return make_float3(y + 1.4746f * v, y - 0.16455f * u - 0.57135f * v, y + 1.8814f * u);
}

// Linear BT.2020 -> linear BT.709 primaries.
__device__ static inline float3 lrgb2020_to_709(float3 c) {
    return make_float3(
        1.660491f * c.x - 0.587641f * c.y - 0.072850f * c.z,
        -0.124550f * c.x + 1.132900f * c.y - 0.008349f * c.z,
        -0.018151f * c.x - 0.100579f * c.y + 1.118730f * c.z);
}

// fast-math intrinsics: ~precision for speed on the SM-bound tonemap path
__device__ static inline float3 map_one_pixel_rgb(float3 rgb, float peak, float average,
                                                  int op, float param, float desat) {
    float sig = fmaxf(fmaxf(rgb.x, fmaxf(rgb.y, rgb.z)), 1e-6f);
    float sig_old = sig;
    float slope = fminf(1.0f, SDR_AVG / average);
    sig *= slope; peak *= slope;
    if (desat > 0.0f) {
        float luma = luma_709(rgb);
        float coeff = fmaxf(sig - 0.18f, 1e-6f) / fmaxf(sig, 1e-6f);
        coeff = __powf(coeff, 10.0f / desat);
        rgb = make_float3(rgb.x * (1 - coeff) + luma * coeff,
                          rgb.y * (1 - coeff) + luma * coeff,
                          rgb.z * (1 - coeff) + luma * coeff);
        sig = sig * (1 - coeff) + luma * slope * coeff;
    }
    sig = tone_op(op, sig, peak, param);
    sig = fminf(sig, 1.0f);
    float g = sig / sig_old;
    return make_float3(rgb.x * g, rgb.y * g, rgb.z * g);
}

// transfer: 0 = HLG, 1 = PQ. Returns BT.709 SDR non-linear RGB in [0,1].
__device__ static inline float3 tonemap_rgb(float3 src, int transfer, float peak,
                                            int op, float param, float desat) {
    float3 lin;
    if (transfer == 1) {
        lin = make_float3(eotf_st2084(src.x), eotf_st2084(src.y), eotf_st2084(src.z));
    } else {
        lin = make_float3(inverse_oetf_hlg(src.x), inverse_oetf_hlg(src.y), inverse_oetf_hlg(src.z));
        lin = ootf_hlg(lin, peak);
    }
    lin = lrgb2020_to_709(lin);
    lin = map_one_pixel_rgb(lin, peak, SDR_AVG, op, param, desat);
    return make_float3(oetf_bt709(fmaxf(lin.x, 0.0f)),
                       oetf_bt709(fmaxf(lin.y, 0.0f)),
                       oetf_bt709(fmaxf(lin.z, 0.0f)));
}

// Offline test entry: N pixels of BT.2020 HLG/PQ 10-bit codes -> BT.709 SDR
// 8-bit codes, per-pixel chroma (no subsampling), matching the oracle.
extern "C" __global__ void tonemap_pixels(
    const unsigned short *y10, const unsigned short *cb10, const unsigned short *cr10,
    unsigned char *y8, unsigned char *u8, unsigned char *v8,
    int n, int transfer, float peak, int op, float param, float desat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 src = yuv2rgb_2020((float)y10[i], (float)cb10[i], (float)cr10[i]);
    float3 s = tonemap_rgb(src, transfer, peak, op, param, desat);
    float y = 0.2126f * s.x + 0.7152f * s.y + 0.0722f * s.z;
    float u = (s.z - y) / 1.8556f;
    float v = (s.x - y) / 1.5748f;
    y8[i] = (unsigned char)fminf(fmaxf(rintf(219.0f * y + 16.0f), 0.0f), 255.0f);
    u8[i] = (unsigned char)fminf(fmaxf(rintf(224.0f * u + 128.0f), 0.0f), 255.0f);
    v8[i] = (unsigned char)fminf(fmaxf(rintf(224.0f * v + 128.0f), 0.0f), 255.0f);
}
