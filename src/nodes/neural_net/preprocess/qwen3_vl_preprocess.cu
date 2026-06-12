// NV12 CUDA frame -> Qwen3-VL normalized NCHW frame tensor.

#include <cuda_fp16.h>
#include "nv12_rgb_cuda.cuh"

__device__ __forceinline__ int nchw_base_index(
    int out_x, int out_y, int sample_index, int channel, int target_width, int target_height)
{
    const int frame_stride = 3 * target_width * target_height;
    const int channel_stride = target_width * target_height;
    return sample_index * frame_stride + channel * channel_stride + out_y * target_width + out_x;
}

extern "C" __global__ void kQwen3VlPreprocessNV12_fp16(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    __half* __restrict__ out,
    int sample_index,
    int src_width, int src_height,
    int target_width, int target_height)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = target_width * target_height;
    if (idx >= total) return;

    const int out_x = idx % target_width;
    const int out_y = idx / target_width;

    const float sx = ((float)out_x + 0.5f) * ((float)src_width / (float)target_width) - 0.5f;
    const float sy = ((float)out_y + 0.5f) * ((float)src_height / (float)target_height) - 0.5f;

    float r, g, b;
    sample_nv12_rgb_bilinear(
        y_plane, pitch_y, uv_plane, pitch_uv,
        src_width, src_height, sx, sy,
        r, g, b);

    out[nchw_base_index(out_x, out_y, sample_index, 0, target_width, target_height)] = __float2half_rn(r * 2.0f - 1.0f);
    out[nchw_base_index(out_x, out_y, sample_index, 1, target_width, target_height)] = __float2half_rn(g * 2.0f - 1.0f);
    out[nchw_base_index(out_x, out_y, sample_index, 2, target_width, target_height)] = __float2half_rn(b * 2.0f - 1.0f);
}

extern "C" __global__ void kQwen3VlPreprocessNV12_fp32(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    float* __restrict__ out,
    int sample_index,
    int src_width, int src_height,
    int target_width, int target_height)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = target_width * target_height;
    if (idx >= total) return;

    const int out_x = idx % target_width;
    const int out_y = idx / target_width;

    const float sx = ((float)out_x + 0.5f) * ((float)src_width / (float)target_width) - 0.5f;
    const float sy = ((float)out_y + 0.5f) * ((float)src_height / (float)target_height) - 0.5f;

    float r, g, b;
    sample_nv12_rgb_bilinear(
        y_plane, pitch_y, uv_plane, pitch_uv,
        src_width, src_height, sx, sy,
        r, g, b);

    out[nchw_base_index(out_x, out_y, sample_index, 0, target_width, target_height)] = r * 2.0f - 1.0f;
    out[nchw_base_index(out_x, out_y, sample_index, 1, target_width, target_height)] = g * 2.0f - 1.0f;
    out[nchw_base_index(out_x, out_y, sample_index, 2, target_width, target_height)] = b * 2.0f - 1.0f;
}
