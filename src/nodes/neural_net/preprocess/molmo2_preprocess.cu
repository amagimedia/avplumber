// NV12 CUDA frame -> Molmo2 patch-packed RGB tensor.
//
// Output layout matches the Molmo2 processor's batch_pixels_to_patches result:
// [frame, patch_y * patch_grid + patch_x, ((dy * patch_size + dx) * 3 + c)].

#include <cuda_fp16.h>
#include "nv12_rgb_cuda.cuh"

__device__ __forceinline__ int patch_packed_base_index(
    int out_x, int out_y, int sample_index, int frame_size, int patch_size)
{
    const int patch_grid = frame_size / patch_size;
    const int patch_x = out_x / patch_size;
    const int patch_y = out_y / patch_size;
    const int dx = out_x - patch_x * patch_size;
    const int dy = out_y - patch_y * patch_size;
    const int patch_index = patch_y * patch_grid + patch_x;
    const int patch_vec = patch_size * patch_size * 3;
    const int frame_stride = patch_grid * patch_grid * patch_vec;
    return sample_index * frame_stride + patch_index * patch_vec + (dy * patch_size + dx) * 3;
}

extern "C" __global__ void kMolmo2PreprocessNV12_fp16(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    __half* __restrict__ out,
    int sample_index,
    int src_width, int src_height,
    int frame_size, int patch_size)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = frame_size * frame_size;
    if (idx >= total) return;

    const int out_x = idx % frame_size;
    const int out_y = idx / frame_size;

    const float sx = ((float)out_x + 0.5f) * ((float)src_width / (float)frame_size) - 0.5f;
    const float sy = ((float)out_y + 0.5f) * ((float)src_height / (float)frame_size) - 0.5f;

    float r, g, b;
    sample_nv12_rgb_bilinear(
        y_plane, pitch_y, uv_plane, pitch_uv,
        src_width, src_height, sx, sy,
        r, g, b);

    const int base = patch_packed_base_index(out_x, out_y, sample_index, frame_size, patch_size);
    out[base + 0] = __float2half_rn(r * 2.0f - 1.0f);
    out[base + 1] = __float2half_rn(g * 2.0f - 1.0f);
    out[base + 2] = __float2half_rn(b * 2.0f - 1.0f);
}

extern "C" __global__ void kMolmo2PreprocessNV12_fp32(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    float* __restrict__ out,
    int sample_index,
    int src_width, int src_height,
    int frame_size, int patch_size)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = frame_size * frame_size;
    if (idx >= total) return;

    const int out_x = idx % frame_size;
    const int out_y = idx / frame_size;

    const float sx = ((float)out_x + 0.5f) * ((float)src_width / (float)frame_size) - 0.5f;
    const float sy = ((float)out_y + 0.5f) * ((float)src_height / (float)frame_size) - 0.5f;

    float r, g, b;
    sample_nv12_rgb_bilinear(
        y_plane, pitch_y, uv_plane, pitch_uv,
        src_width, src_height, sx, sy,
        r, g, b);

    const int base = patch_packed_base_index(out_x, out_y, sample_index, frame_size, patch_size);
    out[base + 0] = r * 2.0f - 1.0f;
    out[base + 1] = g * 2.0f - 1.0f;
    out[base + 2] = b * 2.0f - 1.0f;
}
