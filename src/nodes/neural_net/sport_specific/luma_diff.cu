#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kLumaDiffReduce(
    const uint8_t* __restrict__ y_plane,
    int y_pitch_bytes,
    uint8_t* __restrict__ prev_y,
    int width,
    int height,
    int has_prev,
    float* __restrict__ block_abs,
    float* __restrict__ block_signed)
{
    extern __shared__ float shared[];
    float* s_abs = shared;
    float* s_signed = shared + blockDim.x;

    const int tid = (int)threadIdx.x;
    const int n = width * height;
    const int stride = (int)(blockDim.x * gridDim.x);
    float local_abs = 0.0f;
    float local_signed = 0.0f;

    for (int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x); idx < n; idx += stride) {
        const int y = idx / width;
        const int x = idx - y * width;
        const uint8_t cur = y_plane[(size_t)y * (size_t)y_pitch_bytes + (size_t)x];
        if (has_prev) {
            const float diff = (float)((int)cur - (int)prev_y[idx]);
            local_abs += fabsf(diff);
            local_signed += diff;
        }
        prev_y[idx] = cur;
    }

    s_abs[tid] = local_abs;
    s_signed[tid] = local_signed;
    __syncthreads();

    for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_abs[tid] += s_abs[tid + offset];
            s_signed[tid] += s_signed[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_abs[blockIdx.x] = s_abs[0];
        block_signed[blockIdx.x] = s_signed[0];
    }
}
