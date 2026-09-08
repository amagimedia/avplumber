// CUDA kernels for the HogDiff scene-cut metric.
//
// Per frame we downscale the 8-bit luma plane to an analysis size and compute a
// Dalal-Triggs-style HOG descriptor on it, reduced to a fixed-length vector. The
// host then computes the L1 distance between descriptors of two frames and emits
// it as the `scene_diff` metadata consumed by the recorder's LumaSceneCutDecider
// (identical contract to LumaDiff). Five kernel entry points are exported:
//
//   kHogDownscale - bilinear luma downscale to the analysis plane.
//   kHogCells     - per-cell magnitude-weighted orientation histograms.
//   kHogBlocks    - block aggregation + normalization -> descriptor vector.
//   kHogL1Reduce  - L1 distance reduction between two descriptors.
//   kHogL1Sum     - L1 norm reduction of one descriptor (for mean_norm).

#include <stdint.h>
#include <cuda_runtime.h>

namespace {

__device__ inline uint8_t readY(const uint8_t* y_plane, int pitch, int x, int y,
                                 int w, int h) {
    if (x < 0) x = 0; else if (x >= w) x = w - 1;
    if (y < 0) y = 0; else if (y >= h) y = h - 1;
    return y_plane[(size_t)y * (size_t)pitch + (size_t)x];
}

__device__ inline float sampleY(const uint8_t* y_plane, int pitch, int x, int y,
                                 int w, int h, float gamma) {
    float v = (float)readY(y_plane, pitch, x, y, w, h);
    if (gamma > 0.0f) v = powf(v / 255.0f, gamma) * 255.0f;
    return v;
}

// Bilinear downscale of a luma plane to a compact (dst_w x dst_h) buffer with
// pitch == dst_w. Used to bound HOG cost while forwarding the original frame.
// grid/block: 2D over the destination.
extern "C" __global__ void kHogDownscale(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    int dst_w, int dst_h, uint8_t* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 >= src_w) x0 = src_w - 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y0 >= src_h) y0 = src_h - 1;
    if (y1 >= src_h) y1 = src_h - 1;
    const float v00 = src[(size_t)y0 * (size_t)src_pitch + (size_t)x0];
    const float v01 = src[(size_t)y0 * (size_t)src_pitch + (size_t)x1];
    const float v10 = src[(size_t)y1 * (size_t)src_pitch + (size_t)x0];
    const float v11 = src[(size_t)y1 * (size_t)src_pitch + (size_t)x1];
    const float v = (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01
                 + (1.0f - fx) * fy * v10 + fx * fy * v11;
    dst[(size_t)y * (size_t)dst_w + (size_t)x] = (uint8_t)(v + 0.5f);
}

__device__ inline void blockReduceSum(float* s, int tid, int blockDimX) {
    for (int off = blockDimX >> 1; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
}

} // namespace

// Kernel 1: per-cell orientation histograms.
// grid: (num_cells, 1, 1), block: (threads, 1, 1), shared: num_bins * sizeof(float)
extern "C" __global__ void kHogCells(
    const uint8_t* __restrict__ y_plane, int pitch, int width, int height,
    int cell_w, int cell_h, int cells_x, int num_bins, int signed_ori, float gamma,
    float* __restrict__ cell_hists)
{
    const int cell_idx = blockIdx.x;
    const int cx = cell_idx % cells_x;
    const int cy = cell_idx / cells_x;
    const int x0 = cx * cell_w;
    const int y0 = cy * cell_h;
    const int tid = threadIdx.x;

    extern __shared__ float s_hist[];  // size = num_bins
    for (int b = tid; b < num_bins; b += blockDim.x) s_hist[b] = 0.0f;
    __syncthreads();

    const float bin_width = (signed_ori ? 360.0f : 180.0f) / (float)num_bins;
    const int cell_pixels = cell_w * cell_h;
    for (int p = tid; p < cell_pixels; p += blockDim.x) {
        const int dx = p % cell_w;
        const int dy = p / cell_w;
        const int x = x0 + dx;
        const int y = y0 + dy;
        if (x >= width || y >= height) continue;
        const float gx = sampleY(y_plane, pitch, x + 1, y, width, height, gamma)
                       - sampleY(y_plane, pitch, x - 1, y, width, height, gamma);
        const float gy = sampleY(y_plane, pitch, x, y + 1, width, height, gamma)
                       - sampleY(y_plane, pitch, x, y - 1, width, height, gamma);
        const float mag = sqrtf(gx * gx + gy * gy);
        float ang = atan2f(gy, gx) * (180.0f / 3.14159265358979323846f);  // [-180,180]
        if (ang < 0.0f) ang += 360.0f;                                    // [0,360)
        if (!signed_ori && ang >= 180.0f) ang -= 180.0f;                  // [0,180)
        int bin = (int)(ang / bin_width);
        if (bin < 0) bin = 0; else if (bin >= num_bins) bin = num_bins - 1;
        atomicAdd(&s_hist[bin], mag);
    }
    __syncthreads();

    for (int b = tid; b < num_bins; b += blockDim.x) {
        cell_hists[(size_t)cell_idx * (size_t)num_bins + (size_t)b] = s_hist[b];
    }
}

// Kernel 2: block aggregation + normalization -> descriptor.
// grid: (num_blocks, 1, 1), block: (threads, 1, 1), shared: B * sizeof(float)
//   B = block_size_cells * block_size_cells * num_bins
// norm_mode: 0 = L2, 1 = L1, 2 = L2-Hys
extern "C" __global__ void kHogBlocks(
    const float* __restrict__ cell_hists, int cells_x,
    int block_size_cells, int block_stride_cells, int num_bins, int norm_mode,
    float* __restrict__ desc)
{
    const int blk_idx = blockIdx.x;
    const int blocks_x = (cells_x - block_size_cells) / block_stride_cells + 1;
    const int bx = blk_idx % blocks_x;
    const int by = blk_idx / blocks_x;
    const int cell_x0 = bx * block_stride_cells;
    const int cell_y0 = by * block_stride_cells;
    const int B = block_size_cells * block_size_cells * num_bins;
    const int tid = threadIdx.x;

    extern __shared__ float s_vec[];  // size = B
    for (int i = tid; i < B; i += blockDim.x) {
        int local = i;
        const int b = local % num_bins; local /= num_bins;
        const int lcx = local % block_size_cells; local /= block_size_cells;
        const int lcy = local;
        const int gx = cell_x0 + lcx;
        const int gy = cell_y0 + lcy;
        s_vec[i] = cell_hists[((size_t)gy * (size_t)cells_x + (size_t)gx) * (size_t)num_bins + (size_t)b];
    }
    __syncthreads();

    // Norm computation is serial on tid==0; B is small (a few hundred at most).
    if (tid == 0) {
        float nrm = 0.0f;
        if (norm_mode == 1) {
            for (int i = 0; i < B; ++i) nrm += fabsf(s_vec[i]);
        } else {
            for (int i = 0; i < B; ++i) nrm += s_vec[i] * s_vec[i];
            nrm = sqrtf(nrm);
        }
        float inv = 1.0f / (nrm + 1e-6f);
        for (int i = 0; i < B; ++i) s_vec[i] *= inv;
        if (norm_mode == 2) {  // L2-Hys: clip then renormalize
            for (int i = 0; i < B; ++i) s_vec[i] = fminf(s_vec[i], 0.2f);
            nrm = 0.0f;
            for (int i = 0; i < B; ++i) nrm += s_vec[i] * s_vec[i];
            nrm = sqrtf(nrm);
            inv = 1.0f / (nrm + 1e-6f);
            for (int i = 0; i < B; ++i) s_vec[i] *= inv;
        }
    }
    __syncthreads();

    for (int i = tid; i < B; i += blockDim.x) {
        desc[(size_t)blk_idx * (size_t)B + (size_t)i] = s_vec[i];
    }
}

// Kernel 3: L1 distance reduction between two descriptors a and b.
// grid: (blocks, 1, 1), block: (threads, 1, 1), shared: threads * sizeof(float)
extern "C" __global__ void kHogL1Reduce(
    const float* __restrict__ a, const float* __restrict__ b, int n,
    float* __restrict__ block_out)
{
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += blockDim.x * gridDim.x) {
        local += fabsf(a[idx] - b[idx]);
    }
    extern __shared__ float s[];
    s[tid] = local;
    __syncthreads();
    blockReduceSum(s, tid, blockDim.x);
    if (tid == 0) block_out[blockIdx.x] = s[0];
}

// Kernel 4: L1 norm reduction of a single descriptor (for mean_norm).
extern "C" __global__ void kHogL1Sum(
    const float* __restrict__ a, int n, float* __restrict__ block_out)
{
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += blockDim.x * gridDim.x) {
        local += fabsf(a[idx]);
    }
    extern __shared__ float s[];
    s[tid] = local;
    __syncthreads();
    blockReduceSum(s, tid, blockDim.x);
    if (tid == 0) block_out[blockIdx.x] = s[0];
}
