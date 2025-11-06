// BT.709 LIMITED input -> Linear full-range RGBA8 output written to a cudaArray via surface.
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
#include <stdint.h>

// Clamp helper
__device__ __forceinline__ uint8_t clamp8(float x) {
    x = x < 0.f ? 0.f : (x > 255.f ? 255.f : x);
    return (uint8_t)(x + 0.5f);
}

// Convert BT.709 limited YUV to linear RGB8 (full-range), applying inverse OETF (BT.709 gamma decode).
__device__ __forceinline__ float bt709_inverse_oetf(float x)
{
    // x in [0,1] (nonlinear), return linear value in [0,1]
    return (x < 0.081f) ? (x / 4.5f) : powf((x + 0.099f) / 1.099f, 1.0f / 0.45f);
}

__device__ __forceinline__ void yuv709lim_to_linear_rgb8(
    uint8_t Y, uint8_t U, uint8_t V,
    uint8_t& r, uint8_t& g, uint8_t& b)
{
    // Compute nonlinear R'G'B' in [0,1] from limited-range YUV
    const float y = (float)Y;
    const float u = (float)U - 128.0f;
    const float v = (float)V - 128.0f;
    const float C  = 1.164383f * (y - 16.0f) / 255.0f;
    float Rp = C + 1.792741f * (v / 255.0f);
    float Gp = C - 0.213249f * (u / 255.0f) - 0.532909f * (v / 255.0f);
    float Bp = C + 2.112402f * (u / 255.0f);
    // Clamp to [0,1]
    Rp = Rp < 0.f ? 0.f : (Rp > 1.f ? 1.f : Rp);
    Gp = Gp < 0.f ? 0.f : (Gp > 1.f ? 1.f : Gp);
    Bp = Bp < 0.f ? 0.f : (Bp > 1.f ? 1.f : Bp);
    // Decode to linear
    const float Rlin = bt709_inverse_oetf(Rp);
    const float Glin = bt709_inverse_oetf(Gp);
    const float Blin = bt709_inverse_oetf(Bp);
    // Convert to 8-bit full-range
    r = clamp8(Rlin * 255.0f);
    g = clamp8(Glin * 255.0f);
    b = clamp8(Blin * 255.0f);
}

// Kernel: read planar YUV444p (linear, pitched) and write RGBA8 (A=255) to a cudaArray via surface
extern "C" __global__ void kYUV444p_to_RGBA8_709lim_surface(
    const uint8_t* __restrict__ Y, size_t pitchY,
    const uint8_t* __restrict__ U, size_t pitchU,
    const uint8_t* __restrict__ V, size_t pitchV,
    cudaSurfaceObject_t surfOut,
    int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const uint8_t* yRow = Y + (size_t)y * pitchY;
    const uint8_t* uRow = U + (size_t)y * pitchU;
    const uint8_t* vRow = V + (size_t)y * pitchV;
    uint8_t r, g, b;
    yuv709lim_to_linear_rgb8(yRow[x], uRow[x], vRow[x], r, g, b);
    const uchar4 px = make_uchar4(r, g, b, 255); // A=255
    // surf2Dwrite expects X offset in bytes
    surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

// Kernel: read planar YUV420p (linear, pitched) and write RGBA8 (A=255) to a cudaArray via surface
extern "C" __global__ void kYUV420p_to_RGBA8_709lim_surface(
    const uint8_t* __restrict__ Y, size_t pitchY,
    const uint8_t* __restrict__ U, size_t pitchU,
    const uint8_t* __restrict__ V, size_t pitchV,
    cudaSurfaceObject_t surfOut,
    int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const uint8_t* yRow = Y + (size_t)y * pitchY;
    // 4:2:0 subsampling: U/V at half resolution in both dimensions
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const uint8_t* uRow = U + (size_t)uvy * pitchU;
    const uint8_t* vRow = V + (size_t)uvy * pitchV;

    uint8_t r, g, b;
    yuv709lim_to_linear_rgb8(yRow[x], uRow[uvx], vRow[uvx], r, g, b);
    const uchar4 px = make_uchar4(r, g, b, 255); // A=255
    surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

// Kernel: read NV12 (Y plane + interleaved UV plane) and write RGBA8 (A=255) to a cudaArray via surface
// Signature kept consistent with planar variants; V/pitchV are unused
extern "C" __global__ void kNV12_to_RGBA8_709lim_surface(
    const uint8_t* __restrict__ Y, size_t pitchY,
    const uint8_t* __restrict__ UV, size_t pitchUV,
    const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
    cudaSurfaceObject_t surfOut,
    int W, int H)
{
    (void)V_unused; (void)pitchV_unused;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const uint8_t* yRow = Y + (size_t)y * pitchY;
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const uint8_t* uvRow = UV + (size_t)uvy * pitchUV;
    const uint8_t U = uvRow[(uvx << 1) + 0]; // NV12: UV order
    const uint8_t V = uvRow[(uvx << 1) + 1];

    uint8_t r, g, b;
    yuv709lim_to_linear_rgb8(yRow[x], U, V, r, g, b);
    const uchar4 px = make_uchar4(r, g, b, 255);
    surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

// Optional runtime launcher for YUV444p (unused when launched via Driver API/PTX)
extern "C" cudaError_t yuv444p_to_rgba8_709lim_surface(
    const uint8_t* dY, const uint8_t* dU, const uint8_t* dV,
    int pitchY, int pitchU, int pitchV,
    cudaSurfaceObject_t surfOut,
    int W, int H,
    cudaStream_t stream /* = 0 */)
{
    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    kYUV444p_to_RGBA8_709lim_surface<<<grid, block, 0, stream>>>(
        dY, (size_t)pitchY,
        dU, (size_t)pitchU,
        dV, (size_t)pitchV,
        surfOut, W, H);
    return cudaGetLastError();
}

