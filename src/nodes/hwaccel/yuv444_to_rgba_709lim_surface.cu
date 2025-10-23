// BT.709 LIMITED: YUV444p (planar 8-bit) -> RGBA8 (A=255) written to a cudaArray via surface.
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
#include <stdint.h>
// Clamp helper
__device__ __forceinline__ uint8_t clamp8(float x) {
    x = x < 0.f ? 0.f : (x > 255.f ? 255.f : x);
    return (uint8_t)(x + 0.5f);
}
// BT.709 LIMITED conversion
// C  = 1.164383 * (Y - 16)
// R  = C + 1.792741 * (V - 128)
// G  = C - 0.213249 * (U - 128) - 0.532909 * (V - 128)
// B  = C + 2.112402 * (U - 128)
__device__ __forceinline__ void yuv709lim_to_rgb(
    uint8_t Y, uint8_t U, uint8_t V,
    uint8_t& r, uint8_t& g, uint8_t& b)
{
    const float C  = 1.164383f * ((float)Y - 16.f);
    const float Uf = (float)U - 128.f;
    const float Vf = (float)V - 128.f;
    const float Rf = C + 1.792741f * Vf;
    const float Gf = C - 0.213249f * Uf - 0.532909f * Vf;
    const float Bf = C + 2.112402f * Uf;
    r = clamp8(Rf);
    g = clamp8(Gf);
    b = clamp8(Bf);
}
// Kernel: read planar YUV444p (linear, pitched) and write RGBA8 (A=255) to a cudaArray via surface
__global__ void kYUV444p_to_RGBA8_709lim_surface(
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
    yuv709lim_to_rgb(yRow[x], uRow[x], vRow[x], r, g, b);
    const uchar4 px = make_uchar4(r, g, b, 255); // A=255
    // surf2Dwrite expects X offset in bytes
    surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}
// Launcher: bind your GL texture to a cudaArray -> create a cudaSurfaceObject_t -> pass it here
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
