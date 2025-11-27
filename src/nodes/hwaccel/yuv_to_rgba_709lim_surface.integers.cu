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
// Integer fixed-point implementation (input/output 0..65535). Uses piecewise-linear approximation.
__device__ __forceinline__ uint16_t bt709_inverse_oetf_u16(uint16_t x16)
{
    // Breakpoints in 16-bit domain (0..65535) for R' (nonlinear)
    const int X0 = 5318;   // ~0.081 * 65535
    const int X1 = 13107;  // 0.2  * 65535
    const int X2 = 26214;  // 0.4  * 65535
    const int X3 = 39321;  // 0.6  * 65535
    const int X4 = 52428;  // 0.8  * 65535
    // Corresponding linear values f(x) scaled to 0..65535
    const int Y0 = 1180;    // f(0.081) ~ 0.018
    const int Y1 = 3604;    // f(0.2)  ~ 0.055
    const int Y2 = 11400;   // f(0.4)  ~ 0.173
    const int Y3 = 23815;   // f(0.6)  ~ 0.364
    const int Y4 = 42034;   // f(0.8)  ~ 0.642
    const int Y5 = 65535;   // f(1.0)  = 1.0
    // Slopes (Q16.16) for segments: [X0,X1], [X1,X2], [X2,X3], [X3,X4], [X4,65535]
    const int S0_q16 = 14564;  // (1/4.5) in Q16 for [0, X0] (through origin)
    const int S1_q16 = 20395;  // (Y1-Y0)/(X1-X0)
    const int S2_q16 = 38980;  // (Y2-Y1)/(X2-X1)
    const int S3_q16 = 62076;  // (Y3-Y2)/(X3-X2)
    const int S4_q16 = 91096;  // (Y4-Y3)/(X4-X3)
    const int S5_q16 = 117507; // (Y5-Y4)/(65535-X4)

    int x = (int)x16;
    if (x <= 0) return 0;
    if (x < X0) {
        // y = slope * x
        return (uint16_t)((((int64_t)x * S0_q16) + (1 << 15)) >> 16);
    } else if (x < X1) {
        int dx = x - X0;
        int y = Y0 + (int)((((int64_t)dx * S1_q16) + (1 << 15)) >> 16);
        return (uint16_t)y;
    } else if (x < X2) {
        int dx = x - X1;
        int y = Y1 + (int)((((int64_t)dx * S2_q16) + (1 << 15)) >> 16);
        return (uint16_t)y;
    } else if (x < X3) {
        int dx = x - X2;
        int y = Y2 + (int)((((int64_t)dx * S3_q16) + (1 << 15)) >> 16);
        return (uint16_t)y;
    } else if (x < X4) {
        int dx = x - X3;
        int y = Y3 + (int)((((int64_t)dx * S4_q16) + (1 << 15)) >> 16);
        return (uint16_t)y;
    } else {
        int dx = x - X4;
        int y = Y4 + (int)((((int64_t)dx * S5_q16) + (1 << 15)) >> 16);
        return (uint16_t)(y > 65535 ? 65535 : y);
    }
}

// Float wrapper to preserve existing call sites; routes through fixed-point path
__device__ __forceinline__ float bt709_inverse_oetf(float x)
{
    x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
    const uint16_t x16 = (uint16_t)(x * 65535.0f + 0.5f);
    const uint16_t y16 = bt709_inverse_oetf_u16(x16);
    return (float)y16 * (1.0f / 65535.0f);
}

__device__ __forceinline__ void yuv709lim_to_linear_rgb8(
    uint8_t Y, uint8_t U, uint8_t V,
    uint8_t& r, uint8_t& g, uint8_t& b)
{
    // Compute nonlinear R'G'B' (0..65535) from limited-range YUV using 16-bit fixed-point
    // Scale factors chosen so that 1.0 -> 65535 precisely (65535 = 255 * 257)
    const int y_off = (int)Y - 16;
    const int u_off = (int)U - 128;
    const int v_off = (int)V - 128;
    const int y_clamped = y_off < 0 ? 0 : y_off; // clamp Y-16 to >= 0
    // Coefficients scaled by 257 to map division by 255 into 16-bit full-scale
    // 1.164383*257 ≈ 299, 1.792741*257 ≈ 462, -0.213249*257 ≈ -54, -0.532909*257 ≈ -137, 2.112402*257 ≈ 543
    const int c = 299 * y_clamped;
    int r_tmp = c + 462 * v_off;
    int g_tmp = c -  54 * u_off - 137 * v_off;
    int b_tmp = c + 543 * u_off;
    // Saturate to 0..65535 range
    r_tmp = r_tmp < 0 ? 0 : (r_tmp > 65535 ? 65535 : r_tmp);
    g_tmp = g_tmp < 0 ? 0 : (g_tmp > 65535 ? 65535 : g_tmp);
    b_tmp = b_tmp < 0 ? 0 : (b_tmp > 65535 ? 65535 : b_tmp);
    const uint16_t r16 = (uint16_t)r_tmp;
    const uint16_t g16 = (uint16_t)g_tmp;
    const uint16_t b16 = (uint16_t)b_tmp;
    // Decode to linear using fixed-point inverse OETF
    const uint16_t Rlin16 = bt709_inverse_oetf_u16(r16);
    const uint16_t Glin16 = bt709_inverse_oetf_u16(g16);
    const uint16_t Blin16 = bt709_inverse_oetf_u16(b16);
    // Convert to 8-bit full-range with rounding: round(y16 * 255 / 65535)
    r = (uint8_t)((((uint32_t)Rlin16 * 255u) + 32767u) / 65535u);
    g = (uint8_t)((((uint32_t)Glin16 * 255u) + 32767u) / 65535u);
    b = (uint8_t)((((uint32_t)Blin16 * 255u) + 32767u) / 65535u);
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

