// BT.709 LIMITED and FULL input -> Linear full-range RGBA8 output via surface writes.
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
#include <stdint.h>
#include <math.h>

// Clamp helper
__device__ __forceinline__ uint8_t clamp8(float x) {
	x = x < 0.f ? 0.f : (x > 255.f ? 255.f : x);
	return (uint8_t)(x + 0.5f);
}

// BT.709 inverse OETF: nonlinear to linear
__device__ __forceinline__ float bt709_inverse_oetf(float x)
{
	// x in [0,1] (nonlinear), return linear value in [0,1]
	return (x < 0.081f) ? (x / 4.5f) : powf((x + 0.099f) / 1.099f, 1.0f / 0.45f);
}

// YUV709 LIMITED (TV range) -> linear RGB8
__device__ __forceinline__ void yuv709lim_to_linear_rgb8(
	uint8_t Y, uint8_t U, uint8_t V,
	uint8_t& r, uint8_t& g, uint8_t& b)
{
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

// YUV709 FULL (PC range) -> linear RGB8
__device__ __forceinline__ void yuv709full_to_linear_rgb8(
	uint8_t Y, uint8_t U, uint8_t V,
	uint8_t& r, uint8_t& g, uint8_t& b)
{
	// Normalize to [0,1] and [-0.5,0.5]
	const float Yp = ((float)Y) / 255.0f;
	const float Cb = (((float)U) - 128.0f) / 255.0f;
	const float Cr = (((float)V) - 128.0f) / 255.0f;
	// BT.709 full-range matrix for R'G'B' (nonlinear)
	float Rp = Yp + 1.5748f * Cr;
	float Gp = Yp - 0.1873f * Cb - 0.4681f * Cr;
	float Bp = Yp + 1.8556f * Cb;
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

// ---- LIMITED kernels ----

extern "C" __global__ void kYUV444p_709lim_to_RGBA8_surface(
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
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kYUV420p_709lim_to_RGBA8_surface(
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
	const int uvx = x >> 1;
	const int uvy = y >> 1;
	const uint8_t* uRow = U + (size_t)uvy * pitchU;
	const uint8_t* vRow = V + (size_t)uvy * pitchV;
	uint8_t r, g, b;
	yuv709lim_to_linear_rgb8(yRow[x], uRow[uvx], vRow[uvx], r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kYUV422p_709lim_to_RGBA8_surface(
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
	const int uvx = x >> 1;   // 4:2:2: half horizontal resolution, full vertical
	uint8_t r, g, b;
	yuv709lim_to_linear_rgb8(yRow[x], uRow[uvx], vRow[uvx], r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kNV12_709lim_to_RGBA8_surface(
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
	const uint8_t U = uvRow[(uvx << 1) + 0];
	const uint8_t V = uvRow[(uvx << 1) + 1];
	uint8_t r, g, b;
	yuv709lim_to_linear_rgb8(yRow[x], U, V, r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kNV16_709lim_to_RGBA8_surface(
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
	const int uvx = x >> 1;    // half horizontal resolution
	const uint8_t* uvRow = UV + (size_t)y * pitchUV; // full vertical resolution
	const uint8_t U = uvRow[(uvx << 1) + 0]; // NV16: UV order
	const uint8_t V = uvRow[(uvx << 1) + 1];
	uint8_t r, g, b;
	yuv709lim_to_linear_rgb8(yRow[x], U, V, r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

// ---- FULL kernels ----

extern "C" __global__ void kYUV444p_709full_to_RGBA8_surface(
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
	yuv709full_to_linear_rgb8(yRow[x], uRow[x], vRow[x], r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kYUV420p_709full_to_RGBA8_surface(
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
	const int uvx = x >> 1;
	const int uvy = y >> 1;
	const uint8_t* uRow = U + (size_t)uvy * pitchU;
	const uint8_t* vRow = V + (size_t)uvy * pitchV;
	uint8_t r, g, b;
	yuv709full_to_linear_rgb8(yRow[x], uRow[uvx], vRow[uvx], r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kYUV422p_709full_to_RGBA8_surface(
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
	const int uvx = x >> 1;   // 4:2:2: half horizontal resolution, full vertical
	uint8_t r, g, b;
	yuv709full_to_linear_rgb8(yRow[x], uRow[uvx], vRow[uvx], r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kNV12_709full_to_RGBA8_surface(
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
	yuv709full_to_linear_rgb8(yRow[x], U, V, r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kNV16_709full_to_RGBA8_surface(
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
	const int uvx = x >> 1;    // half horizontal resolution
	const uint8_t* uvRow = UV + (size_t)y * pitchUV; // full vertical resolution
	const uint8_t U = uvRow[(uvx << 1) + 0]; // NV16: UV order
	const uint8_t V = uvRow[(uvx << 1) + 1];
	uint8_t r, g, b;
	yuv709full_to_linear_rgb8(yRow[x], U, V, r, g, b);
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

// ---- PASSTHROUGH kernels (no colorspace transform; only channel reorder/alpha) ----

extern "C" __global__ void kRGBA_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ RGBA, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = RGBA + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[0], p[1], p[2], p[3]);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kBGRA_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ BGRA, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = BGRA + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[2], p[1], p[0], p[3]);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kARGB_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ ARGB, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = ARGB + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[1], p[2], p[3], p[0]);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kABGR_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ ABGR, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = ABGR + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[3], p[2], p[1], p[0]);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kRGB0_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ RGB0, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = RGB0 + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[0], p[1], p[2], 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void kBGR0_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ BGR0, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = BGR0 + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[2], p[1], p[0], 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void k0RGB_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ _0RGB, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = _0RGB + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[1], p[2], p[3], 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

extern "C" __global__ void k0BGR_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ _0BGR, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	const uint8_t* row = _0BGR + (size_t)y * pitch;
	const uint8_t* p = row + ((size_t)x << 2);
	const uchar4 px = make_uchar4(p[3], p[2], p[1], 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}


