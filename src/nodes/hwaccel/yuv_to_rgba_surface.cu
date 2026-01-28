// BT.709 LIMITED and FULL input -> Nonlinear (display-referred) full-range RGBA8 output via surface writes.
// This is intended for OBS' standard linear-sRGB pipeline, where OBS will decode to linear on sampling.
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

__device__ __forceinline__ void yuv709lim_to_nonlinear_rgb8(
	uint8_t Y, uint8_t U, uint8_t V,
	uint8_t& r, uint8_t& g, uint8_t& b)
{
	// Normalize to [0,1]
	float y = (float)Y / 255.0f;
	float u = (float)U / 255.0f;
	float v = (float)V / 255.0f;

	// Match OBS' limited-range clamp:
	//   color_range_min = {16/255, 16/255, 16/255}
	//   color_range_max = {235/255, 240/255, 240/255}
	const float ymin = 16.0f / 255.0f;
	const float umax = 240.0f / 255.0f;
	const float vmin = 16.0f / 255.0f;
	const float ymax = 235.0f / 255.0f;
	const float umin = 16.0f / 255.0f;
	const float vmax = 240.0f / 255.0f;

	// Clamp YUV into the legal range
	y = y < ymin ? ymin : (y > ymax ? ymax : y);
	u = u < umin ? umin : (u > umax ? umax : u);
	v = v < vmin ? vmin : (v > vmax ? vmax : v);

	// Apply the same BT.709 limited-range matrix as OBS
	// (see format_info[VIDEO_CS_709].matrix[0] in video-matrices.c)
	float Rp = 1.164384f * y + 0.000000f * u + 1.792741f * v - 0.972945f;
	float Gp = 1.164384f * y - 0.213249f * u - 0.532909f * v + 0.301483f;
	float Bp = 1.164384f * y + 2.112402f * u + 0.000000f * v - 1.133402f;

	// Clamp to [0,1] and store as 8‑bit nonlinear (display-referred) RGB
	Rp = Rp < 0.f ? 0.f : (Rp > 1.f ? 1.f : Rp);
	Gp = Gp < 0.f ? 0.f : (Gp > 1.f ? 1.f : Gp);
	Bp = Bp < 0.f ? 0.f : (Bp > 1.f ? 1.f : Bp);

	// Store nonlinear values; OBS will handle sRGB decoding during rendering when linear_srgb is enabled.
	r = clamp8(Rp * 255.0f);
	g = clamp8(Gp * 255.0f);
	b = clamp8(Bp * 255.0f);
}

__device__ __forceinline__ void yuv709full_to_nonlinear_rgb8(
	uint8_t Y, uint8_t U, uint8_t V,
	uint8_t& r, uint8_t& g, uint8_t& b)
{
	// Normalize to [0,1]
	const float y = ((float)Y) / 255.0f;
	const float u = ((float)U) / 255.0f;
	const float v = ((float)V) / 255.0f;

	// Full range in OBS is just [0,1] for all components
	const float ymin = 0.0f, ymax = 1.0f;
	const float umin = 0.0f, umax = 1.0f;
	const float vmin = 0.0f, vmax = 1.0f;

	float yc = y < ymin ? ymin : (y > ymax ? ymax : y);
	float uc = u < umin ? umin : (u > umax ? umax : u);
	float vc = v < vmin ? vmin : (v > vmax ? vmax : v);

	// Apply the same BT.709 full-range matrix as OBS
	// (see format_info[VIDEO_CS_709].matrix[1] in video-matrices.c)
	float Rp = 1.000000f * yc + 0.000000f * uc + 1.581000f * vc - 0.793600f;
	float Gp = 1.000000f * yc - 0.188062f * uc - 0.469967f * vc + 0.330305f;
	float Bp = 1.000000f * yc + 1.862906f * uc + 0.000000f * vc - 0.935106f;

	// Clamp to [0,1] and store as 8‑bit nonlinear (display-referred) RGB
	Rp = Rp < 0.f ? 0.f : (Rp > 1.f ? 1.f : Rp);
	Gp = Gp < 0.f ? 0.f : (Gp > 1.f ? 1.f : Gp);
	Bp = Bp < 0.f ? 0.f : (Bp > 1.f ? 1.f : Bp);

	r = clamp8(Rp * 255.0f);
	g = clamp8(Gp * 255.0f);
	b = clamp8(Bp * 255.0f);
}

__device__ __forceinline__ void surf_write_rgba8(cudaSurfaceObject_t surfOut, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	const uchar4 px = make_uchar4(r, g, b, 255);
	surf2Dwrite(px, surfOut, x * (int)sizeof(uchar4), y);
}

__device__ __forceinline__ int2 thread_coords()
{
	return make_int2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ __forceinline__ const uint8_t* plane_row(const uint8_t* plane, size_t pitch, int row)
{
	return plane + (size_t)row * pitch;
}

__device__ __forceinline__ const uint8_t* pixel_ptr(const uint8_t* base, size_t pitch, int x, int y)
{
	return plane_row(base, pitch, y) + ((size_t)x << 2);
}

__device__ __forceinline__ uchar4 swizzle_RGBA(const uint8_t* p) { return make_uchar4(p[0], p[1], p[2], p[3]); }
__device__ __forceinline__ uchar4 swizzle_BGRA(const uint8_t* p) { return make_uchar4(p[2], p[1], p[0], p[3]); }
__device__ __forceinline__ uchar4 swizzle_ARGB(const uint8_t* p) { return make_uchar4(p[1], p[2], p[3], p[0]); }
__device__ __forceinline__ uchar4 swizzle_ABGR(const uint8_t* p) { return make_uchar4(p[3], p[2], p[1], p[0]); }
__device__ __forceinline__ uchar4 swizzle_RGB0(const uint8_t* p) { return make_uchar4(p[0], p[1], p[2], 255); }
__device__ __forceinline__ uchar4 swizzle_BGR0(const uint8_t* p) { return make_uchar4(p[2], p[1], p[0], 255); }
__device__ __forceinline__ uchar4 swizzle_0RGB(const uint8_t* p) { return make_uchar4(p[1], p[2], p[3], 255); }
__device__ __forceinline__ uchar4 swizzle_0BGR(const uint8_t* p) { return make_uchar4(p[3], p[2], p[1], 255); }

template <uchar4 (*SwizzleFn)(const uint8_t*)>
__device__ __forceinline__ void run_passthrough_kernel(
	const uint8_t* __restrict__ src, size_t pitch,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* p = pixel_ptr(src, pitch, coord.x, coord.y);
	const uchar4 px = SwizzleFn(p);
	surf2Dwrite(px, surfOut, coord.x * (int)sizeof(uchar4), coord.y);
}

template <void (*ConvertFn)(uint8_t, uint8_t, uint8_t, uint8_t&, uint8_t&, uint8_t&)>
__device__ __forceinline__ void run_yuv444p_kernel(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* yRow = plane_row(Y, pitchY, coord.y);
	const uint8_t* uRow = plane_row(U, pitchU, coord.y);
	const uint8_t* vRow = plane_row(V, pitchV, coord.y);
	uint8_t r, g, b;
	ConvertFn(yRow[coord.x], uRow[coord.x], vRow[coord.x], r, g, b);
	surf_write_rgba8(surfOut, coord.x, coord.y, r, g, b);
}

template <void (*ConvertFn)(uint8_t, uint8_t, uint8_t, uint8_t&, uint8_t&, uint8_t&)>
__device__ __forceinline__ void run_yuv420p_kernel(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* yRow = plane_row(Y, pitchY, coord.y);
	const int uvx = coord.x >> 1;
	const int uvy = coord.y >> 1;
	const uint8_t* uRow = plane_row(U, pitchU, uvy);
	const uint8_t* vRow = plane_row(V, pitchV, uvy);
	uint8_t r, g, b;
	ConvertFn(yRow[coord.x], uRow[uvx], vRow[uvx], r, g, b);
	surf_write_rgba8(surfOut, coord.x, coord.y, r, g, b);
}

template <void (*ConvertFn)(uint8_t, uint8_t, uint8_t, uint8_t&, uint8_t&, uint8_t&)>
__device__ __forceinline__ void run_yuv422p_kernel(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* yRow = plane_row(Y, pitchY, coord.y);
	const uint8_t* uRow = plane_row(U, pitchU, coord.y);
	const uint8_t* vRow = plane_row(V, pitchV, coord.y);
	const int uvx = coord.x >> 1;   // 4:2:2: half horizontal resolution, full vertical
	uint8_t r, g, b;
	ConvertFn(yRow[coord.x], uRow[uvx], vRow[uvx], r, g, b);
	surf_write_rgba8(surfOut, coord.x, coord.y, r, g, b);
}

template <void (*ConvertFn)(uint8_t, uint8_t, uint8_t, uint8_t&, uint8_t&, uint8_t&)>
__device__ __forceinline__ void run_nv12_kernel(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* yRow = plane_row(Y, pitchY, coord.y);
	const int uvx = coord.x >> 1;
	const int uvy = coord.y >> 1;
	const uint8_t* uvRow = plane_row(UV, pitchUV, uvy);
	const uint8_t U = uvRow[(uvx << 1) + 0]; // NV12: UV order
	const uint8_t V = uvRow[(uvx << 1) + 1];
	uint8_t r, g, b;
	ConvertFn(yRow[coord.x], U, V, r, g, b);
	surf_write_rgba8(surfOut, coord.x, coord.y, r, g, b);
}

template <void (*ConvertFn)(uint8_t, uint8_t, uint8_t, uint8_t&, uint8_t&, uint8_t&)>
__device__ __forceinline__ void run_nv16_kernel(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	const int2 coord = thread_coords();
	if (coord.x >= W || coord.y >= H) return;
	const uint8_t* yRow = plane_row(Y, pitchY, coord.y);
	const int uvx = coord.x >> 1;
	const uint8_t* uvRow = plane_row(UV, pitchUV, coord.y);
	const uint8_t U = uvRow[(uvx << 1) + 0]; // NV16: UV order
	const uint8_t V = uvRow[(uvx << 1) + 1];
	uint8_t r, g, b;
	ConvertFn(yRow[coord.x], U, V, r, g, b);
	surf_write_rgba8(surfOut, coord.x, coord.y, r, g, b);
}

// ---- LIMITED kernels ----

extern "C" __global__ void kYUV444p_709lim_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv444p_kernel<yuv709lim_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kYUV420p_709lim_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv420p_kernel<yuv709lim_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kYUV422p_709lim_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv422p_kernel<yuv709lim_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kNV12_709lim_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)V_unused; (void)pitchV_unused;
	run_nv12_kernel<yuv709lim_to_nonlinear_rgb8>(Y, pitchY, UV, pitchUV, surfOut, W, H);
}

extern "C" __global__ void kNV16_709lim_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)V_unused; (void)pitchV_unused;
	run_nv16_kernel<yuv709lim_to_nonlinear_rgb8>(Y, pitchY, UV, pitchUV, surfOut, W, H);
}

// ---- FULL kernels ----

extern "C" __global__ void kYUV444p_709full_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv444p_kernel<yuv709full_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kYUV420p_709full_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv420p_kernel<yuv709full_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kYUV422p_709full_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ U, size_t pitchU,
	const uint8_t* __restrict__ V, size_t pitchV,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	run_yuv422p_kernel<yuv709full_to_nonlinear_rgb8>(Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H);
}

extern "C" __global__ void kNV12_709full_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)V_unused; (void)pitchV_unused;
	run_nv12_kernel<yuv709full_to_nonlinear_rgb8>(Y, pitchY, UV, pitchUV, surfOut, W, H);
}

extern "C" __global__ void kNV16_709full_to_RGBA8_surface(
	const uint8_t* __restrict__ Y, size_t pitchY,
	const uint8_t* __restrict__ UV, size_t pitchUV,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)V_unused; (void)pitchV_unused;
	run_nv16_kernel<yuv709full_to_nonlinear_rgb8>(Y, pitchY, UV, pitchUV, surfOut, W, H);
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
	run_passthrough_kernel<swizzle_RGBA>(RGBA, pitch, surfOut, W, H);
}

extern "C" __global__ void kBGRA_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ BGRA, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_BGRA>(BGRA, pitch, surfOut, W, H);
}

extern "C" __global__ void kARGB_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ ARGB, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_ARGB>(ARGB, pitch, surfOut, W, H);
}

extern "C" __global__ void kABGR_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ ABGR, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_ABGR>(ABGR, pitch, surfOut, W, H);
}

extern "C" __global__ void kRGB0_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ RGB0, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_RGB0>(RGB0, pitch, surfOut, W, H);
}

extern "C" __global__ void kBGR0_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ BGR0, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_BGR0>(BGR0, pitch, surfOut, W, H);
}

extern "C" __global__ void k0RGB_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ _0RGB, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_0RGB>(_0RGB, pitch, surfOut, W, H);
}

extern "C" __global__ void k0BGR_to_RGBA8_passthrough_surface(
	const uint8_t* __restrict__ _0BGR, size_t pitch,
	const uint8_t* __restrict__ U_unused, size_t pitchU_unused,
	const uint8_t* __restrict__ V_unused, size_t pitchV_unused,
	cudaSurfaceObject_t surfOut,
	int W, int H)
{
	(void)U_unused; (void)V_unused; (void)pitchU_unused; (void)pitchV_unused;
	run_passthrough_kernel<swizzle_0BGR>(_0BGR, pitch, surfOut, W, H);
}


