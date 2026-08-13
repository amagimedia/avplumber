#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t toByte(float value)
{
	value = fminf(fmaxf(value, 0.0f), 1.0f);
	return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

__device__ __forceinline__ float channel(const float4 &pixel, int lane)
{
	switch (lane) {
	case 0:
		return pixel.x;
	case 1:
		return pixel.y;
	case 2:
		return pixel.z;
	default:
		return pixel.w;
	}
}

extern "C" __global__ void clear_rgb0(
	uint8_t *destination, size_t destination_pitch, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	auto *row = reinterpret_cast<uchar4 *>(destination + static_cast<size_t>(y) * destination_pitch);
	row[x] = make_uchar4(0, 0, 0, 255);
}

extern "C" __global__ void composite_rgba_texture(
	cudaTextureObject_t source,
	uint8_t *destination,
	size_t destination_pitch,
	int destination_x,
	int destination_y,
	int destination_width,
	int destination_height,
	int red_lane,
	int green_lane,
	int blue_lane)
{
	const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int local_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (local_x >= destination_width || local_y >= destination_height)
		return;

	const float u = (static_cast<float>(local_x) + 0.5f) /
	                static_cast<float>(destination_width);
	const float v = (static_cast<float>(local_y) + 0.5f) /
	                static_cast<float>(destination_height);
	const float4 pixel = tex2D<float4>(source, u, v);

	auto *row = reinterpret_cast<uchar4 *>(
		destination + static_cast<size_t>(destination_y + local_y) * destination_pitch);
	row[destination_x + local_x] = make_uchar4(
		toByte(channel(pixel, red_lane)),
		toByte(channel(pixel, green_lane)),
		toByte(channel(pixel, blue_lane)),
		255);
}
