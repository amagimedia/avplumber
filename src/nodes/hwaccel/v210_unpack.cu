#include <stdint.h>
#include <stddef.h>

// One thread owns a 4:2:2 pixel pair. A pair's U/Y/V/Y samples span two
// little-endian v210 words, each containing three samples and two unused bits.
extern "C" __global__ void unpack_v210(
    const uint8_t* input, int input_pitch, int width, int height,
    uint8_t* output_y, int y_pitch, uint8_t* output_u, int u_pitch,
    uint8_t* output_v, int v_pitch, int semiplanar)
{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (pair >= width / 2 || row >= height) return;

    const uint32_t* packed = reinterpret_cast<const uint32_t*>(
        input + static_cast<size_t>(row) * input_pitch);
    const int sample = pair * 4;
    const int word = sample / 3;
    const uint64_t bits = ((static_cast<uint64_t>(packed[word + 1] & 0x3fffffff) << 30)
                          | (packed[word] & 0x3fffffff)) >> ((sample % 3) * 10);
    const unsigned shift = semiplanar ? 6 : 0;
    const uint16_t u = (bits & 1023) << shift;
    const uint16_t y0 = ((bits >> 10) & 1023) << shift;
    const uint16_t v = ((bits >> 20) & 1023) << shift;
    const uint16_t y1 = ((bits >> 30) & 1023) << shift;

    uint16_t* y = reinterpret_cast<uint16_t*>(output_y + static_cast<size_t>(row) * y_pitch);
    uint16_t* uv = reinterpret_cast<uint16_t*>(output_u + static_cast<size_t>(row) * u_pitch);
    y[pair * 2] = y0;
    y[pair * 2 + 1] = y1;
    if (semiplanar) {
        uv[pair * 2] = u;
        uv[pair * 2 + 1] = v;
    } else {
        uv[pair] = u;
        reinterpret_cast<uint16_t*>(output_v + static_cast<size_t>(row) * v_pitch)[pair] = v;
    }
}
