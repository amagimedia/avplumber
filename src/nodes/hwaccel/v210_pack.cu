#include <stdint.h>
#include <stddef.h>

// 0..3 and 1020..1023 are reserved for SDI sync words, so v210 excludes them.
// The libavcodec v210 encoder clamps the same way, which keeps the two packers
// byte-for-byte comparable.
static __device__ __forceinline__ uint32_t clip10(uint32_t sample)
{
    return min(max(sample, 4u), 1019u);
}

// One thread owns a v210 block: six 4:2:2 pixels, twelve 10-bit samples, four
// little-endian words of three samples each. A block never shares a word with
// its neighbours, so the packing needs no cooperation between threads.
//
// A partial block at the right edge pads its missing samples with zero and
// leaves the words past the end of the row unwritten, which is what libavcodec
// does; the caller has already zeroed the row padding.
extern "C" __global__ void pack_v210(
    uint8_t* output, int output_pitch, int width, int height,
    const uint8_t* input_y, int y_pitch, const uint8_t* input_u, int u_pitch,
    const uint8_t* input_v, int v_pitch, int semiplanar)
{
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (block >= (width + 5) / 6 || row >= height) return;

    const uint16_t* y = reinterpret_cast<const uint16_t*>(
        input_y + static_cast<size_t>(row) * y_pitch);
    const uint16_t* u = reinterpret_cast<const uint16_t*>(
        input_u + static_cast<size_t>(row) * u_pitch);
    const uint16_t* v = reinterpret_cast<const uint16_t*>(
        input_v + static_cast<size_t>(row) * v_pitch);
    // p210le keeps its samples in the high bits of each 16-bit word.
    const unsigned shift = semiplanar ? 6 : 0;

    uint32_t sample[12];
    for (int pair = 0; pair < 3; pair++) {
        // Sample order within a block is U Y V Y per pixel pair, and the width
        // is even, so one guard covers both luma samples.
        const int x = block * 6 + pair * 2;
        uint32_t* s = sample + pair * 4;
        if (x >= width) {
            s[0] = s[1] = s[2] = s[3] = 0;
            continue;
        }
        const int chroma = semiplanar ? x : x / 2;
        s[0] = clip10(u[chroma] >> shift);
        s[1] = clip10(y[x] >> shift);
        s[2] = clip10((semiplanar ? u[chroma + 1] : v[chroma]) >> shift);
        s[3] = clip10(y[x + 1] >> shift);
    }

    uint32_t* dst = reinterpret_cast<uint32_t*>(
        output + static_cast<size_t>(row) * output_pitch) + block * 4;
    const int pixels = min(6, width - block * 6);
    const int words = (pixels * 2 + 2) / 3; // three samples per word, last one partial
    for (int word = 0; word < words; word++) {
        dst[word] = sample[word * 3] | (sample[word * 3 + 1] << 10) |
                    (sample[word * 3 + 2] << 20);
    }
}
