#include <stdint.h>
#include <stddef.h>

// 0..3 and 1020..1023 are reserved for SDI sync words, so v210 excludes them.
// The libavcodec v210 encoder clamps the same way, which keeps the two packers
// byte-for-byte comparable.
static __device__ __forceinline__ uint32_t clip10(uint32_t sample)
{
    return min(max(sample, 4u), 1019u);
}

// One input sample as ten bits. 8-bit samples (nv12) gain two zero low bits, the SDI padding that
// keeps limited range exact (16..235 becomes 64..940); 16-bit words drop `shift`, the six low bits
// p010le and p210le leave empty.
static __device__ __forceinline__ uint32_t sample10(const uint8_t* row, int index, int depth, int shift)
{
    return depth == 8 ? row[index] << 2 : reinterpret_cast<const uint16_t*>(row)[index] >> shift;
}

// One thread owns a v210 block: six 4:2:2 pixels, twelve 10-bit samples, four
// little-endian words of three samples each. A block never shares a word with
// its neighbours, so the packing needs no cooperation between threads.
//
// A partial block at the right edge pads its missing samples with zero and
// leaves the words past the end of the row unwritten, which is what libavcodec
// does; the caller has already zeroed the row padding.
//
// `depth` and `shift` are those of the input's pixel format descriptor. With `chroma_shift` 1 the
// input is 4:2:0 (nv12, p010le) and each output row blends its two nearest chroma rows 3:1, which
// is linear interpolation for chroma sited midway between two luma rows (the MPEG-2, H.264 and
// HEVC default); the picture is taken as progressive. With 0 both rows are the same one and the
// blend returns its sample unchanged.
extern "C" __global__ void pack_v210(
    uint8_t* output, int output_pitch, int width, int height,
    const uint8_t* input_y, int y_pitch, const uint8_t* input_u, int u_pitch,
    const uint8_t* input_v, int v_pitch, int semiplanar, int depth, int shift, int chroma_shift)
{
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (block >= (width + 5) / 6 || row >= height) return;

    const uint8_t* y = input_y + static_cast<size_t>(row) * y_pitch;
    // The chroma row of this output row, and its other neighbour: above an even row, below an odd one.
    const size_t near = row >> chroma_shift;
    const size_t far = chroma_shift ? min(max(row / 2 + row % 2 * 2 - 1, 0), (height - 1) / 2) : near;
    const uint8_t* v_plane = semiplanar ? input_u : input_v;
    const int v_step = semiplanar ? u_pitch : v_pitch;
    const auto chroma = [=](const uint8_t* plane, int pitch, int index) {
        return clip10((3 * sample10(plane + near * pitch, index, depth, shift) +
                       sample10(plane + far * pitch, index, depth, shift) + 2) >> 2);
    };

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
        const int c = semiplanar ? x : x / 2;
        s[0] = chroma(input_u, u_pitch, c);
        s[1] = clip10(sample10(y, x, depth, shift));
        s[2] = chroma(v_plane, v_step, c + semiplanar);
        s[3] = clip10(sample10(y, x + 1, depth, shift));
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

// ---- Unpack: v210 rows to the canvas storage (p210le, p010le or nv12) in one pass ----
//
// The same kernels as FFmpeg's hwupload_cuda=v210_width (deps/ffmpeg/9/0017, libavfilter/
// vf_hwupload_cuda.cu), which the demuxer path of an MXL source and tests/cuda still use: a change
// to one belongs in the other. Here CudaPinnedUploader launches them for raw_video_file_cuda.
// `input_pitch` is the v210 row stride, a multiple of 32 bytes (the standard stride is one of 128).

#define INLINE __device__ static inline __attribute__((always_inline))

// 12 pixels of a v210 row: two blocks of four words, each word three 10-bit codes, in the order
// Cb Y Cr Y... Of the 24 codes the even ones are chroma (Cb, Cr alternating), the odd ones luma.
struct __align__(16) Block { uint32_t word[4]; };
struct Row { Block block[2]; };
#define CODE(row, n) ((row).block[(n) / 12].word[(n) % 12 / 3] >> (n) % 3 * 10 & 1023)

// The 12 pixels from x of a picture row: two aligned 128-bit loads.
INLINE Row load(const uint8_t *input, int pitch, int row, int x)
{
    const Block *block = reinterpret_cast<const Block *>(input + static_cast<size_t>(row) * pitch) + x / 6;
    const Row loaded = { { block[0], block[1] } };
    return loaded;
}

// The rounded mean of two 10-bit codes: 8 bits, or 10 in the high bits of 16.
template <typename T> INLINE T mean(uint32_t a, uint32_t b)
{
    const int drop = sizeof(T) == 1 ? 3 : 1;
    const uint32_t v = (a + b + (1 << drop >> 1)) >> drop;
    return sizeof(T) == 1 ? (v > 255 ? 255 : v) : v << 6;
}

// 12 samples (T, four to a word W) of a plane row from sample x, each the mean of the luma or
// the chroma code of picture rows a and b: three aligned word stores, or sample by sample
// where the picture ends. Every index is a constant, so the codes stay in registers.
template <typename T, typename W, int luma>
INLINE void store(uint8_t *plane, int pitch, int row, int x, int width, const Row &a, const Row &b)
{
    const int bits = sizeof(T) * 8;
#define SAMPLE(i) static_cast<W>(mean<T>(CODE(a, 2 * (i) + luma), CODE(b, 2 * (i) + luma)))
#define WORD(i) (SAMPLE(i) | SAMPLE(i + 1) << bits | SAMPLE(i + 2) << 2 * bits | SAMPLE(i + 3) << 3 * bits)
    const W w0 = WORD(0), w1 = WORD(4), w2 = WORD(8);
    T *dst = reinterpret_cast<T *>(plane + static_cast<size_t>(row) * pitch) + x;
    if (x + 12 <= width) {
        reinterpret_cast<W *>(dst)[0] = w0;
        reinterpret_cast<W *>(dst)[1] = w1;
        reinterpret_cast<W *>(dst)[2] = w2;
    } else {
        for (int i = 0; i < 12; i++)
            if (x + i < width)
                dst[i] = (i < 4 ? w0 : i < 8 ? w1 : w2) >> i % 4 * bits;
    }
}

// One thread owns 12 pixels of a chroma row and of the `rows` picture rows it is made of (2 for
// 4:2:0), so no packed word is read twice and no P210 intermediate exists. In an interlaced
// frame 4:2:0 chroma rows alternate fields, and the two picture rows are of the chroma row's field.
template <typename T, typename W, int rows>
INLINE void unpack(const uint8_t *input, int input_pitch, int width, int height, int interlaced,
                   uint8_t *output_y, int y_pitch, uint8_t *output_uv, int uv_pitch)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
    const int chroma = blockIdx.y * blockDim.y + threadIdx.y;
    const int fields = rows == 2 && interlaced;
    const int first = chroma * rows - (fields & chroma), next = first + 1 + fields;
    if (x >= width || first >= height) return;
    const int second = rows == 2 && next < height ? next : first;
    const Row a = load(input, input_pitch, first, x);
    const Row b = rows == 2 ? load(input, input_pitch, second, x) : a;
    store<T, W, 1>(output_y, y_pitch, first, x, width, a, a);
    if (second != first)
        store<T, W, 1>(output_y, y_pitch, second, x, width, b, b);
    if (chroma * rows < height)   // a last bottom-field row can lie below the last chroma row
        store<T, W, 0>(output_uv, uv_pitch, chroma, x, width, a, b);
}

#define UNPACK(format, T, W, rows) \
extern "C" __global__ void unpack_v210_##format( \
    const uint8_t *input, int input_pitch, int width, int height, int interlaced, \
    uint8_t *output_y, int y_pitch, uint8_t *output_uv, int uv_pitch) \
{ \
    unpack<T, W, rows>(input, input_pitch, width, height, interlaced, \
                       output_y, y_pitch, output_uv, uv_pitch); \
}

UNPACK(p210le, uint16_t, uint64_t, 1)
UNPACK(p010le, uint16_t, uint64_t, 2)
UNPACK(nv12, uint8_t, uint32_t, 2)

// ---- Unpack: UYVY rows (8-bit 4:2:2, the bytes Cb Y Cr Y; what an NDI receiver delivers) to nv12 ----
//
// Not in FFmpeg's hwupload_cuda: the kernels the two files share end above this line.
// CudaPinnedUploader launches it for uyvy422_to_cuda. `input_pitch` is a multiple of 16 bytes
// (uyvy422Stride), so a thread's 8 pixels are two aligned words of each of its two picture rows.

// The even bytes of a word, as four bytes.
INLINE uint32_t evenBytes(uint64_t word)
{
    word &= 0x00ff00ff00ff00ffull;
    word = (word | word >> 8) & 0x0000ffff0000ffffull;
    return static_cast<uint32_t>(word | word >> 16);
}

// 8 samples of a plane row from sample x: one word store, or sample by sample where the picture ends.
INLINE void store8(uint8_t *plane, int pitch, int row, int x, int width, uint64_t samples)
{
    uint8_t *dst = plane + static_cast<size_t>(row) * pitch + x;
    if (x + 8 <= width) {
        *reinterpret_cast<uint64_t *>(dst) = samples;
    } else {
        for (int i = 0; x + i < width; i++)
            dst[i] = samples >> i * 8;
    }
}

extern "C" __global__ void unpack_uyvy422_nv12(
    const uint8_t *input, int input_pitch, int width, int height, int interlaced,
    uint8_t *output_y, int y_pitch, uint8_t *output_uv, int uv_pitch)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int chroma = blockIdx.y * blockDim.y + threadIdx.y;
    // The two picture rows of the chroma row, of its field in an interlaced frame: as in unpack()
    // above, whose text the FFmpeg patch pins, so the rule is written out here again.
    const int first = chroma * 2 - (interlaced ? chroma & 1 : 0), next = first + 1 + (interlaced ? 1 : 0);
    if (x >= width || first >= height) return;
    const int second = next < height ? next : first;
    const uint64_t *a = reinterpret_cast<const uint64_t *>(input + static_cast<size_t>(first) * input_pitch) + x / 4;
    const uint64_t *b = reinterpret_cast<const uint64_t *>(input + static_cast<size_t>(second) * input_pitch) + x / 4;
    const uint64_t a0 = a[0], a1 = a[1], b0 = b[0], b1 = b[1];
    // Luma is the odd bytes. Chroma, the even ones, is already in nv12's order (Cb Cr): the
    // rounded mean of the two rows, each sum in the 16 bits of its byte pair.
    const auto luma = [](uint64_t w0, uint64_t w1) { return evenBytes(w0 >> 8) | static_cast<uint64_t>(evenBytes(w1 >> 8)) << 32; };
    const auto mean = [](uint64_t p, uint64_t q) {
        const uint64_t even = 0x00ff00ff00ff00ffull;
        return evenBytes(((p & even) + (q & even) + 0x0001000100010001ull) >> 1);
    };
    store8(output_y, y_pitch, first, x, width, luma(a0, a1));
    if (second != first)
        store8(output_y, y_pitch, second, x, width, luma(b0, b1));
    if (chroma * 2 < height)   // a last bottom-field row can lie below the last chroma row
        store8(output_uv, uv_pitch, chroma, x, width, mean(a0, b0) | static_cast<uint64_t>(mean(a1, b1)) << 32);
}
