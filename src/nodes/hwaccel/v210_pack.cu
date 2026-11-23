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

// ---- Unpack: UYVY rows and packed RGB to nv12 or bgra ----
//
// Not in FFmpeg's hwupload_cuda: the kernels the two files share end above this line.
// CudaPinnedUploader launches these for packed_to_cuda. UYVY is 8-bit 4:2:2, the bytes Cb Y Cr Y:
// what an NDI receiver delivers. Packed RGB is 8-bit with a fourth byte (bgra, bgr0, rgba, rgb0):
// what it delivers for a sender with alpha. `input_pitch` is that of whole groups of 8 pixels
// (packedStride), which a thread unpacks, so its loads are aligned words.
//
// They share their parameters, of which each reads what it needs. `red_shift` is the bit position
// of red in an RGB pixel's little-endian word, 16 for bgra and 0 for rgba; blue is at the other of
// the two. `kr` and `kb` are the luma weights of red and blue (0.2126 and 0.0722 for BT.709) and
// `full_range` chooses Y'CbCr codes of 0..255 over 16..235 and 16..240. `opaque`: the fourth byte
// of the RGB pixels is not alpha.
#define PACKED_PARAMETERS \
    const uint8_t *input, int input_pitch, int width, int height, int interlaced, \
    uint8_t *output_y, int y_pitch, uint8_t *output_uv, int uv_pitch, \
    int red_shift, float kr, float kb, int full_range, int opaque

// The even bytes of a word, as four bytes.
INLINE uint32_t evenBytes(uint64_t word)
{
    word &= 0x00ff00ff00ff00ffull;
    word = (word | word >> 8) & 0x0000ffff0000ffffull;
    return static_cast<uint32_t>(word | word >> 16);
}

// The two picture rows a 4:2:0 chroma row is made of, of its field in an interlaced frame: as in
// unpack() above, whose text the FFmpeg patch pins, so the rule is written out here again. `second`
// is `first` where the picture ends; false below the picture.
INLINE bool chromaRows(int chroma, int height, int interlaced, int &first, int &second)
{
    const int fields = interlaced ? 1 : 0;
    first = chroma * 2 - (fields & chroma);
    const int next = first + 1 + fields;
    second = next < height ? next : first;
    return first < height;
}

// 8 samples (bytes, or words of a bgra row) of a plane row from sample x: sample by sample where
// the picture ends.
template <typename T> INLINE void store8(uint8_t *plane, int pitch, int row, int x, int width, const T *samples)
{
    T *dst = reinterpret_cast<T *>(plane + static_cast<size_t>(row) * pitch) + x;
    for (int i = 0; i < 8 && x + i < width; i++)
        dst[i] = samples[i];
}
INLINE void store8(uint8_t *plane, int pitch, int row, int x, int width, uint64_t samples)
{
    if (x + 8 <= width)
        *reinterpret_cast<uint64_t *>(plane + static_cast<size_t>(row) * pitch + x) = samples;
    else
        store8(plane, pitch, row, x, width, reinterpret_cast<const uint8_t *>(&samples));
}

INLINE uint32_t code8(float value) { return static_cast<uint32_t>(min(max(value + 0.5f, 0.0f), 255.0f)); }

// UYVY to nv12: luma as it is, a chroma row the rounded mean of its two picture rows.
extern "C" __global__ void unpack_uyvy422_nv12(PACKED_PARAMETERS)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int chroma = blockIdx.y * blockDim.y + threadIdx.y;
    int first, second;
    if (x >= width || !chromaRows(chroma, height, interlaced, first, second)) return;
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

// UYVY to opaque bgra, for a source that keeps alpha when its sender has none: a pixel pair
// shares its chroma sample.
extern "C" __global__ void unpack_uyvy422_bgra(PACKED_PARAMETERS)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || row >= height) return;
    const uint8_t *in = input + static_cast<size_t>(row) * input_pitch + x * 2;
    const float kg = 1.0f - kr - kb;
    const float luma_gain = full_range ? 1.0f : 255.0f / 219.0f, luma_zero = full_range ? 0.0f : 16.0f;
    const float chroma_gain = full_range ? 2.0f : 2.0f * 255.0f / 224.0f;
    uint32_t pixels[8];
    for (int i = 0; i < 8; i++) {
        const uint8_t *pair = in + (i & ~1) * 2;
        const float y = luma_gain * (pair[1 + (i & 1) * 2] - luma_zero);
        const float blue = y + chroma_gain * (1.0f - kb) * (pair[0] - 128.0f);
        const float red = y + chroma_gain * (1.0f - kr) * (pair[2] - 128.0f);
        pixels[i] = code8(blue) | code8((y - kr * red - kb * blue) / kg) << 8 | code8(red) << 16 | 255u << 24;
    }
    store8(output_y, y_pitch, row, x, width, pixels);
}

// RGB to nv12. A chroma sample is that of the mean of its 2x2 pixels (two rows of one field in an
// interlaced frame); a picture of odd width repeats its last column. The fourth byte is not read:
// alpha is dropped.
extern "C" __global__ void unpack_rgb32_nv12(PACKED_PARAMETERS)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int chroma = blockIdx.y * blockDim.y + threadIdx.y;
    int first, second;
    if (x >= width || !chromaRows(chroma, height, interlaced, first, second)) return;
    const uint32_t *row[2] = {
        reinterpret_cast<const uint32_t *>(input + static_cast<size_t>(first) * input_pitch),
        reinterpret_cast<const uint32_t *>(input + static_cast<size_t>(second) * input_pitch) };
    const float kg = 1.0f - kr - kb;
    const float luma_gain = full_range ? 1.0f : 219.0f / 255.0f, luma_zero = full_range ? 0.0f : 16.0f;
    const float chroma_gain = full_range ? 0.5f : 112.0f / 255.0f;
    uint8_t luma[2][8], pairs[8];
    for (int pair = 0; pair < 4; pair++) {
        float red = 0, green = 0, blue = 0;
        for (int i = 0; i < 4; i++) {   // the pair's two pixels in each of the two rows
            const int column = pair * 2 + (i & 1);
            const uint32_t pixel = row[i >> 1][min(x + column, width - 1)];
            const float r = pixel >> red_shift & 255, g = pixel >> 8 & 255, b = pixel >> (16 - red_shift) & 255;
            luma[i >> 1][column] = code8(luma_zero + luma_gain * (kr * r + kg * g + kb * b));
            red += r, green += g, blue += b;
        }
        const float y = (kr * red + kg * green + kb * blue) / 4;
        pairs[pair * 2] = code8(128.0f + chroma_gain * (blue / 4 - y) / (1.0f - kb));
        pairs[pair * 2 + 1] = code8(128.0f + chroma_gain * (red / 4 - y) / (1.0f - kr));
    }
    store8(output_y, y_pitch, first, x, width, luma[0]);
    if (second != first)
        store8(output_y, y_pitch, second, x, width, luma[1]);
    if (chroma * 2 < height)   // a last bottom-field row can lie below the last chroma row
        store8(output_uv, uv_pitch, chroma, x, (width + 1) & ~1, pairs);
}

// RGB to bgra, for a source that keeps alpha: the bytes in bgra's order, alpha 255 when the
// pixels have none (`opaque`). bgra itself needs no kernel.
extern "C" __global__ void unpack_rgb32_bgra(PACKED_PARAMETERS)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || row >= height) return;
    const uint32_t *in = reinterpret_cast<const uint32_t *>(input + static_cast<size_t>(row) * input_pitch) + x;
    uint32_t pixels[8];
    for (int i = 0; i < 8; i++) {
        const uint32_t pixel = in[i];
        pixels[i] = (pixel >> (16 - red_shift) & 255) | (pixel & 0xff00) | (pixel >> red_shift & 255) << 16 |
                    (opaque ? 255u << 24 : pixel & 0xff000000);
    }
    store8(output_y, y_pitch, row, x, width, pixels);
}
