// CUDA color-conversion kernels for the rife_vfi node.
//
// nv12_to_rgb_fp16   : NV12 (limited-range BT.709) -> RGB planar fp16 in [0,1],
//                     padded from H*W to Hp*Wp (edge-replicate).
// rgb_fp16_to_nv12   : RGB planar fp16 in [0,1] -> NV12 (limited-range BT.709),
//                     cropped from Hp*Wp to H*W. UV plane written at 4:2:0.
//
// Convention: RIFE 4.x was trained on nonlinear sRGB scaled to [0,1]. We treat
// the R'G'B' recovered from Y'CbCr as that directly (no gamma decode). Full
// linear-light preprocessing would drift from what the network expects.
//
// All memory is CUDA linear (device pointers with explicit pitches); no
// surfaces or arrays. Kernel launch grid covers the padded output for the
// forward direction and the input crop for the inverse.

#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ void yuv709lim_to_rgb01(
    float Y, float U, float V,
    float& R, float& G, float& B)
{
    // Limited-range Y in [16, 235], Cb/Cr in [16, 240]. Normalize.
    float y  = (Y - 16.0f) * (1.0f / 219.0f);
    float cb = (U - 128.0f) * (1.0f / 224.0f);
    float cr = (V - 128.0f) * (1.0f / 224.0f);
    // BT.709 R'G'B' recovery.
    R = y + 1.5748f * cr;
    G = y - 0.1873f * cb - 0.4681f * cr;
    B = y + 1.8556f * cb;
    R = clampf(R, 0.0f, 1.0f);
    G = clampf(G, 0.0f, 1.0f);
    B = clampf(B, 0.0f, 1.0f);
}

__device__ __forceinline__ void rgb01_to_yuv709lim(
    float R, float G, float B,
    float& Y, float& U, float& V)
{
    R = clampf(R, 0.0f, 1.0f);
    G = clampf(G, 0.0f, 1.0f);
    B = clampf(B, 0.0f, 1.0f);
    // BT.709 Y' Cb Cr in [0,1].
    float y  = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float cb = -0.1146f * R - 0.3854f * G + 0.5f     * B;
    float cr = 0.5f     * R - 0.4542f * G - 0.0458f * B;
    // Limited-range encoding.
    Y = 16.0f  + 219.0f * y;
    U = 128.0f + 224.0f * cb;
    V = 128.0f + 224.0f * cr;
    Y = clampf(Y, 0.0f, 255.0f);
    U = clampf(U, 0.0f, 255.0f);
    V = clampf(V, 0.0f, 255.0f);
}

// One thread per output pixel of the padded fp16 RGB tensor.
// Layout of dst_rgb: 3 planes of size Hp*Wp, tightly packed (stride = Wp * sizeof(__half)).
extern "C" __global__ void nv12_to_rgb_fp16(
    const uint8_t* __restrict__ srcY, int y_pitch,
    const uint8_t* __restrict__ srcUV, int uv_pitch,
    int H, int W,
    __half* __restrict__ dst_rgb, int Hp, int Wp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Wp || y >= Hp) return;

    // Edge-replicate padding for pixels outside the real frame.
    int sx = x < W ? x : (W - 1);
    int sy = y < H ? y : (H - 1);
    int cx = sx >> 1;
    int cy = sy >> 1;

    float Y = (float)srcY[sy * y_pitch + sx];
    // NV12 UV plane: interleaved U,V pairs at half height/width.
    int uv_row = cy * uv_pitch;
    float U = (float)srcUV[uv_row + 2 * cx + 0];
    float V = (float)srcUV[uv_row + 2 * cx + 1];

    float R, G, B;
    yuv709lim_to_rgb01(Y, U, V, R, G, B);

    int plane = Hp * Wp;
    int idx = y * Wp + x;
    dst_rgb[0 * plane + idx] = __float2half(R);
    dst_rgb[1 * plane + idx] = __float2half(G);
    dst_rgb[2 * plane + idx] = __float2half(B);
}

// One thread per 2x2 output block. Each thread writes 4 Y samples and 1 UV pair
// (chroma subsampled 4:2:0 by averaging). Only pixels inside [0, H) x [0, W)
// are written; the padded RGB rows beyond H are ignored.
extern "C" __global__ void rgb_fp16_to_nv12(
    const __half* __restrict__ src_rgb, int Hp, int Wp,
    uint8_t* __restrict__ dstY, int y_pitch,
    uint8_t* __restrict__ dstUV, int uv_pitch,
    int H, int W)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;  // 2x2 block column
    int by = blockIdx.y * blockDim.y + threadIdx.y;  // 2x2 block row
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= W || y0 >= H) return;

    int plane = Hp * Wp;
    auto load_rgb = [&](int xx, int yy, float& R, float& G, float& B) {
        int idx = yy * Wp + xx;
        R = __half2float(src_rgb[0 * plane + idx]);
        G = __half2float(src_rgb[1 * plane + idx]);
        B = __half2float(src_rgb[2 * plane + idx]);
    };

    float rgb[4][3];
    float ysum[4] = {0}, usum = 0, vsum = 0;
    int count = 0;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            int xx = x0 + i;
            int yy = y0 + j;
            if (xx >= W || yy >= H) {
                // Duplicate last valid pixel so partial blocks still contribute
                // sensible chroma; Y for out-of-bounds pixels is simply not
                // written below.
                xx = xx >= W ? (W - 1) : xx;
                yy = yy >= H ? (H - 1) : yy;
            }
            float R, G, B;
            load_rgb(xx, yy, R, G, B);
            rgb[2*j+i][0] = R; rgb[2*j+i][1] = G; rgb[2*j+i][2] = B;
            float Y, U, V;
            rgb01_to_yuv709lim(R, G, B, Y, U, V);
            ysum[2*j+i] = Y;
            usum += U;
            vsum += V;
            ++count;
        }
    }

    // Write Y samples that lie inside the frame.
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            int xx = x0 + i;
            int yy = y0 + j;
            if (xx < W && yy < H) {
                dstY[yy * y_pitch + xx] = (uint8_t)(ysum[2*j+i] + 0.5f);
            }
        }
    }

    // Write one UV pair per 2x2 block (chroma subsampled by mean).
    float U = usum / (float)count;
    float V = vsum / (float)count;
    int cx = bx;
    int cy = by;
    if (cx * 2 < W && cy * 2 < H) {
        dstUV[cy * uv_pitch + 2 * cx + 0] = (uint8_t)(U + 0.5f);
        dstUV[cy * uv_pitch + 2 * cx + 1] = (uint8_t)(V + 0.5f);
    }
}

// Fill a small fp16 buffer with a scalar (used for RIFE's timestep input, which
// TRT expects as a [1,1,1,1] tensor). Launch as a single thread.
extern "C" __global__ void fill_fp16_scalar(__half* dst, float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = __float2half(value);
    }
}
