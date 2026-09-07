#include <cuda_runtime.h>

// Bilinear interpolation. This lightweight
// sampler does not widen its support when downscaling. Each lane is sampled
// separately, including interleaved UV.
extern "C" __global__ void scale_plane(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    unsigned char *dst, int dst_pitch, int dx, int dy, int dw, int dh,
    int canvas_w, int canvas_h, int lanes) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = dx + ox, y = dy + oy;
    if (ox >= dw || oy >= dh || x < 0 || y < 0 || x >= canvas_w || y >= canvas_h) return;
    const float fx = (ox + 0.5f) * sw / dw - 0.5f;
    const float fy = (oy + 0.5f) * sh / dh - 0.5f;
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1));
    const int x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1));
    const int y1 = sy + max(0, min(iy + 1, sh - 1));
    for (int c = 0; c < lanes; ++c) {
        const float a = src[y0 * src_pitch + x0 * lanes + c];
        const float b = src[y0 * src_pitch + x1 * lanes + c];
        const float d = src[y1 * src_pitch + x0 * lanes + c];
        const float e = src[y1 * src_pitch + x1 * lanes + c];
        const float top = a + tx * (b - a), bottom = d + tx * (e - d);
        dst[y * dst_pitch + x * lanes + c] = (unsigned char)(top + ty * (bottom - top) + 0.5f);
    }
}
