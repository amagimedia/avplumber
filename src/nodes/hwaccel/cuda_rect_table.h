#pragma once
// Rect table shared by the compositor host code (cuda_rect_draw.cpp) and the
// single-launch composite kernel (cuda_rect_scale.cu). Plain C layout: nvcc
// and the host compiler must agree on it byte for byte.
//
// One entry per resolved draw op, in draw order (bottom to top). Geometry is
// precomputed per canvas plane: index 0 is luma, 1 is the interleaved chroma
// plane, both in lane-group units (luma samples, chroma pairs) exactly as the
// per-layer kernels receive them, so the batched kernel samples identically.

#define AVP_RECT_MAX_LAYERS 256
// Luma samples per thread in the composite kernel (chroma pairs: half). Block 32x8 threads
// therefore covers a 128x8 luma tile; the launch grid is sized from this.
#define AVP_RECT_PX 4

// Draw kinds. Same numbers on both sides.
#define AVP_RECT_KIND_YUV 0       // source in the canvas format: bilinear scale/blit
#define AVP_RECT_KIND_PROMOTE 1   // lower-depth semiplanar source: bilinear + code multiply
#define AVP_RECT_KIND_RGB 2       // packed 8-bit RGB(A) treated opaque: fused convert
#define AVP_RECT_KIND_RGBA 3      // packed 8-bit RGBA blended over what is below
#define AVP_RECT_KIND_RGB_TEX 4   // as RGB, but src[0] is a CUtexObject over a CUDA array (zero-copy DMA-BUF)
#define AVP_RECT_KIND_RGBA_TEX 5  // as RGBA, texture-backed

struct AvpRectLayer {
    unsigned long long src[2];   // device pointers of the source planes (RGB: [0] only; *_TEX: [0] is the texture)
    int src_pitch[2];
    int sx[2], sy[2], sw[2], sh[2];   // source rect per plane, lane groups (RGB: packed pixels in [0])
    int dx[2], dy[2], dw[2], dh[2];   // destination rect per canvas plane, lane groups
    int kind;
    int src_bytes, src_shift;         // PROMOTE: source storage; YUV: same as the canvas
    float mul;                        // PROMOTE: code multiplier; YUV: 1
    int step, r_off, g_off, b_off, a_off, premultiplied;   // RGB(A)
};
