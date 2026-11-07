#pragma once
// Rect table shared by the compositor host code (cuda_rect_draw.cpp) and the
// single-launch composite kernel (cuda_rect_scale.cu). Plain C layout: nvcc
// and the host compiler must agree on it byte for byte.
//
// One entry per resolved draw op, in draw order (bottom to top). Geometry is
// precomputed per canvas plane: index 0 is luma, 1 is the interleaved chroma
// plane, both in lane-group units (luma samples, chroma pairs).

// Luma samples per thread in the composite kernel (chroma pairs: half). Block 32x8 threads
// therefore covers a 128x8 luma tile; the launch grid is sized from this.
#define AVP_RECT_PX 4

// Draw kinds, same numbers on both sides. The order is a contract: kinds >= RGB are
// packed RGB(A) on one plane, kinds >= RGB_TEX read that plane through src[0] as a
// texture object.
#define AVP_RECT_KIND_YUV 0       // source in the canvas format: bilinear scale/blit
#define AVP_RECT_KIND_PROMOTE 1   // lower-depth semiplanar source: bilinear + code multiply
#define AVP_RECT_KIND_RGB 2       // packed 8-bit RGB(A) treated opaque: fused convert
#define AVP_RECT_KIND_RGBA 3      // packed 8-bit RGBA blended over what is below
#define AVP_RECT_KIND_RGB_TEX 4   // as RGB, but src[0] is a CUtexObject over a CUDA array (zero-copy DMA-BUF)
#define AVP_RECT_KIND_RGBA_TEX 5  // as RGBA, texture-backed
// Below the packed kinds, so the order above holds. Drawn only by the *_filter entry points:
// its codes need the limit to the canvas range that the entries without filters leave out.
#define AVP_RECT_KIND_DEMOTE -1   // higher-depth semiplanar source: sample, code multiply below 1, limit

// How a YUV/PROMOTE/DEMOTE layer is resampled, decided by the host once per layer (ScaleFilter
// in compositor_geometry.hpp resolves `auto`). 0 is the bilinear path every kernel has; the
// other values are drawn only by the *_filter entry points, which the host launches when a
// table holds one or a DEMOTE layer. Packed RGB(A) layers and packed canvases are always bilinear.
#define AVP_RECT_FILTER_BILINEAR 0
#define AVP_RECT_FILTER_BICUBIC 1        // 4x4 cubic, coefficient `filter_param`
#define AVP_RECT_FILTER_BICUBIC_A0 2     // the cubic at A = 0: its outer taps weigh nothing, so 2x2 taps
#define AVP_RECT_FILTER_MULTISAMPLE4 3   // mean of 4 bilinear samples, 2x2 grid over the pixel's source area
#define AVP_RECT_FILTER_MULTISAMPLE8 4   // mean of 8 bilinear samples, Direct3D 8-sample pattern

struct AvpRectLayer {
    unsigned long long src[2];   // device pointers, or texture handles when yuv_texture / *_TEX is set
    int src_pitch[2];
    int sx[2], sy[2], sw[2], sh[2];   // source rect per plane, lane groups (RGB: packed pixels in [0])
    int dx[2], dy[2], dw[2], dh[2];   // destination rect per canvas plane, lane groups
    int kind;
    int yuv_texture;                  // YUV/PROMOTE/DEMOTE: src[] holds exact integer plane textures
    int src_bytes, src_shift;         // PROMOTE, DEMOTE: source storage; YUV: same as the canvas
    float mul;                        // PROMOTE, DEMOTE: code multiplier; YUV, RGB: 1; RGBA: opacity in (0, 1]
    int step, r_off, g_off, b_off, a_off, premultiplied;   // RGB(A)
    int filter;                       // YUV/PROMOTE/DEMOTE: AVP_RECT_FILTER_*
    float filter_param;               // BICUBIC: the cubic's A (0: Hermite, -0.5: Catmull-Rom)
    // Appended for the affine entries only (composite_planes_yuv_affine, composite_planes_affine;
    // compositor_affine.hpp computes them): with `affine` set, plane p maps destination continuous
    // coordinates (canvas-absolute, plane units) to source ones, u = m0 x + m1 y + m2,
    // v = m3 x + m4 y + m5; sx..sh are the taps' rect, dx..dh the soft-edged quad's bounds. Packed
    // RGB(A) sources use plane 0's values for every luma position.
    int affine;
    float m[2][6];
    float edge[2][4];                 // u0, v0, u1, v1: the crop's edges for the coverage
    float aa[2][2];                   // destination samples per source unit across a u- and a v-edge
    float footprint[2][2];            // source units per destination sample along u and v (multisample)
    float transparency;               // 1 - opacity of a faded non-blend layer: scales the coverage (0: opaque)
};
