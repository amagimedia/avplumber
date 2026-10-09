// Device functions shared by the draw kernels: what a pixel of a box border, a trail segment and a
// label is. Each draw kernel and the one-pass ml_debug kernel include this file, so a primitive
// has one definition and the kernels cannot disagree about a pixel.
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ bool inside_bbox_border(int x, int y,
                                                   int x1, int y1,
                                                   int x2, int y2,
                                                   int thickness) {
    if (x < x1 || x >= x2 || y < y1 || y >= y2) return false;
    return x < (x1 + thickness) || x >= (x2 - thickness)
        || y < (y1 + thickness) || y >= (y2 - thickness);
}

// Squared distance from point (px, py) to line segment (ax, ay)-(bx, by)
__device__ __forceinline__ float point_segment_dist_sq(int px, int py,
                                                        int ax, int ay,
                                                        int bx, int by) {
    float dx = (float)(bx - ax);
    float dy = (float)(by - ay);
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) {
        // Degenerate segment (point)
        float ex = (float)(px - ax);
        float ey = (float)(py - ay);
        return ex * ex + ey * ey;
    }
    float t = ((float)(px - ax) * dx + (float)(py - ay) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));
    float proj_x = (float)ax + t * dx;
    float proj_y = (float)ay + t * dy;
    float ex = (float)px - proj_x;
    float ey = (float)py - proj_y;
    return ex * ex + ey * ey;
}

// Whether pixel (px, py) is in the filled dot of `radius` around (cx, cy).
__device__ __forceinline__ bool inside_keypoint(int px, int py, float cx, float cy, int radius) {
    const int r2 = radius * radius;
    float dx = (float)px - cx;
    float dy = (float)py - cy;
    return dx * dx + dy * dy <= (float)r2;
}

// A label's background over one 8-bit sample: opacity clamped to [0,1], rounded to nearest.
__device__ __forceinline__ uint8_t blendLabelBackground(float opacity, uint8_t sample, int color) {
    const float a = opacity < 0.f ? 0.f : (opacity > 1.f ? 1.f : opacity);
    const float out = (1.0f - a) * (float)sample + a * (float)color;
    return (uint8_t)(out + 0.5f);
}

#define PACK7(r0, r1, r2, r3, r4, r5, r6) \
    ((uint64_t)(r0) | ((uint64_t)(r1) << 5) | ((uint64_t)(r2) << 10) | \
     ((uint64_t)(r3) << 15) | ((uint64_t)(r4) << 20) | ((uint64_t)(r5) << 25) | \
     ((uint64_t)(r6) << 30))

__device__ __forceinline__ uint8_t glyphRowBits(char c, int row) {
    uint64_t packed = 0;
    switch (c) {
    case 'A': packed = PACK7(0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11); break;
    case 'B': packed = PACK7(0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E); break;
    case 'C': packed = PACK7(0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E); break;
    case 'D': packed = PACK7(0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E); break;
    case 'E': packed = PACK7(0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F); break;
    case 'F': packed = PACK7(0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10); break;
    case 'G': packed = PACK7(0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0F); break;
    case 'H': packed = PACK7(0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11); break;
    case 'I': packed = PACK7(0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E); break;
    case 'J': packed = PACK7(0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E); break;
    case 'K': packed = PACK7(0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11); break;
    case 'L': packed = PACK7(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F); break;
    case 'M': packed = PACK7(0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11); break;
    case 'N': packed = PACK7(0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11); break;
    case 'O': packed = PACK7(0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E); break;
    case 'P': packed = PACK7(0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10); break;
    case 'Q': packed = PACK7(0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D); break;
    case 'R': packed = PACK7(0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11); break;
    case 'S': packed = PACK7(0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E); break;
    case 'T': packed = PACK7(0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04); break;
    case 'U': packed = PACK7(0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E); break;
    case 'V': packed = PACK7(0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04); break;
    case 'W': packed = PACK7(0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A); break;
    case 'X': packed = PACK7(0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11); break;
    case 'Y': packed = PACK7(0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04); break;
    case 'Z': packed = PACK7(0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F); break;
    case '0': packed = PACK7(0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E); break;
    case '1': packed = PACK7(0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E); break;
    case '2': packed = PACK7(0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F); break;
    case '3': packed = PACK7(0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E); break;
    case '4': packed = PACK7(0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02); break;
    case '5': packed = PACK7(0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E); break;
    case '6': packed = PACK7(0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E); break;
    case '7': packed = PACK7(0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08); break;
    case '8': packed = PACK7(0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E); break;
    case '9': packed = PACK7(0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C); break;
    case ':': packed = PACK7(0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00); break;
    case '_': packed = PACK7(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F); break;
    case '-': packed = PACK7(0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00); break;
    case '.': packed = PACK7(0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06); break;
    case ' ': default: packed = PACK7(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); break;
    }
    return (uint8_t)((packed >> (row * 5)) & 0x1F);
}

__device__ __forceinline__ bool glyphBit5x7(char c, int row, int col) {
    if (row < 0 || row >= 7 || col < 0 || col >= 5) return false;
    const uint8_t row_bits = glyphRowBits(c, row);
    return (row_bits & (1u << (4 - col))) != 0;
}

__device__ __forceinline__ void glyphDims(int glyph_preset, int& glyph_w, int& glyph_h) {
    if (glyph_preset == 1) {
        glyph_w = 10;
        glyph_h = 14;
    } else {
        glyph_w = 5;
        glyph_h = 7;
    }
}

__device__ __forceinline__ bool glyphBit(char c, int row, int col, int glyph_preset) {
    if (glyph_preset == 1) {
        return glyphBit5x7(c, row >> 1, col >> 1);
    }
    return glyphBit5x7(c, row, col);
}

__device__ __forceinline__ bool isGlyphPixel(const char* text, int len,
                                             int x, int y,
                                             int start_x, int start_y,
                                             int font_scale, int glyph_preset) {
    if (!text || len <= 0) return false;
    const int rel_x = x - start_x;
    const int rel_y = y - start_y;
    if (rel_x < 0 || rel_y < 0) return false;

    int glyph_w = 5;
    int glyph_h = 7;
    glyphDims(glyph_preset, glyph_w, glyph_h);
    const int char_advance = (glyph_w + 1) * font_scale;
    const int line_height = glyph_h * font_scale;
    if (rel_y >= line_height) return false;

    const int char_idx = rel_x / char_advance;
    if (char_idx < 0 || char_idx >= len) return false;

    const int glyph_x = (rel_x % char_advance) / font_scale;
    if (glyph_x >= glyph_w) return false;
    const int glyph_y = rel_y / font_scale;
    if (glyph_y < 0 || glyph_y >= glyph_h) return false;
    return glyphBit(text[char_idx], glyph_y, glyph_x, glyph_preset);
}

__device__ __forceinline__ bool isTextPixel(const char* line1, const char* line2, const char* line3,
                                            int x, int y,
                                            int origin_x, int origin_y,
                                            int font_scale, int line_spacing,
                                            int glyph_preset,
                                            int line1_len, int line2_len, int line3_len) {
    int glyph_w = 5;
    int glyph_h = 7;
    glyphDims(glyph_preset, glyph_w, glyph_h);
    if (isGlyphPixel(line1, line1_len, x, y, origin_x, origin_y, font_scale, glyph_preset)) {
        return true;
    }
    const int line2_y = origin_y + glyph_h * font_scale + line_spacing;
    if (isGlyphPixel(line2, line2_len, x, y, origin_x, line2_y, font_scale, glyph_preset)) {
        return true;
    }
    const int line3_y = line2_y + glyph_h * font_scale + line_spacing;
    return isGlyphPixel(line3, line3_len, x, y, origin_x, line3_y, font_scale, glyph_preset);
}

} // namespace
