// Draw batched text labels onto an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

#include "draw_batch_shared.hpp"

namespace {

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

extern "C" __global__ void kDrawTextNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    int frame_width, int frame_height,
    const cuda_overlay::BatchedTextLabel* __restrict__ labels,
    int num_labels,
    const char* __restrict__ text_blob)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x < 0 || y < 0 || x >= frame_width || y >= frame_height) return;

    uint8_t* y_px = &y_plane[(size_t)y * pitch_y + (size_t)x];
    for (int i = 0; i < num_labels; ++i) {
        const cuda_overlay::BatchedTextLabel label = labels[i];
        if (x < label.bg_x || x >= (label.bg_x + label.bg_w) ||
            y < label.bg_y || y >= (label.bg_y + label.bg_h)) {
            continue;
        }

        if (label.draw_background) {
            const float a = label.background_opacity < 0.f ? 0.f : (label.background_opacity > 1.f ? 1.f : label.background_opacity);
            const float out = (1.0f - a) * (float)(*y_px) + a * (float)label.bg_y_color;
            *y_px = (uint8_t)(out + 0.5f);
        }

        const char* line1 = label.line1_len > 0 ? (text_blob + label.line1_offset) : nullptr;
        const char* line2 = label.line2_len > 0 ? (text_blob + label.line2_offset) : nullptr;
        const char* line3 = label.line3_len > 0 ? (text_blob + label.line3_offset) : nullptr;
        if (isTextPixel(line1, line2, line3,
                        x, y, label.origin_x, label.origin_y,
                        label.font_scale, label.line_spacing, label.glyph_preset,
                        label.line1_len, label.line2_len, label.line3_len)) {
            *y_px = (uint8_t)label.text_y;
        }
    }
}

extern "C" __global__ void kDrawTextNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int frame_width, int frame_height,
    const cuda_overlay::BatchedTextLabel* __restrict__ labels,
    int num_labels,
    const char* __restrict__ text_blob)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int uv_w = (frame_width + 1) >> 1;
    const int uv_h = (frame_height + 1) >> 1;
    if (uv_x >= uv_w || uv_y >= uv_h) return;

    const int x0 = uv_x << 1;
    const int y0 = uv_y << 1;
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
    const size_t idx = (size_t)(uv_x << 1);
    for (int i = 0; i < num_labels; ++i) {
        const cuda_overlay::BatchedTextLabel label = labels[i];
        const int px[4] = {x0, x1, x0, x1};
        const int py[4] = {y0, y0, y1, y1};
        bool in_bg = false;
        bool in_text = false;
        const char* line1 = label.line1_len > 0 ? (text_blob + label.line1_offset) : nullptr;
        const char* line2 = label.line2_len > 0 ? (text_blob + label.line2_offset) : nullptr;
        const char* line3 = label.line3_len > 0 ? (text_blob + label.line3_offset) : nullptr;
        for (int p = 0; p < 4; ++p) {
            if (px[p] < label.bg_x || px[p] >= (label.bg_x + label.bg_w) ||
                py[p] < label.bg_y || py[p] >= (label.bg_y + label.bg_h) ||
                px[p] < 0 || py[p] < 0 || px[p] >= frame_width || py[p] >= frame_height) {
                continue;
            }
            if (label.draw_background) in_bg = true;
            if (isTextPixel(line1, line2, line3,
                            px[p], py[p], label.origin_x, label.origin_y,
                            label.font_scale, label.line_spacing, label.glyph_preset,
                            label.line1_len, label.line2_len, label.line3_len)) {
                in_text = true;
            }
        }

        if (in_bg) {
            const float a = label.background_opacity < 0.f ? 0.f : (label.background_opacity > 1.f ? 1.f : label.background_opacity);
            const float out_u = (1.0f - a) * (float)row[idx + 0] + a * (float)label.bg_u;
            const float out_v = (1.0f - a) * (float)row[idx + 1] + a * (float)label.bg_v;
            row[idx + 0] = (uint8_t)(out_u + 0.5f);
            row[idx + 1] = (uint8_t)(out_v + 0.5f);
        }
        if (in_text) {
            row[idx + 0] = (uint8_t)label.text_u;
            row[idx + 1] = (uint8_t)label.text_v;
        }
    }
}
