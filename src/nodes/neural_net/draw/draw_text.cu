// Draw batched text labels onto an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

#include "draw_batch_shared.hpp"
#include "draw_primitives.cuh"

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
            *y_px = blendLabelBackground(label.background_opacity, *y_px, label.bg_y_color);
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
            row[idx + 0] = blendLabelBackground(label.background_opacity, row[idx + 0], label.bg_u);
            row[idx + 1] = blendLabelBackground(label.background_opacity, row[idx + 1], label.bg_v);
        }
        if (in_text) {
            row[idx + 0] = (uint8_t)label.text_u;
            row[idx + 1] = (uint8_t)label.text_v;
        }
    }
}
