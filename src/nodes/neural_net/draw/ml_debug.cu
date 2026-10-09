// Everything drawn on one NV12 CUDA frame in one pass: boxes, labels, keypoints and trail
// segments, painted in list order with the rules of the draw kernels (draw_primitives.cuh).
//
// The host bins the items into 16x16-pixel tiles and launches one block per tile that some item
// touches. A thread owns one 2x2 quad of its tile: four luma samples and the chroma pair they
// share. It walks the tile's items in order, keeps the quad in registers and writes what changed,
// so a pixel ends as it would after the draw nodes had painted the same items one after another.
#include <stdint.h>
#include <cuda_runtime.h>

#include "draw_batch_shared.hpp"
#include "draw_primitives.cuh"

using cuda_overlay::BatchedTextLabel;
using cuda_overlay::MlDebugItem;
using cuda_overlay::MlDebugTile;

extern "C" __global__ void kMlDebugNV12(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int width, int height,
    const MlDebugTile* __restrict__ tiles,
    const int* __restrict__ tile_items,
    const MlDebugItem* __restrict__ items,
    const BatchedTextLabel* __restrict__ labels,
    const char* __restrict__ text_blob)
{
    const MlDebugTile tile = tiles[blockIdx.x];
    const int x0 = tile.x + ((int)threadIdx.x << 1);
    const int y0 = tile.y + ((int)threadIdx.y << 1);
    if (x0 >= width || y0 >= height) return;

    // The quad's luma samples: top left, top right, bottom left, bottom right. A frame of odd
    // width or height has quads with one column or row.
    const int px[4] = {x0, x0 + 1, x0, x0 + 1};
    const int py[4] = {y0, y0, y0 + 1, y0 + 1};
    const bool in_frame[4] = {true, x0 + 1 < width, y0 + 1 < height, x0 + 1 < width && y0 + 1 < height};

    uint8_t* const row0 = y_plane + (size_t)y0 * pitch_y + (size_t)x0;
    uint8_t* const sample[4] = {row0, row0 + 1, row0 + pitch_y, row0 + pitch_y + 1};
    uint8_t* const chroma = uv_plane + (size_t)(y0 >> 1) * pitch_uv + (size_t)x0;
    uint8_t luma[4] = {0, 0, 0, 0};
    bool luma_changed[4] = {false, false, false, false};
    for (int p = 0; p < 4; ++p) {
        if (in_frame[p]) luma[p] = *sample[p];
    }
    uint8_t u = chroma[0];
    uint8_t v = chroma[1];
    bool chroma_changed = false;

    for (int k = 0; k < tile.count; ++k) {
        const MlDebugItem item = items[tile_items[tile.first + k]];
        switch (item.kind) {
        case cuda_overlay::kMlDebugBox: {
            // The chroma pair takes the box's colour when any of its luma samples is on the border.
            bool any = false;
            for (int p = 0; p < 4; ++p) {
                if (!in_frame[p] || !inside_bbox_border(px[p], py[p], item.a, item.b, item.c, item.d, item.size)) continue;
                luma[p] = (uint8_t)item.y_color;
                luma_changed[p] = true;
                any = true;
            }
            if (any) {
                u = (uint8_t)item.u_color;
                v = (uint8_t)item.v_color;
                chroma_changed = true;
            }
            break;
        }
        case cuda_overlay::kMlDebugDot: {
            for (int p = 0; p < 4; ++p) {
                if (!in_frame[p] || !inside_keypoint(px[p], py[p], item.fx, item.fy, item.size)) continue;
                luma[p] = (uint8_t)item.y_color;
                luma_changed[p] = true;
            }
            // A keypoint colours a chroma pair by its top-left luma sample alone.
            if (inside_keypoint(px[0], py[0], item.fx, item.fy, item.size)) {
                u = (uint8_t)item.u_color;
                v = (uint8_t)item.v_color;
                chroma_changed = true;
            }
            break;
        }
        case cuda_overlay::kMlDebugSegment: {
            const float thickness_sq = (float)(item.size * item.size);
            bool any = false;
            for (int p = 0; p < 4; ++p) {
                if (!in_frame[p] ||
                    !(point_segment_dist_sq(px[p], py[p], item.a, item.b, item.c, item.d) <= thickness_sq)) continue;
                luma[p] = (uint8_t)item.y_color;
                luma_changed[p] = true;
                any = true;
            }
            if (any) {
                u = (uint8_t)item.u_color;
                v = (uint8_t)item.v_color;
                chroma_changed = true;
            }
            break;
        }
        case cuda_overlay::kMlDebugLabel: {
            const BatchedTextLabel label = labels[item.a];
            const char* line1 = label.line1_len > 0 ? (text_blob + label.line1_offset) : nullptr;
            const char* line2 = label.line2_len > 0 ? (text_blob + label.line2_offset) : nullptr;
            const char* line3 = label.line3_len > 0 ? (text_blob + label.line3_offset) : nullptr;
            bool in_bg = false;
            bool in_text = false;
            for (int p = 0; p < 4; ++p) {
                if (!in_frame[p] ||
                    px[p] < label.bg_x || px[p] >= (label.bg_x + label.bg_w) ||
                    py[p] < label.bg_y || py[p] >= (label.bg_y + label.bg_h)) {
                    continue;
                }
                if (label.draw_background) {
                    luma[p] = blendLabelBackground(label.background_opacity, luma[p], label.bg_y_color);
                    luma_changed[p] = true;
                    in_bg = true;
                }
                if (isTextPixel(line1, line2, line3,
                                px[p], py[p], label.origin_x, label.origin_y,
                                label.font_scale, label.line_spacing, label.glyph_preset,
                                label.line1_len, label.line2_len, label.line3_len)) {
                    luma[p] = (uint8_t)label.text_y;
                    luma_changed[p] = true;
                    in_text = true;
                }
            }
            if (in_bg) {
                u = blendLabelBackground(label.background_opacity, u, label.bg_u);
                v = blendLabelBackground(label.background_opacity, v, label.bg_v);
                chroma_changed = true;
            }
            if (in_text) {
                u = (uint8_t)label.text_u;
                v = (uint8_t)label.text_v;
                chroma_changed = true;
            }
            break;
        }
        default:
            break;
        }
    }

    for (int p = 0; p < 4; ++p) {
        if (luma_changed[p]) *sample[p] = luma[p];
    }
    if (chroma_changed) {
        chroma[0] = u;
        chroma[1] = v;
    }
}
