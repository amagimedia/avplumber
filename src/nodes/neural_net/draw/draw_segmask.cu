// Draw segmentation mask overlay onto an NV12 CUDA frame.
// Each mask covers the full prototype spatial area (proto_h x proto_w) which maps
// to the full model input (model_w x model_h). The kernel samples the mask at the
// prototype coordinate corresponding to each frame pixel and alpha-blends.
#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kDrawSegMaskNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    const float* __restrict__ mask,
    int proto_w, int proto_h,
    int model_w, int model_h,
    int bbox_x1, int bbox_y1, int bbox_x2, int bbox_y2,
    float scale_x, float scale_y, float offset_x, float offset_y,
    int frame_w, int frame_h,
    int y_color, float opacity, float threshold)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x) + bbox_x1;
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y) + bbox_y1;
    if (x >= bbox_x2 || y >= bbox_y2 || x >= frame_w || y >= frame_h) return;

    // Frame pixel -> model space -> prototype space
    const float model_x = (float)x * scale_x + offset_x;
    const float model_y = (float)y * scale_y + offset_y;
    const float px = model_x * (float)proto_w / (float)model_w;
    const float py = model_y * (float)proto_h / (float)model_h;

    // Bilinear sample
    const int px0 = max(0, min((int)floorf(px), proto_w - 1));
    const int py0 = max(0, min((int)floorf(py), proto_h - 1));
    const int px1 = min(px0 + 1, proto_w - 1);
    const int py1 = min(py0 + 1, proto_h - 1);
    const float fx = px - floorf(px);
    const float fy = py - floorf(py);

    const float v00 = mask[py0 * proto_w + px0];
    const float v10 = mask[py0 * proto_w + px1];
    const float v01 = mask[py1 * proto_w + px0];
    const float v11 = mask[py1 * proto_w + px1];
    const float val = v00 * (1.f - fx) * (1.f - fy) + v10 * fx * (1.f - fy)
                    + v01 * (1.f - fx) * fy + v11 * fx * fy;

    if (val < threshold) return;

    const float alpha = val * opacity;
    const uint8_t orig = y_plane[(size_t)y * pitch_y + (size_t)x];
    const float blended = (float)orig * (1.f - alpha) + (float)y_color * alpha;
    y_plane[(size_t)y * pitch_y + (size_t)x] = (uint8_t)min(max((int)(blended + 0.5f), 0), 255);
}

extern "C" __global__ void kDrawSegMaskNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    const float* __restrict__ mask,
    int proto_w, int proto_h,
    int model_w, int model_h,
    int bbox_x1, int bbox_y1, int bbox_x2, int bbox_y2,
    float scale_x, float scale_y, float offset_x, float offset_y,
    int frame_w, int frame_h,
    int u_color, int v_color, float opacity, float threshold)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x) + (bbox_x1 >> 1);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y) + (bbox_y1 >> 1);
    const int uv_bbox_x2 = (bbox_x2 + 1) >> 1;
    const int uv_bbox_y2 = (bbox_y2 + 1) >> 1;
    const int uv_frame_w = (frame_w + 1) >> 1;
    const int uv_frame_h = (frame_h + 1) >> 1;
    if (uv_x >= uv_bbox_x2 || uv_y >= uv_bbox_y2 || uv_x >= uv_frame_w || uv_y >= uv_frame_h) return;

    const int luma_x = (uv_x << 1);
    const int luma_y = (uv_y << 1);

    const float model_x = (float)luma_x * scale_x + offset_x;
    const float model_y = (float)luma_y * scale_y + offset_y;
    const float ppx = model_x * (float)proto_w / (float)model_w;
    const float ppy = model_y * (float)proto_h / (float)model_h;

    const int px0 = max(0, min((int)floorf(ppx), proto_w - 1));
    const int py0 = max(0, min((int)floorf(ppy), proto_h - 1));
    const int px1 = min(px0 + 1, proto_w - 1);
    const int py1 = min(py0 + 1, proto_h - 1);
    const float fx = ppx - floorf(ppx);
    const float fy = ppy - floorf(ppy);

    const float v00 = mask[py0 * proto_w + px0];
    const float v10 = mask[py0 * proto_w + px1];
    const float v01 = mask[py1 * proto_w + px0];
    const float v11 = mask[py1 * proto_w + px1];
    const float val = v00 * (1.f - fx) * (1.f - fy) + v10 * fx * (1.f - fy)
                    + v01 * (1.f - fx) * fy + v11 * fx * fy;

    if (val < threshold) return;

    const float alpha = val * opacity;
    uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
    const uint8_t orig_u = row[(size_t)(uv_x << 1) + 0];
    const uint8_t orig_v = row[(size_t)(uv_x << 1) + 1];
    row[(size_t)(uv_x << 1) + 0] = (uint8_t)min(max((int)((float)orig_u * (1.f - alpha) + (float)u_color * alpha + 0.5f), 0), 255);
    row[(size_t)(uv_x << 1) + 1] = (uint8_t)min(max((int)((float)orig_v * (1.f - alpha) + (float)v_color * alpha + 0.5f), 0), 255);
}
