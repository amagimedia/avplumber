#include <limits.h>
#include <stddef.h>

extern "C" __global__ void kPlayerFeetSegFindBottom(
    const float* __restrict__ masks,
    int proto_w, int proto_h,
    int model_w, int model_h,
    const int* __restrict__ bboxes_model_xyxy,
    const int* __restrict__ det_plane_indices,
    int num_dets,
    float mask_threshold,
    float foot_x_margin_rel,
    float foot_y_start_rel,
    float foot_y_end_margin_rel,
    int* __restrict__ out_bottom_y100
) {
    const int mx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int my = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int det = (int)blockIdx.z;
    if (det >= num_dets || mx >= proto_w || my >= proto_h) return;

    const int plane_idx = det_plane_indices[det];
    if (plane_idx < 0) return;

    const int mask_stride = proto_w * proto_h;
    const int out_idx = my * proto_w + mx;
    const float* src = masks + (size_t)plane_idx * (size_t)mask_stride;
    const float src_val = src[out_idx];
    if (src_val <= mask_threshold) return;

    const int* bbox = bboxes_model_xyxy + det * 4;
    const float x1 = (float)bbox[0];
    const float y1 = (float)bbox[1];
    const float x2 = (float)bbox[2];
    const float y2 = (float)bbox[3];
    const float bw = x2 - x1;
    const float bh = y2 - y1;
    if (bw <= 1.0f || bh <= 1.0f || model_w <= 0 || model_h <= 0) return;

    const float model_x = ((float)mx + 0.5f) * (float)model_w / (float)proto_w;
    const float model_y = ((float)my + 0.5f) * (float)model_h / (float)proto_h;

    const float crop_x1 = x1 + foot_x_margin_rel * bw;
    const float crop_x2 = x2 - foot_x_margin_rel * bw;
    const float crop_y1 = y1 + foot_y_start_rel * bh;
    const float crop_y2 = y2 - foot_y_end_margin_rel * bh;
    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1) return;
    if (model_x < crop_x1 || model_x > crop_x2 || model_y < crop_y1 || model_y > crop_y2) return;

    atomicMax(out_bottom_y100 + det, __float2int_rn(model_y * 100.0f));
}

extern "C" __global__ void kPlayerFeetSegMaskAndStats(
    const float* __restrict__ masks,
    float* __restrict__ out_masks,
    int proto_w, int proto_h,
    int model_w, int model_h,
    const int* __restrict__ bboxes_model_xyxy,
    const int* __restrict__ det_plane_indices,
    int num_dets,
    float mask_threshold,
    float foot_x_margin_rel,
    float foot_y_start_rel,
    float foot_y_end_margin_rel,
    float foot_band_height_rel,
    float foot_min_band_px,
    float foot_max_band_px,
    const int* __restrict__ bottom_y100,
    int* __restrict__ out_count,
    int* __restrict__ out_sum_x100,
    int* __restrict__ out_sum_y100,
    int* __restrict__ out_min_x100,
    int* __restrict__ out_max_x100,
    int* __restrict__ out_min_y100,
    int* __restrict__ out_max_y100
) {
    const int mx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int my = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int det = (int)blockIdx.z;
    if (det >= num_dets || mx >= proto_w || my >= proto_h) return;

    const int mask_stride = proto_w * proto_h;
    const int out_idx = my * proto_w + mx;
    float* out = out_masks + (size_t)det * (size_t)mask_stride;
    out[out_idx] = 0.0f;

    const int plane_idx = det_plane_indices[det];
    if (plane_idx < 0) return;

    const float* src = masks + (size_t)plane_idx * (size_t)mask_stride;
    const float src_val = src[out_idx];
    if (src_val <= mask_threshold) return;

    const int* bbox = bboxes_model_xyxy + det * 4;
    const float x1 = (float)bbox[0];
    const float y1 = (float)bbox[1];
    const float x2 = (float)bbox[2];
    const float y2 = (float)bbox[3];
    const float bw = x2 - x1;
    const float bh = y2 - y1;
    if (bw <= 1.0f || bh <= 1.0f || model_w <= 0 || model_h <= 0) return;

    const float model_x = ((float)mx + 0.5f) * (float)model_w / (float)proto_w;
    const float model_y = ((float)my + 0.5f) * (float)model_h / (float)proto_h;

    const float crop_x1 = x1 + foot_x_margin_rel * bw;
    const float crop_x2 = x2 - foot_x_margin_rel * bw;
    const float crop_y1 = y1 + foot_y_start_rel * bh;
    const float crop_y2 = y2 - foot_y_end_margin_rel * bh;
    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1) return;
    if (model_x < crop_x1 || model_x > crop_x2 || model_y < crop_y1 || model_y > crop_y2) return;

    const int bottom100 = bottom_y100[det];
    if (bottom100 == INT_MIN) return;

    float band_h = foot_band_height_rel * bh;
    if (band_h < foot_min_band_px) band_h = foot_min_band_px;
    if (foot_max_band_px > 0.0f && band_h > foot_max_band_px) band_h = foot_max_band_px;
    if (band_h < 1.0f) band_h = 1.0f;

    const float bottom_y = (float)bottom100 / 100.0f;
    if (model_y < bottom_y - band_h || model_y > bottom_y + 0.75f) return;

    out[out_idx] = src_val;

    const int x100 = __float2int_rn(model_x * 100.0f);
    const int y100 = __float2int_rn(model_y * 100.0f);
    atomicAdd(out_count + det, 1);
    atomicAdd(out_sum_x100 + det, x100);
    atomicAdd(out_sum_y100 + det, y100);
    atomicMin(out_min_x100 + det, x100);
    atomicMax(out_max_x100 + det, x100);
    atomicMin(out_min_y100 + det, y100);
    atomicMax(out_max_y100 + det, y100);
}
