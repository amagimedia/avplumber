#include <stddef.h>

namespace {

__device__ __forceinline__ bool isLikelySkinYuv(
    int y, int u, int v,
    int skin_y_min, int skin_y_max,
    int skin_u_min, int skin_u_max,
    int skin_v_min, int skin_v_max,
    int neutral_y_min,
    int neutral_u_tol,
    int neutral_v_tol
) {
    (void)neutral_y_min;
    if (abs(u - 128) <= neutral_u_tol &&
        abs(v - 128) <= neutral_v_tol) {
        return false;
    }
    return y >= skin_y_min && y <= skin_y_max &&
           u >= skin_u_min && u <= skin_u_max &&
           v >= skin_v_min && v <= skin_v_max;
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

} // namespace

extern "C" __global__ void kPlayerTorsoSegMask(
    const unsigned char* __restrict__ y_plane,
    int y_pitch_bytes,
    const unsigned char* __restrict__ uv_plane,
    int uv_pitch_bytes,
    int frame_w, int frame_h,
    const float* __restrict__ masks,
    float* __restrict__ out_masks,
    int proto_w, int proto_h,
    int model_w, int model_h,
    const int* __restrict__ bboxes_model_xyxy,
    const int* __restrict__ det_plane_indices,
    int num_dets,
    float mask_threshold,
    float torso_x_margin_rel,
    float torso_y_start_rel,
    float torso_y_end_rel,
    float sample_inner_x_margin_rel,
    float sample_top_y_exclusion_rel,
    int skin_filter_enabled,
    int skin_y_min,
    int skin_y_max,
    int skin_u_min,
    int skin_u_max,
    int skin_v_min,
    int skin_v_max,
    int neutral_y_min,
    int neutral_u_tol,
    int neutral_v_tol
) {
    const int mx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int my = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int det = (int)blockIdx.z;
    if (det >= num_dets || mx >= proto_w || my >= proto_h) return;

    const int mask_stride = proto_w * proto_h;
    float* out = out_masks + (size_t)det * (size_t)mask_stride;
    const int out_idx = my * proto_w + mx;
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

    const float crop_x1 = x1 + torso_x_margin_rel * bw;
    const float crop_x2 = x2 - torso_x_margin_rel * bw;
    const float crop_y1 = y1 + torso_y_start_rel * bh;
    const float crop_y2 = y1 + torso_y_end_rel * bh;
    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1) return;

    const float model_x = ((float)mx + 0.5f) * (float)model_w / (float)proto_w;
    const float model_y = ((float)my + 0.5f) * (float)model_h / (float)proto_h;
    if (model_x < crop_x1 || model_x > crop_x2 || model_y < crop_y1 || model_y > crop_y2) return;

    const float rel_x = (model_x - crop_x1) / max(1.0f, crop_x2 - crop_x1);
    const float rel_y = (model_y - crop_y1) / max(1.0f, crop_y2 - crop_y1);
    if (rel_y < sample_top_y_exclusion_rel) return;
    if (rel_x < sample_inner_x_margin_rel || rel_x > (1.0f - sample_inner_x_margin_rel)) return;

    if (skin_filter_enabled && y_plane && uv_plane && frame_w > 1 && frame_h > 1) {
        const int lx = clampi((int)(model_x * (float)frame_w / (float)model_w), 0, frame_w - 1);
        const int ly = clampi((int)(model_y * (float)frame_h / (float)model_h), 0, frame_h - 1);
        const unsigned char* py = y_plane + (size_t)ly * (size_t)y_pitch_bytes + (size_t)lx;
        const int y = (int)(*py);

        const int uv_w = (frame_w + 1) >> 1;
        const int uv_h = (frame_h + 1) >> 1;
        const int ux = clampi(lx >> 1, 0, uv_w - 1);
        const int uy = clampi(ly >> 1, 0, uv_h - 1);
        const unsigned char* puv = uv_plane + (size_t)uy * (size_t)uv_pitch_bytes + (size_t)ux * 2u;
        const int u = (int)puv[0];
        const int v = (int)puv[1];

        if (isLikelySkinYuv(y, u, v,
                            skin_y_min, skin_y_max,
                            skin_u_min, skin_u_max,
                            skin_v_min, skin_v_max,
                            neutral_y_min, neutral_u_tol, neutral_v_tol)) {
            return;
        }
    }

    out[out_idx] = src_val;
}
