namespace {

constexpr int kHistBinsU = 16;
constexpr int kHistBinsV = 16;
constexpr int kHistBins = kHistBinsU * kHistBinsV;
constexpr int kHistBinsL = 16;

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

} // namespace

extern "C" __global__ void kJerseyUVMean(
    const unsigned char* __restrict__ y_plane,
    int y_pitch_bytes,
    const unsigned char* __restrict__ uv_plane,
    int uv_pitch_bytes,
    int chroma_w, int chroma_h,
    const float* __restrict__ masks,
    int proto_w, int proto_h,
    int model_w, int model_h,
    const int* __restrict__ bboxes_model_xyxy,
    const int* __restrict__ det_plane_indices,
    float mask_threshold,
    int body_region_mode,
    float torso_x_margin_rel,
    float torso_y_start_rel,
    float torso_y_end_rel,
    float sample_inner_x_margin_rel,
    float sample_top_y_exclusion_rel,
    int skin_y_min,
    int skin_y_max,
    int skin_u_min,
    int skin_u_max,
    int skin_v_min,
    int skin_v_max,
    int neutral_y_min,
    int neutral_u_tol,
    int neutral_v_tol,
    float* __restrict__ out_debug_masks,
    float* __restrict__ out_best_yuv,
    int* __restrict__ out_best_count,
    int* __restrict__ out_cloth_count,
    int* __restrict__ out_skin_count,
    float* __restrict__ out_confidence,
    float* __restrict__ out_uv_hist,
    float* __restrict__ out_l_hist
) {
    const int det = blockIdx.x;
    if (det < 0) return;

    __shared__ int s_hist_count[kHistBins];
    __shared__ float s_hist_sum_y[kHistBins];
    __shared__ float s_hist_sum_u[kHistBins];
    __shared__ float s_hist_sum_v[kHistBins];
    __shared__ int s_l_hist[kHistBinsL];
    __shared__ int s_cloth_count;
    __shared__ int s_skin_count;

    const int tid = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (tid < kHistBins) {
        s_hist_count[tid] = 0;
        s_hist_sum_y[tid] = 0.0f;
        s_hist_sum_u[tid] = 0.0f;
        s_hist_sum_v[tid] = 0.0f;
    }
    if (tid < kHistBinsL) {
        s_l_hist[tid] = 0;
    }
    if (tid == 0) {
        s_cloth_count = 0;
        s_skin_count = 0;
    }
    __syncthreads();

    const int* bbox = bboxes_model_xyxy + det * 4;
    int x1 = bbox[0];
    int y1 = bbox[1];
    int x2 = bbox[2];
    int y2 = bbox[3];

    if (x2 <= x1 || y2 <= y1 || chroma_w <= 0 || chroma_h <= 0 || proto_w <= 0 || proto_h <= 0) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            out_best_yuv[det * 3] = 0.0f;
            out_best_yuv[det * 3 + 1] = 0.0f;
            out_best_yuv[det * 3 + 2] = 0.0f;
            out_best_count[det] = 0;
            out_cloth_count[det] = 0;
            out_skin_count[det] = 0;
            out_confidence[det] = 0.0f;
            for (int i = 0; i < kHistBins; ++i) out_uv_hist[det * kHistBins + i] = 0.0f;
            for (int i = 0; i < kHistBinsL; ++i) out_l_hist[det * kHistBinsL + i] = 0.0f;
        }
        return;
    }

    const float scale_x = (model_w > 0) ? ((float)chroma_w / (float)model_w) : 1.0f;
    const float scale_y = (model_h > 0) ? ((float)chroma_h / (float)model_h) : 1.0f;

    int chroma_x1 = max(0, (int)(x1 * scale_x));
    int chroma_y1 = max(0, (int)(y1 * scale_y));
    int chroma_x2 = min(chroma_w, (int)((x2 + 1) * scale_x));
    int chroma_y2 = min(chroma_h, (int)((y2 + 1) * scale_y));

    if (body_region_mode == 1) {
        const int h = chroma_y2 - chroma_y1;
        const int w = chroma_x2 - chroma_x1;
        chroma_x1 = max(0, chroma_x1 + (int)(torso_x_margin_rel * (float)w));
        chroma_x2 = min(chroma_w, chroma_x2 - (int)(torso_x_margin_rel * (float)w));
        chroma_y1 = max(0, chroma_y1 + (int)(torso_y_start_rel * (float)h));
        chroma_y2 = min(chroma_h, chroma_y1 + max(1, (int)((torso_y_end_rel - torso_y_start_rel) * (float)h)));
    }

    if (chroma_x2 <= chroma_x1 || chroma_y2 <= chroma_y1) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            out_best_yuv[det * 3] = 0.0f;
            out_best_yuv[det * 3 + 1] = 0.0f;
            out_best_yuv[det * 3 + 2] = 0.0f;
            out_best_count[det] = 0;
            out_cloth_count[det] = 0;
            out_skin_count[det] = 0;
            out_confidence[det] = 0.0f;
            for (int i = 0; i < kHistBins; ++i) out_uv_hist[det * kHistBins + i] = 0.0f;
            for (int i = 0; i < kHistBinsL; ++i) out_l_hist[det * kHistBinsL + i] = 0.0f;
        }
        return;
    }

    const int plane_idx = det_plane_indices[det];
    const int mask_stride = proto_w * proto_h;
    const float* mask = masks + plane_idx * mask_stride;
    float* debug_mask = out_debug_masks ? (out_debug_masks + det * mask_stride) : nullptr;
    if (debug_mask) {
        for (int idx = tid; idx < mask_stride; idx += (int)(blockDim.x * blockDim.y)) {
            debug_mask[idx] = 0.0f;
        }
    }
    __syncthreads();

    const int crop_w = chroma_x2 - chroma_x1;
    const int crop_h = chroma_y2 - chroma_y1;

    for (int cy = chroma_y1 + (int)threadIdx.y; cy < chroma_y2; cy += (int)blockDim.y) {
        for (int cx = chroma_x1 + (int)threadIdx.x; cx < chroma_x2; cx += (int)blockDim.x) {
            int mx = (int)((float)cx * (float)proto_w / (float)chroma_w);
            int my = (int)((float)cy * (float)proto_h / (float)chroma_h);
            mx = max(0, min(mx, proto_w - 1));
            my = max(0, min(my, proto_h - 1));
            const float m = mask[my * proto_w + mx];
            if (m <= mask_threshold) continue;

            if (body_region_mode == 1) {
                const float rel_x = (crop_w > 1) ? ((float)(cx - chroma_x1) / (float)(crop_w - 1)) : 0.5f;
                const float rel_y = (crop_h > 1) ? ((float)(cy - chroma_y1) / (float)(crop_h - 1)) : 0.5f;
                if (rel_y < sample_top_y_exclusion_rel) continue;
                if (rel_x < sample_inner_x_margin_rel || rel_x > (1.0f - sample_inner_x_margin_rel)) continue;
            }

            const unsigned char* p = uv_plane + (size_t)cy * (size_t)uv_pitch_bytes + (size_t)cx * 2u;
            const int u = (int)p[0];
            const int v = (int)p[1];
            const int yx = min((cx << 1), max(0, model_w - 1));
            const int yy = min((cy << 1), max(0, model_h - 1));
            const unsigned char* py = y_plane + (size_t)yy * (size_t)y_pitch_bytes + (size_t)yx;
            const int y00 = (int)py[0];
            const int y01 = (yx + 1 < model_w) ? (int)py[1] : y00;
            const int y10 = (yy + 1 < model_h) ? (int)*(py + y_pitch_bytes) : y00;
            const int y11 = (yy + 1 < model_h && yx + 1 < model_w) ? (int)*(py + y_pitch_bytes + 1) : y10;
            const int y = (y00 + y01 + y10 + y11) >> 2;

            if (isLikelySkinYuv(y, u, v,
                                skin_y_min, skin_y_max,
                                skin_u_min, skin_u_max,
                                skin_v_min, skin_v_max,
                                neutral_y_min, neutral_u_tol, neutral_v_tol)) {
                atomicAdd(&s_skin_count, 1);
                continue;
            }

            const int bu = min(kHistBinsU - 1, max(0, u * kHistBinsU / 256));
            const int bv = min(kHistBinsV - 1, max(0, v * kHistBinsV / 256));
            const int bin = bv * kHistBinsU + bu;
            atomicAdd(&s_hist_count[bin], 1);
            atomicAdd(&s_hist_sum_y[bin], (float)y);
            atomicAdd(&s_hist_sum_u[bin], (float)u);
            atomicAdd(&s_hist_sum_v[bin], (float)v);
            const int bl = min(kHistBinsL - 1, max(0, y * kHistBinsL / 256));
            atomicAdd(&s_l_hist[bl], 1);
            atomicAdd(&s_cloth_count, 1);
            if (debug_mask) {
                atomicExch((int*)&debug_mask[my * proto_w + mx], __float_as_int(1.0f));
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int best_bin = -1;
        int best_count = 0;
        int second_count = 0;
        for (int i = 0; i < kHistBins; ++i) {
            const int count = s_hist_count[i];
            if (count > best_count) {
                second_count = best_count;
                best_count = count;
                best_bin = i;
            } else if (count > second_count) {
                second_count = count;
            }
        }

        out_best_count[det] = best_count;
        out_cloth_count[det] = s_cloth_count;
        out_skin_count[det] = s_skin_count;
        out_confidence[det] = (s_cloth_count > 0) ? ((float)(best_count - second_count) / (float)s_cloth_count) : 0.0f;

        if (best_bin >= 0 && best_count > 0) {
            const float inv = 1.0f / (float)best_count;
            out_best_yuv[det * 3] = s_hist_sum_y[best_bin] * inv;
            out_best_yuv[det * 3 + 1] = s_hist_sum_u[best_bin] * inv;
            out_best_yuv[det * 3 + 2] = s_hist_sum_v[best_bin] * inv;
        } else {
            out_best_yuv[det * 3] = 0.0f;
            out_best_yuv[det * 3 + 1] = 0.0f;
            out_best_yuv[det * 3 + 2] = 0.0f;
        }

        const float inv_cloth = (s_cloth_count > 0) ? (1.0f / (float)s_cloth_count) : 0.0f;
        for (int i = 0; i < kHistBins; ++i) {
            out_uv_hist[det * kHistBins + i] = (float)s_hist_count[i] * inv_cloth;
        }
        for (int i = 0; i < kHistBinsL; ++i) {
            out_l_hist[det * kHistBinsL + i] = (float)s_l_hist[i] * inv_cloth;
        }
    }
}
