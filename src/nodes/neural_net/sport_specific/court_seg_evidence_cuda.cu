#include <stddef.h>

extern "C" __global__ void kCourtSegEvidenceDownsample(
    const float* __restrict__ masks,
    float* __restrict__ out_masks,
    unsigned int* __restrict__ counts,
    int num_masks,
    int proto_w,
    int proto_h,
    int out_w,
    int out_h,
    float threshold,
    int write_masks
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int det = (int)blockIdx.z;
    if (det >= num_masks || x >= out_w || y >= out_h) return;

    const float src_x = (float)x * (float)proto_w / (float)out_w;
    const float src_y = (float)y * (float)proto_h / (float)out_h;
    const int x0 = (int)src_x;
    const int y0 = (int)src_y;
    const int x1 = min(x0 + 1, proto_w - 1);
    const int y1 = min(y0 + 1, proto_h - 1);
    const float fx = src_x - (float)x0;
    const float fy = src_y - (float)y0;

    const size_t src_stride = (size_t)proto_w * (size_t)proto_h;
    const float* src = masks + (size_t)det * src_stride;
    const float v =
        src[(size_t)y0 * (size_t)proto_w + (size_t)x0] * (1.0f - fx) * (1.0f - fy) +
        src[(size_t)y0 * (size_t)proto_w + (size_t)x1] * fx * (1.0f - fy) +
        src[(size_t)y1 * (size_t)proto_w + (size_t)x0] * (1.0f - fx) * fy +
        src[(size_t)y1 * (size_t)proto_w + (size_t)x1] * fx * fy;

    if (write_masks && out_masks) {
        const size_t dst_stride = (size_t)out_w * (size_t)out_h;
        out_masks[(size_t)det * dst_stride + (size_t)y * (size_t)out_w + (size_t)x] = v;
    }
    if (v >= threshold) {
        atomicAdd(counts + det, 1u);
    }
}
