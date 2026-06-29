#include <stdint.h>
#include <cuda_runtime.h>

struct FlowVector {
    int16_t flowx;
    int16_t flowy;
};

extern "C" __global__ void kCameraMotionAffineReduce(
    const FlowVector* __restrict__ flow,
    const uint8_t* __restrict__ cost,
    int flow_stride_bytes,
    int cost_stride_bytes,
    int grid_w,
    int grid_h,
    int grid_size,
    int image_w,
    int image_h,
    const float4* __restrict__ boxes,
    int num_boxes,
    float mask_margin_frac,
    int use_weights,
    double p0,
    double p1,
    double p2,
    double p3,
    double p4,
    double p5,
    double huber,
    double* __restrict__ block_out)
{
    extern __shared__ double shared[];
    const int tid = (int)threadIdx.x;
    const int n = grid_w * grid_h;
    const int stride = (int)(blockDim.x * gridDim.x);

    double xx = 0.0;
    double xy = 0.0;
    double sx = 0.0;
    double yy = 0.0;
    double sy = 0.0;
    double sw = 0.0;
    double bx0 = 0.0;
    double bx1 = 0.0;
    double bx2 = 0.0;
    double by0 = 0.0;
    double by1 = 0.0;
    double by2 = 0.0;
    double point_count = 0.0;
    double inlier_count = 0.0;
    double residual_sum = 0.0;
    double cost_sum = 0.0;
    double mag_sum = 0.0;
    double mag_min = 1.0e30;
    double total_count = 0.0;
    double bg_count = 0.0;

    for (int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x); idx < n; idx += stride) {
        const int gy = idx / grid_w;
        const int gx = idx - gy * grid_w;
        const FlowVector* row = (const FlowVector*)((const char*)flow + (size_t)gy * (size_t)flow_stride_bytes);
        const uint8_t* cost_row = (const uint8_t*)((const char*)cost + (size_t)gy * (size_t)cost_stride_bytes);
        const FlowVector v = row[gx];
        const double dx = (double)v.flowx / 32.0;
        const double dy = (double)v.flowy / 32.0;
        const double x = ((double)gx + 0.5) * (double)grid_size;
        const double y = ((double)gy + 0.5) * (double)grid_size;
        const float cxn = (((float)gx + 0.5f) * (float)grid_size) / (float)image_w;
        const float cyn = (((float)gy + 0.5f) * (float)grid_size) / (float)image_h;

        total_count += 1.0;
        cost_sum += (double)cost_row[gx];

        int masked = 0;
        for (int bi = 0; bi < num_boxes; ++bi) {
            const float4 b = boxes[bi];
            const float mw = mask_margin_frac * fmaxf(0.0f, b.z - b.x);
            const float mh = mask_margin_frac * fmaxf(0.0f, b.w - b.y);
            if (cxn >= b.x - mw && cxn <= b.z + mw && cyn >= b.y - mh && cyn <= b.w + mh) {
                masked = 1;
                break;
            }
        }
        if (masked) continue;

        bg_count += 1.0;
        double w = 1.0;
        const double px = p0 * x + p1 * y + p2;
        const double py = p3 * x + p4 * y + p5;
        const double ex = px - (x + dx);
        const double ey = py - (y + dy);
        const double r = sqrt(ex * ex + ey * ey);
        if (use_weights && r > huber && r > 1.0e-9) {
            w = huber / r;
        }

        point_count += 1.0;
        if (r <= huber) {
            inlier_count += 1.0;
            residual_sum += r;
        }
        mag_sum += sqrt(dx * dx + dy * dy);
        mag_min = fmin(mag_min, sqrt(dx * dx + dy * dy));

        const double dstx = x + dx;
        const double dsty = y + dy;
        xx += w * x * x;
        xy += w * x * y;
        sx += w * x;
        yy += w * y * y;
        sy += w * y;
        sw += w;
        bx0 += w * x * dstx;
        bx1 += w * y * dstx;
        bx2 += w * dstx;
        by0 += w * x * dsty;
        by1 += w * y * dsty;
        by2 += w * dsty;
    }

    double vals[20] = {
        xx, xy, sx, yy, sy, sw, bx0, bx1, bx2, by0,
        by1, by2, point_count, inlier_count, residual_sum, cost_sum,
        mag_sum, mag_min, total_count, bg_count
    };

    for (int k = 0; k < 20; ++k) {
        shared[tid] = vals[k];
        __syncthreads();
        for (int off = (int)blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) {
                if (k == 17) shared[tid] = fmin(shared[tid], shared[tid + off]);
                else shared[tid] += shared[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) block_out[(size_t)blockIdx.x * 20u + (size_t)k] = shared[0];
        __syncthreads();
    }
}
