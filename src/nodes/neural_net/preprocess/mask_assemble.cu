// yolo_mask_assemble.cu
// For each detection: dot-product of 32 coefficients against 32 prototype
// channels at each spatial position, followed by sigmoid activation.
//
// Grid: (proto_w + 31) / 32, (proto_h + 7) / 8, num_detections
// Block: 32, 8, 1
//
// Inputs:
//   prototypes: [32, proto_h, proto_w] float
//   coefficients: [num_detections, 32] float
//   proto_h, proto_w: prototype spatial dims
//   num_coefficients: 32 (or configurable)
//
// Output:
//   masks: [num_detections, proto_h, proto_w] float (after sigmoid)

extern "C" __global__ void kMaskAssemble(
    const float* __restrict__ prototypes,  // [32, proto_h, proto_w]
    const float* __restrict__ coefficients, // [num_dets, 32]
    float* __restrict__ masks,              // [num_dets, proto_h, proto_w]
    int proto_h,
    int proto_w,
    int num_coefficients
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int det = blockIdx.z;

    if (x >= proto_w || y >= proto_h) return;

    const int spatial_idx = y * proto_w + x;
    const int spatial_size = proto_h * proto_w;

    float sum = 0.0f;
    for (int c = 0; c < num_coefficients; ++c) {
        sum += coefficients[det * num_coefficients + c] * prototypes[c * spatial_size + spatial_idx];
    }

    // Sigmoid
    sum = 1.0f / (1.0f + expf(-sum));

    masks[det * spatial_size + spatial_idx] = sum;
}

// Bilinear downsample kernel for CPU mask path
extern "C" __global__ void kMaskDownsample(
    const float* __restrict__ src,   // [proto_h, proto_w]
    float* __restrict__ dst,          // [dst_h, dst_w]
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    float src_x = (float)x * (float)src_w / (float)dst_w;
    float src_y = (float)y * (float)src_h / (float)dst_h;

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    float v = src[y0 * src_w + x0] * (1.0f - fx) * (1.0f - fy)
            + src[y0 * src_w + x1] * fx * (1.0f - fy)
            + src[y1 * src_w + x0] * (1.0f - fx) * fy
            + src[y1 * src_w + x1] * fx * fy;

    dst[y * dst_w + x] = v;
}
