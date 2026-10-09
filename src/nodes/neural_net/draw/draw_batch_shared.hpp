#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace cuda_overlay {

struct BatchedBBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int thickness = 1;
    int y_color = 173;
    int u_color = 42;
    int v_color = 26;
};

struct BatchedTextLabel {
    int line1_offset = 0;
    int line2_offset = 0;
    int line3_offset = 0;
    int line1_len = 0;
    int line2_len = 0;
    int line3_len = 0;
    int origin_x = 0;
    int origin_y = 0;
    int font_scale = 1;
    int line_spacing = 0;
    int glyph_preset = 0;
    int bg_x = 0;
    int bg_y = 0;
    int bg_w = 0;
    int bg_h = 0;
    int draw_background = 1;
    float background_opacity = 1.0f;
    int text_y = 235;
    int text_u = 128;
    int text_v = 128;
    int bg_y_color = 16;
    int bg_u = 128;
    int bg_v = 128;
};

// A keypoint's centre in frame coordinates.
struct KeypointPos {
    float x;
    float y;
};

// A trail segment from (x0, y0) to (x1, y1) in frame coordinates.
struct LineSegment {
    int x0, y0, x1, y1;
};

// One item of the ml_debug pass. Items are painted in the order they are listed.
enum MlDebugKind : int { kMlDebugBox = 0, kMlDebugLabel = 1, kMlDebugDot = 2, kMlDebugSegment = 3 };

struct MlDebugItem {
    int kind = kMlDebugBox;
    // Box: x1, y1, x2, y2. Segment: x0, y0, x1, y1. Label: a is its index in the label list.
    int a = 0, b = 0, c = 0, d = 0;
    // Dot: centre. Others: unused.
    float fx = 0.f, fy = 0.f;
    // Box: border thickness. Dot: radius. Segment: thickness.
    int size = 1;
    int y_color = 173, u_color = 42, v_color = 26;
};

// A 16x16-pixel tile that at least one item touches: its top-left pixel and its run of item
// numbers in the tile index list.
struct MlDebugTile {
    int x = 0, y = 0;
    int first = 0, count = 0;
};

template <typename T>
class DeviceBuffer {
private:
    CUdeviceptr ptr_ = 0;
    size_t bytes_ = 0;

public:
    ~DeviceBuffer() = default;

    CUdeviceptr ptr() const { return ptr_; }
    size_t bytes() const { return bytes_; }

    void release(CUcontext ctx) {
        if (!ptr_) return;
        if (ctx) {
            cuCtxSetCurrent(ctx);
        }
        cuMemFree(ptr_);
        ptr_ = 0;
        bytes_ = 0;
    }

    bool ensureBytes(size_t required_bytes, CUcontext ctx) {
        if (required_bytes == 0) return true;
        if (ptr_ && bytes_ >= required_bytes) return true;
        release(ctx);
        if (ctx && cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return false;
        if (cuMemAlloc(&ptr_, required_bytes) != CUDA_SUCCESS) {
            ptr_ = 0;
            bytes_ = 0;
            return false;
        }
        bytes_ = required_bytes;
        return true;
    }

    bool upload(const std::vector<T>& items, CUcontext ctx, CUstream stream) {
        const size_t required_bytes = items.size() * sizeof(T);
        if (required_bytes == 0) return true;
        if (!ensureBytes(required_bytes, ctx)) return false;
        return cuMemcpyHtoDAsync(ptr_, items.data(), required_bytes, stream) == CUDA_SUCCESS;
    }

    bool uploadBytes(const void* src, size_t required_bytes, CUcontext ctx, CUstream stream) {
        if (required_bytes == 0) return true;
        if (!ensureBytes(required_bytes, ctx)) return false;
        return cuMemcpyHtoDAsync(ptr_, src, required_bytes, stream) == CUDA_SUCCESS;
    }
};

inline int appendTextBlob(std::vector<char>& blob, const char* text, int len) {
    if (!text || len <= 0) return 0;
    const int offset = (int)blob.size();
    blob.insert(blob.end(), text, text + len);
    return offset;
}

} // namespace cuda_overlay
