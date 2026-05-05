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
