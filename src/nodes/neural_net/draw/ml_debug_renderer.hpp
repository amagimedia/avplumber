#pragma once
// The host side of the ml_debug pass: the frame's items in paint order, binned into the tiles they
// touch, staged in pinned memory and drawn with one kernel launch.
#include "cuda_overlay_base.hpp"
#include "draw_batch_shared.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cuda_overlay {

class MlDebugRenderer {
public:
    static constexpr int kTile = 16;      // pixels a side; the kernel's block is kTile/2 quads a side

private:
    struct Cover {      // pixels [x0, x1) x [y0, y1) of the frame that item `item` may change
        int item, x0, y0, x1, y1;
    };

    int width_ = 0, height_ = 0;
    std::vector<MlDebugItem> items_;
    std::vector<BatchedTextLabel> labels_;
    std::vector<char> text_;
    std::vector<Cover> cover_;

    // Per tile of the frame, kept between frames and reset only where a frame touched them.
    std::vector<int> stamp_, cursor_;
    std::vector<int> touched_;
    std::vector<MlDebugTile> tiles_;
    std::vector<int> tile_items_;

    void* host_ = nullptr;
    size_t host_bytes_ = 0;
    CUdeviceptr device_ = 0;
    size_t device_bytes_ = 0;

    void cover(int x0, int y0, int x1, int y1) {
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(width_, x1);
        y1 = std::min(height_, y1);
        if (x1 > x0 && y1 > y0) cover_.push_back({(int)items_.size(), x0, y0, x1, y1});
    }

    template <typename Visit>
    void forEachTile(const Cover& c, int tiles_x, Visit&& visit) const {
        for (int ty = c.y0 / kTile; ty <= (c.y1 - 1) / kTile; ++ty) {
            for (int tx = c.x0 / kTile; tx <= (c.x1 - 1) / kTile; ++tx) visit(ty * tiles_x + tx);
        }
    }

    /// Fills tiles_ and tile_items_: every tile an item covers lists the items covering it, in
    /// item order. A counting sort; the per-tile arrays are cleared again where they were used.
    void bin() {
        const int tiles_x = (width_ + kTile - 1) / kTile, tiles_y = (height_ + kTile - 1) / kTile;
        if ((int)stamp_.size() != tiles_x * tiles_y) {
            stamp_.assign((size_t)tiles_x * tiles_y, -1);
            cursor_.assign((size_t)tiles_x * tiles_y, 0);
        }
        touched_.clear();
        int total = 0;
        for (const Cover& c : cover_) {
            forEachTile(c, tiles_x, [&](int t) {
                if (stamp_[t] == c.item) return;      // another strip of the same item got here first
                stamp_[t] = c.item;
                if (cursor_[t]++ == 0) touched_.push_back(t);
                ++total;
            });
        }
        tiles_.clear();
        tiles_.reserve(touched_.size());
        int first = 0;
        for (int t : touched_) {
            tiles_.push_back({(t % tiles_x) * kTile, (t / tiles_x) * kTile, first, cursor_[t]});
            const int count = cursor_[t];
            cursor_[t] = first;
            first += count;
            stamp_[t] = -1;
        }
        tile_items_.resize((size_t)total);
        for (const Cover& c : cover_) {
            forEachTile(c, tiles_x, [&](int t) {
                if (stamp_[t] == c.item) return;
                stamp_[t] = c.item;
                tile_items_[(size_t)cursor_[t]++] = c.item;
            });
        }
        for (int t : touched_) {
            stamp_[t] = -1;
            cursor_[t] = 0;
        }
    }

    static size_t aligned(size_t bytes) { return (bytes + 15) & ~(size_t)15; }

public:
    MlDebugRenderer() = default;
    MlDebugRenderer(const MlDebugRenderer&) = delete;
    MlDebugRenderer& operator=(const MlDebugRenderer&) = delete;

    /// Frees the staging memory under `ctx`; the renderer can be used again afterwards.
    void release(CUcontext ctx) {
        if (!host_ && !device_) return;
        if (ctx) cuCtxSetCurrent(ctx);
        if (host_) cuMemFreeHost(host_);
        if (device_) cuMemFree(device_);
        host_ = nullptr;
        device_ = 0;
        host_bytes_ = device_bytes_ = 0;
    }

    /// Starts the item list of a frame of `width` x `height`.
    void begin(int width, int height) {
        width_ = width;
        height_ = height;
        items_.clear();
        labels_.clear();
        text_.clear();
        cover_.clear();
    }

    size_t itemCount() const { return items_.size(); }
    size_t tileCount() const { return tiles_.size(); }

    void addBox(const BatchedBBox& box) {
        MlDebugItem item;
        item.kind = kMlDebugBox;
        item.a = box.x1;
        item.b = box.y1;
        item.c = box.x2;
        item.d = box.y2;
        item.size = box.thickness;
        item.y_color = box.y_color;
        item.u_color = box.u_color;
        item.v_color = box.v_color;
        const int t = std::max(1, box.thickness);
        // Only the border is painted: four strips, which meet when the box is thinner than two borders.
        cover(box.x1, box.y1, box.x2, std::min(box.y2, box.y1 + t));
        cover(box.x1, std::max(box.y1, box.y2 - t), box.x2, box.y2);
        cover(box.x1, box.y1, std::min(box.x2, box.x1 + t), box.y2);
        cover(std::max(box.x1, box.x2 - t), box.y1, box.x2, box.y2);
        items_.push_back(item);
    }

    void addDot(const KeypointPos& point, int radius, const DrawColor& color) {
        MlDebugItem item;
        item.kind = kMlDebugDot;
        item.fx = point.x;
        item.fy = point.y;
        item.size = radius;
        item.y_color = color.y;
        item.u_color = color.u;
        item.v_color = color.v;
        const int r = std::abs(radius);
        cover((int)std::floor(point.x) - r, (int)std::floor(point.y) - r,
              (int)std::ceil(point.x) + r + 1, (int)std::ceil(point.y) + r + 1);
        items_.push_back(item);
    }

    void addSegment(const LineSegment& segment, int thickness, const DrawColor& color) {
        MlDebugItem item;
        item.kind = kMlDebugSegment;
        item.a = segment.x0;
        item.b = segment.y0;
        item.c = segment.x1;
        item.d = segment.y1;
        item.size = thickness;
        item.y_color = color.y;
        item.u_color = color.u;
        item.v_color = color.v;
        const int t = std::abs(thickness);
        // The rectangle around the capsule. A long diagonal covers tiles it does not paint; they
        // test the segment and leave the frame alone.
        cover(std::min(segment.x0, segment.x1) - t, std::min(segment.y0, segment.y1) - t,
              std::max(segment.x0, segment.x1) + t + 1, std::max(segment.y0, segment.y1) + t + 1);
        items_.push_back(item);
    }

    /// The label list and its text, for a label collector to append to; addLabels then lists them.
    std::vector<BatchedTextLabel>& labels() { return labels_; }
    std::vector<char>& text() { return text_; }

    /// Lists the labels from index `first` on as items, in their order.
    void addLabels(size_t first) {
        for (size_t i = first; i < labels_.size(); ++i) {
            const BatchedTextLabel& label = labels_[i];
            MlDebugItem item;
            item.kind = kMlDebugLabel;
            item.a = (int)i;
            cover(label.bg_x, label.bg_y, label.bg_x + label.bg_w, label.bg_y + label.bg_h);
            items_.push_back(item);
        }
    }

    /// Draws the listed items on the NV12 frame `out`: one upload, one launch of `kernel`, one wait
    /// for `stream`. Nothing is uploaded or launched when no item touches the frame.
    bool draw(CUcontext ctx, CUstream stream, CUfunction kernel, AVFrame* out, const char* node) {
        if (cover_.empty()) {
            tiles_.clear();
            return true;
        }
        bin();

        const size_t tiles_at = 0;
        const size_t tile_items_at = aligned(tiles_at + tiles_.size() * sizeof(MlDebugTile));
        const size_t items_at = aligned(tile_items_at + tile_items_.size() * sizeof(int));
        const size_t labels_at = aligned(items_at + items_.size() * sizeof(MlDebugItem));
        const size_t text_at = aligned(labels_at + labels_.size() * sizeof(BatchedTextLabel));
        const size_t bytes = aligned(text_at + text_.size());

        if (bytes > host_bytes_) {
            // Grown with room to spare: the list changes size with every detection.
            release(ctx);
            const size_t capacity = aligned(bytes * 2);
            if (CUDA_OVERLAY_CHECK_CU(cuMemHostAlloc(&host_, capacity, 0))) {
                host_ = nullptr;
                return false;
            }
            if (CUDA_OVERLAY_CHECK_CU(cuMemAlloc(&device_, capacity))) {
                cuMemFreeHost(host_);
                host_ = nullptr;
                device_ = 0;
                return false;
            }
            host_bytes_ = device_bytes_ = capacity;
        }
        char* staging = (char*)host_;
        std::memcpy(staging + tiles_at, tiles_.data(), tiles_.size() * sizeof(MlDebugTile));
        std::memcpy(staging + tile_items_at, tile_items_.data(), tile_items_.size() * sizeof(int));
        std::memcpy(staging + items_at, items_.data(), items_.size() * sizeof(MlDebugItem));
        if (!labels_.empty()) std::memcpy(staging + labels_at, labels_.data(), labels_.size() * sizeof(BatchedTextLabel));
        if (!text_.empty()) std::memcpy(staging + text_at, text_.data(), text_.size());
        if (CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(device_, host_, bytes, stream))) {
            logstream << node << ": failed uploading the draw list";
            return false;
        }

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)out->data[0];
        size_t pitch_y = (size_t)out->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)out->data[1];
        size_t pitch_uv = (size_t)out->linesize[1];
        int width = width_;
        int height = height_;
        CUdeviceptr tiles_ptr = device_ + tiles_at;
        CUdeviceptr tile_items_ptr = device_ + tile_items_at;
        CUdeviceptr items_ptr = device_ + items_at;
        CUdeviceptr labels_ptr = device_ + labels_at;
        CUdeviceptr text_ptr = device_ + text_at;
        void* args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&tiles_ptr, (void*)&tile_items_ptr, (void*)&items_ptr,
            (void*)&labels_ptr, (void*)&text_ptr
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(kernel,
                                    (unsigned int)tiles_.size(), 1, 1,
                                    kTile / 2, kTile / 2, 1,
                                    0, stream, args, nullptr))) {
            logstream << node << ": failed launching the draw kernel";
            return false;
        }
        // The frame goes to the next node complete, as every draw node hands it on.
        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(stream)) == 0;
    }
};

} // namespace cuda_overlay
