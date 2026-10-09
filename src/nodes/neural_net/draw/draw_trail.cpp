#include "cuda_overlay_base.hpp"
#include "draw_trail_items.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_trail.ptx.h"

using namespace cuda_overlay;

class DrawTrail : public CudaOverlayBase {
    TrailItems items_;

    CUdeviceptr d_segments_ = 0;
    size_t d_segments_capacity_ = 0;

    const char* nodeName() const override { return "draw_trail"; }

    void onKernelsUnloaded() override {
        if (d_segments_) {
            cuMemFree(d_segments_);
            d_segments_ = 0;
            d_segments_capacity_ = 0;
        }
    }

    bool uploadSegments(const std::vector<LineSegment>& segments) {
        size_t bytes = segments.size() * sizeof(LineSegment);
        if (bytes == 0) return false;

        if (bytes > d_segments_capacity_) {
            if (d_segments_) cuMemFree(d_segments_);
            if (CUDA_OVERLAY_CHECK_CU(cuMemAlloc(&d_segments_, bytes))) {
                d_segments_ = 0;
                d_segments_capacity_ = 0;
                return false;
            }
            d_segments_capacity_ = bytes;
        }
        return CUDA_OVERLAY_CHECK_CU(
            cuMemcpyHtoDAsync(d_segments_, segments.data(), bytes, cuda_dev_ctx_->stream)) == 0;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_trail_ptx, avpl_draw_trail_ptx_len,
                         "kDrawTrailNV12Luma", "kDrawTrailNV12Chroma")) {
            throw Error("draw_trail: failed to initialize CUDA kernels");
        }

        FramePayloads payloads(input.raw());
        std::vector<LineSegment> segments;
        if (!items_.collect(payloads, geometry(), frame_counter_, segments)) return;

        if (!uploadSegments(segments)) {
            logstream << "draw_trail: failed to upload segments to GPU";
            return;
        }

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = ((unsigned int)output.width() + block_x - 1) / block_x;
        const unsigned int grid_y = ((unsigned int)output.height() + block_y - 1) / block_y;
        const int uv_width = (output.width() + 1) / 2;
        const int uv_height = (output.height() + 1) / 2;
        const unsigned int uv_grid_x = ((unsigned int)uv_width + block_x - 1) / block_x;
        const unsigned int uv_grid_y = ((unsigned int)uv_height + block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];
        int width = output.width();
        int height = output.height();
        CUdeviceptr seg_ptr = d_segments_;
        int num_segments = (int)segments.size();
        float thickness_sq = (float)(items_.thickness() * items_.thickness());
        int y_color = items_.color().y;
        int u_color = items_.color().u;
        int v_color = items_.color().v;

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&seg_ptr, (void*)&num_segments,
            (void*)&thickness_sq, (void*)&y_color
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_trail: failed launching luma kernel";
            return;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&seg_ptr, (void*)&num_segments,
            (void*)&thickness_sq, (void*)&u_color, (void*)&v_color
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_trail: failed launching chroma kernel";
            return;
        }

        CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
    }

public:
    using CudaOverlayBase::CudaOverlayBase;

    static std::shared_ptr<DrawTrail> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<DrawTrail>(src_edge->makeSource(), dst_edge->makeSink());

        UpstreamInfo info = resolveUpstreamInfo(src_edge, params);
        r->input_params_ = info.input_params;
        r->frame_rate_ = info.frame_rate;
        r->timebase_ = info.timebase;

        r->items_ = TrailItems::fromParams(params, "draw_trail");

        return r;
    }
};

DECLNODE(draw_trail, DrawTrail)
