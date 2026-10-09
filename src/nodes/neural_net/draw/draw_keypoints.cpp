#include "cuda_overlay_base.hpp"
#include "draw_keypoint_items.hpp"

#include <cstdint>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_keypoints.ptx.h"

using cuda_overlay::KeypointPos;

class DrawKeypoints : public CudaOverlayBase {
private:
    cuda_overlay::KeypointItems items_;

    CUdeviceptr gpu_points_buf_ = 0;
    size_t gpu_points_capacity_ = 0;

    const char* nodeName() const override { return "draw_keypoints"; }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_keypoints_ptx, avpl_draw_keypoints_ptx_len,
                         "kDrawKeypointsNV12Luma", "kDrawKeypointsNV12Chroma")) {
            throw Error("draw_keypoints: failed to initialize CUDA kernels");
        }

        cuda_overlay::FramePayloads payloads(input.raw());
        std::vector<KeypointPos> points;
        items_.collect(payloads, geometry(), frame_counter_, points);

        if (points.empty()) return;

        // Upload points to GPU
        size_t pts_bytes = points.size() * sizeof(KeypointPos);
        if (pts_bytes > gpu_points_capacity_) {
            if (gpu_points_buf_) cuMemFree(gpu_points_buf_);
            gpu_points_capacity_ = pts_bytes * 2;  // over-allocate
            if (CUDA_OVERLAY_CHECK_CU(cuMemAlloc(&gpu_points_buf_, gpu_points_capacity_))) {
                gpu_points_buf_ = 0;
                gpu_points_capacity_ = 0;
                return;
            }
        }
        if (CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(gpu_points_buf_, points.data(), pts_bytes, cuda_dev_ctx_->stream)))
            return;

        int num_points = (int)points.size();
        int radius = items_.radius();
        int frame_w = output.width();
        int frame_h = output.height();
        int y_color = items_.color().y;
        int u_color = items_.color().u;
        int v_color = items_.color().v;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        unsigned int grid_x = ((unsigned int)frame_w + block_x - 1) / block_x;
        unsigned int grid_y = ((unsigned int)frame_h + block_y - 1) / block_y;

        // Luma kernel
        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&gpu_points_buf_, (void*)&num_points,
            (void*)&radius, (void*)&y_color,
            (void*)&frame_w, (void*)&frame_h
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_keypoints: failed launching luma kernel";
            return;
        }

        // Chroma kernel
        unsigned int uv_grid_x = (((unsigned int)frame_w + 1) / 2 + block_x - 1) / block_x;
        unsigned int uv_grid_y = (((unsigned int)frame_h + 1) / 2 + block_y - 1) / block_y;
        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&gpu_points_buf_, (void*)&num_points,
            (void*)&radius, (void*)&u_color, (void*)&v_color,
            (void*)&frame_w, (void*)&frame_h
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_keypoints: failed launching chroma kernel";
            return;
        }

        CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
    }

public:
    DrawKeypoints(std::unique_ptr<Source<av::VideoFrame>> &&source,
                  std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                  cuda_overlay::KeypointItems items,
                  VideoParameters input_params,
                  av::Rational frame_rate,
                  av::Rational timebase)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          items_(std::move(items)) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    ~DrawKeypoints() {
        if (gpu_points_buf_) {
            // The destroying thread may have no current context (as ~CudaOverlayBase does).
            if (cu_ctx_) CUDA_OVERLAY_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
            cuMemFree(gpu_points_buf_);
            gpu_points_buf_ = 0;
        }
    }

    static std::shared_ptr<DrawKeypoints> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawKeypoints>(
            edges, params, cuda_overlay::KeypointItems::fromParams(params, "draw_keypoints"),
            upstream.input_params, upstream.frame_rate, upstream.timebase);
    }
};

DECLNODE(draw_keypoints, DrawKeypoints)
