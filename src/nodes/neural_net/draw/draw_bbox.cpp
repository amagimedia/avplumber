#include "cuda_overlay_base.hpp"
#include "draw_bbox_items.hpp"

#include <cstdint>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_bbox.ptx.h"

class DrawBBox : public CudaOverlayBase {
private:
    cuda_overlay::BBoxItems items_;
    cuda_overlay::DeviceBuffer<cuda_overlay::BatchedBBox> d_boxes_;

    const char* nodeName() const override { return "draw_bbox"; }

    void onKernelsUnloaded() override {
        d_boxes_.release(cu_ctx_);
    }

    bool drawBBoxesOnFrame(av::VideoFrame &frm, const std::vector<cuda_overlay::BatchedBBox> &batch) {
        if (batch.empty()) return true;

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = (unsigned int)(frm.width() + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(frm.height() + (int)block_y - 1) / block_y;
        const int uv_width = (frm.width() + 1) / 2;
        const int uv_height = (frm.height() + 1) / 2;
        const unsigned int uv_grid_x = (unsigned int)(uv_width + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(uv_height + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        size_t pitch_y = (size_t)frm.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        size_t pitch_uv = (size_t)frm.raw()->linesize[1];
        int width = frm.width();
        int height = frm.height();
        if (!d_boxes_.upload(batch, cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_bbox: failed uploading bbox batch";
            return false;
        }
        CUdeviceptr boxes_ptr = d_boxes_.ptr();
        int num_boxes = (int)batch.size();

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&boxes_ptr, (void*)&num_boxes
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_bbox: failed launching luma kernel";
            return false;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&boxes_ptr, (void*)&num_boxes
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_bbox: failed launching chroma kernel";
            return false;
        }

        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_bbox_ptx, avpl_draw_bbox_ptx_len,
                         "kDrawBBoxNV12Luma", "kDrawBBoxNV12Chroma")) {
            throw Error("draw_bbox: failed to initialize CUDA kernels");
        }

        cuda_overlay::FramePayloads payloads(input.raw());
        std::vector<cuda_overlay::BatchedBBox> boxes;
        items_.collect(payloads, geometry(), frame_counter_, boxes);
        if (!drawBBoxesOnFrame(output, boxes)) {
            throw Error("draw_bbox: failed drawing bbox batch");
        }
    }

public:
    DrawBBox(std::unique_ptr<Source<av::VideoFrame>> &&source,
             std::unique_ptr<Sink<av::VideoFrame>> &&sink,
             cuda_overlay::BBoxItems items,
             VideoParameters input_params,
             av::Rational frame_rate,
             av::Rational timebase)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          items_(std::move(items)) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    static std::shared_ptr<DrawBBox> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawBBox>(
            edges, params, cuda_overlay::BBoxItems::fromParams(params, "draw_bbox"),
            upstream.input_params, upstream.frame_rate, upstream.timebase);
    }
};

DECLNODE(draw_bbox, DrawBBox)
