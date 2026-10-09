#include "cuda_overlay_base.hpp"
#include "draw_label_items.hpp"

#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_text.ptx.h"

class DrawBBoxLabels : public CudaOverlayBase {
private:
    cuda_overlay::LabelItems items_;
    cuda_overlay::DeviceBuffer<cuda_overlay::BatchedTextLabel> d_labels_;
    cuda_overlay::DeviceBuffer<char> d_text_blob_;

    const char* nodeName() const override { return "draw_bbox_labels"; }

    void onKernelsUnloaded() override {
        d_labels_.release(cu_ctx_);
        d_text_blob_.release(cu_ctx_);
    }

    bool drawLabelsOnFrame(av::VideoFrame& frm, const std::vector<cuda_overlay::BatchedTextLabel>& batched_labels,
                           const std::vector<char>& text_blob) {
        if (batched_labels.empty()) return true;
        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = (unsigned int)(frm.width() + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(frm.height() + (int)block_y - 1) / block_y;
        const unsigned int uv_grid_x = (unsigned int)(((frm.width() + 1) / 2) + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(((frm.height() + 1) / 2) + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        size_t pitch_y = (size_t)frm.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        size_t pitch_uv = (size_t)frm.raw()->linesize[1];
        int width = frm.width();
        int height = frm.height();

        if (!d_labels_.upload(batched_labels, cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_bbox_labels: failed uploading label descriptors";
            return false;
        }
        if (!d_text_blob_.uploadBytes(text_blob.data(), text_blob.size(), cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_bbox_labels: failed uploading label text blob";
            return false;
        }
        CUdeviceptr labels_ptr = d_labels_.ptr();
        int num_labels = (int)batched_labels.size();
        CUdeviceptr text_blob_ptr = d_text_blob_.ptr();

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&labels_ptr, (void*)&num_labels,
            (void*)&text_blob_ptr
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            return false;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&labels_ptr, (void*)&num_labels,
            (void*)&text_blob_ptr
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            return false;
        }
        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_text_ptx, avpl_draw_text_ptx_len,
                         "kDrawTextNV12Luma", "kDrawTextNV12Chroma")) {
            throw Error("draw_bbox_labels: failed to initialize text kernels");
        }

        cuda_overlay::FramePayloads payloads(input.raw());
        std::vector<cuda_overlay::BatchedTextLabel> labels;
        std::vector<char> text_blob;
        items_.collect(payloads, geometry(), frame_counter_, labels, text_blob);
        if (!drawLabelsOnFrame(output, labels, text_blob)) {
            throw Error("draw_bbox_labels: failed to draw labels");
        }
    }

public:
    DrawBBoxLabels(std::unique_ptr<Source<av::VideoFrame>>&& source,
                   std::unique_ptr<Sink<av::VideoFrame>>&& sink,
                   cuda_overlay::LabelItems items,
                   VideoParameters input_params,
                   av::Rational frame_rate,
                   av::Rational timebase)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          items_(std::move(items)) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    static std::shared_ptr<DrawBBoxLabels> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawBBoxLabels>(
            edges, params, cuda_overlay::LabelItems::fromParams(params, "draw_bbox_labels"),
            upstream.input_params, upstream.frame_rate, upstream.timebase);
    }
};

DECLNODE(draw_bbox_labels, DrawBBoxLabels)
