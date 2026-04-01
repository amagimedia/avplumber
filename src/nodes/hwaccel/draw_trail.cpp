#include "cuda_overlay_base.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/draw_trail.ptx.h"

using namespace cuda_overlay;

namespace {

struct LineSegment {
    int x0, y0, x1, y1;
};

} // namespace

class DrawTrail : public CudaOverlayBase {
    std::string metadata_key_ = "yolo_detections";
    DrawColor color_;
    int thickness_ = 2;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;
    int debug_log_every_n_ = 0;

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

    bool remapCoord(double x, double y, double model_w, double model_h,
                    double& out_x, double& out_y) const {
        if (model_content_width_ > 0.0 && model_content_height_ > 0.0) {
            const double cx = std::max(0.0, std::min(x - model_content_offset_x_, model_content_width_));
            const double cy = std::max(0.0, std::min(y - model_content_offset_y_, model_content_height_));
            out_x = cx * ((double)input_params_.width / model_content_width_);
            out_y = cy * ((double)input_params_.height / model_content_height_);
            return true;
        }
        const double sx = model_w > 0.0 ? (double)input_params_.width / model_w : 1.0;
        const double sy = model_h > 0.0 ? (double)input_params_.height / model_h : 1.0;
        out_x = x * sx;
        out_y = y * sy;
        return true;
    }

    bool parseTrail(const av::VideoFrame& frm, std::vector<LineSegment>& segments_out) const {
        segments_out.clear();
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            if (!md.contains("trail") || !md["trail"].is_array()) return false;

            const double model_w = md.value("model_width", (double)frm.width());
            const double model_h = md.value("model_height", (double)frm.height());

            const auto& trail = md["trail"];
            if (trail.size() < 2) return false;

            double prev_x = 0.0, prev_y = 0.0;
            bool have_prev = false;

            for (const auto& pt : trail) {
                if (!pt.is_array() || pt.size() < 2) continue;
                double mx = pt[0].get<double>();
                double my = pt[1].get<double>();
                double fx = 0.0, fy = 0.0;
                remapCoord(mx, my, model_w, model_h, fx, fy);

                if (have_prev) {
                    LineSegment seg;
                    seg.x0 = (int)std::round(prev_x);
                    seg.y0 = (int)std::round(prev_y);
                    seg.x1 = (int)std::round(fx);
                    seg.y1 = (int)std::round(fy);
                    segments_out.push_back(seg);
                }
                prev_x = fx;
                prev_y = fy;
                have_prev = true;
            }
            return !segments_out.empty();
        } catch (...) {
            return false;
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

        std::vector<LineSegment> segments;
        if (!parseTrail(input, segments)) {
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "draw_trail: frame=" << frame_counter_ << " no trail data";
            }
            return;
        }

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
        float thickness_sq = (float)(thickness_ * thickness_);
        int y_color = color_.y;
        int u_color = color_.u;
        int v_color = color_.v;

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

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_trail: frame=" << frame_counter_ << " segments=" << num_segments;
        }
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

        r->metadata_key_ = params.value("metadata_key", std::string("yolo_detections"));
        r->thickness_ = params.value("thickness", 2);

        std::string color_name = params.value("color", std::string("red"));
        if (!tryParseNamedColor(color_name, r->color_)) {
            r->color_ = DrawColor{81, 90, 240}; // red fallback
        }

        r->model_content_width_ = params.value("model_content_width", 0.0);
        r->model_content_height_ = params.value("model_content_height", 0.0);
        r->model_content_offset_x_ = params.value("model_content_offset_x", 0.0);
        r->model_content_offset_y_ = params.value("model_content_offset_y", 0.0);
        r->debug_log_every_n_ = params.value("debug_log_every_n", 0);

        return r;
    }
};

DECLNODE(draw_trail, DrawTrail)
