#include "cuda_overlay_base.hpp"

#include <cstdint>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/draw_keypoints.ptx.h"

using cuda_overlay::DrawColor;

class DrawKeypoints : public CudaOverlayBase {
private:
    std::string metadata_key_;
    DrawColor color_{};
    int radius_ = 3;
    double min_conf_ = 0.0;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;
    int debug_log_every_n_ = 0;

    CUdeviceptr gpu_points_buf_ = 0;
    size_t gpu_points_capacity_ = 0;

    const char* nodeName() const override { return "draw_keypoints"; }

    bool remapModelCoord(double x, double y,
                         double model_w, double model_h,
                         double& out_x, double& out_y) const {
        if (model_content_width_ > 0.0 && model_content_height_ > 0.0) {
            const double content_x = std::max(0.0, std::min(x - model_content_offset_x_, model_content_width_));
            const double content_y = std::max(0.0, std::min(y - model_content_offset_y_, model_content_height_));
            out_x = content_x * ((double)input_params_.width / model_content_width_);
            out_y = content_y * ((double)input_params_.height / model_content_height_);
            return true;
        }
        const double sx = model_w > 0.0 ? (double)input_params_.width / model_w : 1.0;
        const double sy = model_h > 0.0 ? (double)input_params_.height / model_h : 1.0;
        out_x = x * sx;
        out_y = y * sy;
        return true;
    }

    struct KeypointPos {
        float x;
        float y;
    };

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_keypoints_ptx, avpl_draw_keypoints_ptx_len,
                         "kDrawKeypointsNV12Luma", "kDrawKeypointsNV12Chroma")) {
            throw Error("draw_keypoints: failed to initialize CUDA kernels");
        }

        const AVFrame* raw = input.raw();
        if (!raw || !raw->metadata) return;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return;

        // Parse pose metadata
        std::vector<KeypointPos> points;
        double model_w = 0, model_h = 0;
        try {
            Parameters md = Parameters::parse(entry->value);
            model_w = md.value("model_width", 0.0);
            model_h = md.value("model_height", 0.0);
            int num_kpts = md.value("num_keypoints", 0);
            if (num_kpts <= 0) return;

            if (!md.contains("poses") || !md["poses"].is_array()) return;

            for (const auto& pose : md["poses"]) {
                if (!pose.contains("keypoints") || !pose["keypoints"].is_array()) continue;
                const auto& kpts = pose["keypoints"];
                int n = (int)kpts.size() / 3;
                for (int k = 0; k < n; ++k) {
                    float kx = kpts[(size_t)(k * 3 + 0)].get<float>();
                    float ky = kpts[(size_t)(k * 3 + 1)].get<float>();
                    float kc = kpts[(size_t)(k * 3 + 2)].get<float>();
                    if ((double)kc < min_conf_) continue;

                    double fx, fy;
                    remapModelCoord((double)kx, (double)ky, model_w, model_h, fx, fy);
                    // Clamp to frame bounds
                    if (fx < 0 || fy < 0 || fx >= input_params_.width || fy >= input_params_.height) continue;
                    points.push_back({(float)fx, (float)fy});
                }
            }
        } catch (const std::exception&) {
            return;
        }

        if (points.empty()) return;

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_keypoints: frame=" << frame_counter_
                      << " points=" << points.size();
        }

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
        int radius = radius_;
        int frame_w = output.width();
        int frame_h = output.height();
        int y_color = color_.y;
        int u_color = color_.u;
        int v_color = color_.v;

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
                  std::string metadata_key,
                  DrawColor color,
                  int radius,
                  double min_conf,
                  double model_content_width,
                  double model_content_height,
                  double model_content_offset_x,
                  double model_content_offset_y,
                  VideoParameters input_params,
                  av::Rational frame_rate,
                  av::Rational timebase,
                  int debug_log_every_n)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          color_(color),
          radius_(radius),
          min_conf_(min_conf),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y),
          debug_log_every_n_(debug_log_every_n) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    ~DrawKeypoints() {
        if (gpu_points_buf_) {
            cuMemFree(gpu_points_buf_);
            gpu_points_buf_ = 0;
        }
    }

    static std::shared_ptr<DrawKeypoints> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        const std::string metadata_key = params.value("metadata_key", std::string("yolo_pose"));
        const int radius = params.value("radius", 3);
        const double min_conf = params.value("min_conf", 0.0);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);

        DrawColor color{};
        std::string color_name = params.value("color", std::string("white"));
        if (!cuda_overlay::tryParseNamedColor(color_name, color)) {
            throw Error("draw_keypoints: unknown color: " + color_name);
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawKeypoints>(
            edges, params, metadata_key, color, radius, min_conf,
            model_content_width, model_content_height, model_content_offset_x, model_content_offset_y,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_keypoints, DrawKeypoints)
