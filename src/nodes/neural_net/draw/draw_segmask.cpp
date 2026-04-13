#include "cuda_overlay_base.hpp"
#include "../common/infer_trt_base.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_segmask.ptx.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

using cuda_overlay::DrawColor;

class DrawSegMask : public CudaOverlayBase {
private:
    struct MaskDetection {
        int bbox_x1, bbox_y1, bbox_x2, bbox_y2;
        int mask_index;
        int cls = -1;
    };

    std::string metadata_key_;
    std::string shot_metadata_key_ = "shot_info";
    DrawColor mask_color_{};
    float opacity_ = 0.5f;
    float threshold_ = 0.5f;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;
    int debug_log_every_n_ = 0;
    bool require_wide_shot_ = false;
    double min_conf_ = 0.0;
    std::unordered_map<int, DrawColor> class_colors_;

    const char* nodeName() const override { return "draw_segmask"; }

    static int clampInt(int value, int lo, int hi) {
        return std::max(lo, std::min(hi, value));
    }

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

    DrawColor resolveColor(int cls) const {
        auto it = class_colors_.find(cls);
        if (it != class_colors_.end()) return it->second;
        return mask_color_;
    }

    std::string readShotType(const av::VideoFrame& frm) const {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return {};

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, shot_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};

        try {
            Parameters md = Parameters::parse(entry->value);
            return md.value("shot_type", std::string());
        } catch (const std::exception&) {
            return {};
        }
    }

    void renderMasks(av::VideoFrame& output,
                     CUdeviceptr gpu_masks,
                     const GpuMaskSideDataHeader& header,
                     const std::vector<MaskDetection>& detections) {
        const int num_masks = (int)header.num_masks;
        const int proto_w = (int)header.proto_w;
        const int proto_h = (int)header.proto_h;
        const int model_w = (int)header.model_w;
        const int model_h = (int)header.model_h;

        if (num_masks <= 0 || proto_w <= 0 || proto_h <= 0 || !gpu_masks || detections.empty()) return;

        // Compute frame->model coordinate mapping
        // frame_x * scale_x + offset_x = model_x
        float scale_x, scale_y, offset_x, offset_y;
        if (model_content_width_ > 0.0 && model_content_height_ > 0.0) {
            scale_x = (float)(model_content_width_ / (double)input_params_.width);
            scale_y = (float)(model_content_height_ / (double)input_params_.height);
            offset_x = (float)model_content_offset_x_;
            offset_y = (float)model_content_offset_y_;
        } else {
            scale_x = (float)model_w / (float)input_params_.width;
            scale_y = (float)model_h / (float)input_params_.height;
            offset_x = 0.0f;
            offset_y = 0.0f;
        }

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];
        int frame_w = output.width();
        int frame_h = output.height();
        float op = opacity_;
        float thresh = threshold_;

        for (const auto& det : detections) {
            if (det.mask_index >= num_masks) continue;

            const DrawColor color = resolveColor(det.cls);
            int y_color = color.y;
            int u_color = color.u;
            int v_color = color.v;

            CUdeviceptr mask_ptr = gpu_masks + (size_t)det.mask_index * proto_h * proto_w * sizeof(float);

            int bx1 = det.bbox_x1, by1 = det.bbox_y1;
            int bx2 = det.bbox_x2, by2 = det.bbox_y2;
            int bbox_w = bx2 - bx1;
            int bbox_h = by2 - by1;

            unsigned int grid_x = ((unsigned int)bbox_w + block_x - 1) / block_x;
            unsigned int grid_y = ((unsigned int)bbox_h + block_y - 1) / block_y;

            void* y_args[] = {
                (void*)&y_plane, (void*)&pitch_y,
                (void*)&mask_ptr,
                (void*)&proto_w, (void*)&proto_h,
                (void*)&model_w, (void*)&model_h,
                (void*)&bx1, (void*)&by1, (void*)&bx2, (void*)&by2,
                (void*)&scale_x, (void*)&scale_y, (void*)&offset_x, (void*)&offset_y,
                (void*)&frame_w, (void*)&frame_h,
                (void*)&y_color, (void*)&op, (void*)&thresh
            };
            if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                        grid_x, grid_y, 1,
                                        block_x, block_y, 1,
                                        0, cuda_dev_ctx_->stream, y_args, nullptr))) {
                logstream << "draw_segmask: failed launching luma kernel";
                return;
            }

            int uv_bbox_w = ((bx2 + 1) >> 1) - (bx1 >> 1);
            int uv_bbox_h = ((by2 + 1) >> 1) - (by1 >> 1);
            unsigned int uv_grid_x = ((unsigned int)uv_bbox_w + block_x - 1) / block_x;
            unsigned int uv_grid_y = ((unsigned int)uv_bbox_h + block_y - 1) / block_y;

            void* uv_args[] = {
                (void*)&uv_plane, (void*)&pitch_uv,
                (void*)&mask_ptr,
                (void*)&proto_w, (void*)&proto_h,
                (void*)&model_w, (void*)&model_h,
                (void*)&bx1, (void*)&by1, (void*)&bx2, (void*)&by2,
                (void*)&scale_x, (void*)&scale_y, (void*)&offset_x, (void*)&offset_y,
                (void*)&frame_w, (void*)&frame_h,
                (void*)&u_color, (void*)&v_color, (void*)&op, (void*)&thresh
            };
            if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                        uv_grid_x, uv_grid_y, 1,
                                        block_x, block_y, 1,
                                        0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
                logstream << "draw_segmask: failed launching chroma kernel";
                return;
            }
        }

        CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
    }

    std::vector<MaskDetection> parseDetections(const av::VideoFrame& frm, int model_w, int model_h) const {
        std::vector<MaskDetection> result;
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return result;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return result;

        try {
            Parameters md = Parameters::parse(entry->value);
            if (!md.contains("detections") || !md["detections"].is_array()) return result;

            int mask_idx = 0;
            for (const auto& det : md["detections"]) {
                if (!det.is_object()) continue;
                if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) {
                    ++mask_idx;
                    continue;
                }

                const double conf = det.value("conf", 0.0);
                if (conf < min_conf_) {
                    ++mask_idx;
                    continue;
                }

                const auto& xyxy = det["xyxy"];
                double x1 = xyxy[0].get<double>();
                double y1 = xyxy[1].get<double>();
                double x2 = xyxy[2].get<double>();
                double y2 = xyxy[3].get<double>();

                // Remap from model space to frame space
                double fx1, fy1, fx2, fy2;
                remapModelCoord(x1, y1, (double)model_w, (double)model_h, fx1, fy1);
                remapModelCoord(x2, y2, (double)model_w, (double)model_h, fx2, fy2);

                MaskDetection md_out;
                md_out.bbox_x1 = clampInt((int)std::lround(fx1), 0, input_params_.width);
                md_out.bbox_y1 = clampInt((int)std::lround(fy1), 0, input_params_.height);
                md_out.bbox_x2 = clampInt((int)std::lround(fx2), 0, input_params_.width);
                md_out.bbox_y2 = clampInt((int)std::lround(fy2), 0, input_params_.height);
                if (md_out.bbox_x2 <= md_out.bbox_x1 || md_out.bbox_y2 <= md_out.bbox_y1) {
                    ++mask_idx;
                    continue;
                }
                md_out.mask_index = mask_idx;
                md_out.cls = det.value("cls", -1);
                result.push_back(md_out);
                ++mask_idx;
            }
        } catch (const std::exception&) {}
        return result;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_segmask_ptx, avpl_draw_segmask_ptx_len,
                         "kDrawSegMaskNV12Luma", "kDrawSegMaskNV12Chroma")) {
            throw Error("draw_segmask: failed to initialize CUDA kernels");
        }

        const AVFrameSideData* sd = av_frame_get_side_data(input.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS_GPU);
        const std::string shot_type = readShotType(input);
        if (require_wide_shot_ && shot_type != "wide") {
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "draw_segmask: frame=" << frame_counter_
                          << " suppressed shot_type=" << (shot_type.empty() ? "<missing>" : shot_type);
            }
            return;
        }

        if (!sd || !sd->buf || sd->buf->size < (int)sizeof(GpuMaskSideDataHeader)) {
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "draw_segmask: frame=" << frame_counter_ << " no GPU mask side data";
            }
            return;
        }

        const auto* header = (const GpuMaskSideDataHeader*)sd->buf->data;
        const CUdeviceptr gpu_masks = (CUdeviceptr)header->gpu_ptr;
        const int num_masks = (int)header->num_masks;
        const int proto_w = (int)header->proto_w;
        const int proto_h = (int)header->proto_h;
        const int model_w = (int)header->model_w;
        const int model_h = (int)header->model_h;

        if (num_masks <= 0 || proto_w <= 0 || proto_h <= 0 || !gpu_masks) return;

        auto detections = parseDetections(input, model_w, model_h);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_segmask: frame=" << frame_counter_
                      << " masks=" << num_masks
                      << " proto=" << proto_w << "x" << proto_h
                      << " detections=" << detections.size();
        }
        renderMasks(output, gpu_masks, *header, detections);
    }

public:
    DrawSegMask(std::unique_ptr<Source<av::VideoFrame>> &&source,
                std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                std::string metadata_key,
                std::string shot_metadata_key,
                DrawColor mask_color,
                float opacity,
                float threshold,
                bool require_wide_shot,
                double min_conf,
                std::unordered_map<int, DrawColor> class_colors,
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
          shot_metadata_key_(std::move(shot_metadata_key)),
          mask_color_(mask_color),
          opacity_(opacity),
          threshold_(threshold),
          require_wide_shot_(require_wide_shot),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y),
          debug_log_every_n_(debug_log_every_n),
          min_conf_(min_conf),
          class_colors_(std::move(class_colors)) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    static std::shared_ptr<DrawSegMask> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        const std::string metadata_key = params.value("metadata_key", std::string("yolo_detections"));
        const std::string shot_metadata_key = params.value("shot_metadata_key", std::string("shot_info"));
        const float opacity = params.value("opacity", 0.5f);
        const float threshold = params.value("threshold", 0.5f);
        const bool require_wide_shot = params.value("require_wide_shot", false);
        const double min_conf = params.value("min_conf", 0.0);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);

        DrawColor mask_color{};
        std::string color_name = params.value("mask_color", std::string("green"));
        if (!cuda_overlay::tryParseNamedColor(color_name, mask_color)) {
            throw Error("draw_segmask: unknown mask_color: " + color_name);
        }

        std::unordered_map<int, DrawColor> class_colors;
        if (params.count("class_colors") && params["class_colors"].is_object()) {
            for (auto it = params["class_colors"].begin(); it != params["class_colors"].end(); ++it) {
                DrawColor color;
                if (!it.value().is_string() || !cuda_overlay::tryParseNamedColor(it.value().get<std::string>(), color)) {
                    throw Error("draw_segmask: class_colors values must be named colors (red, green, light_blue, yellow, white, black)");
                }
                class_colors[std::stoi(it.key())] = color;
            }
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawSegMask>(
            edges, params, metadata_key, shot_metadata_key, mask_color, opacity, threshold,
            require_wide_shot, min_conf,
            std::move(class_colors),
            model_content_width, model_content_height, model_content_offset_x, model_content_offset_y,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_segmask, DrawSegMask)
