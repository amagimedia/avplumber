#include "cuda_overlay_base.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/draw_bbox.ptx.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

using cuda_overlay::DrawColor;

namespace {
struct BBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int y_color = 173;
    int u_color = 42;
    int v_color = 26;
};
}

class DrawBBox : public CudaOverlayBase {
private:
    std::vector<std::string> metadata_keys_;
    bool have_last_viewport_crop_ = false;
    int last_viewport_crop_x_ = 0;
    int last_viewport_crop_y_ = 0;
    int last_viewport_input_w_ = 0;
    int last_viewport_input_h_ = 0;
    int last_viewport_dst_w_ = 0;
    int last_viewport_dst_h_ = 0;
    int bbox_thickness_ = 2;
    int debug_log_every_n_ = 0;
    double min_conf_ = 0.0;
    std::unordered_set<int> allowed_classes_;
    std::unordered_set<std::string> allowed_labels_;
    DrawColor default_color_{};
    std::unordered_map<int, DrawColor> model_colors_;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;

    const char* nodeName() const override { return "draw_bbox"; }

    static int clampInt(int value, int lo, int hi) {
        return std::max(lo, std::min(hi, value));
    }

    DrawColor resolveModelColor(const Parameters& det) const {
        if (det.contains("model_index")) {
            const int model_index = det["model_index"].get<int>();
            const auto it = model_colors_.find(model_index);
            if (it != model_colors_.end()) {
                return it->second;
            }
        }
        return default_color_;
    }

    bool scaleAndClampBBox(double x1, double y1, double x2, double y2, BBox &bbox_out) const {
        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
            return false;
        }

        bbox_out.x1 = clampInt((int)std::lround(x1), 0, input_params_.width);
        bbox_out.y1 = clampInt((int)std::lround(y1), 0, input_params_.height);
        bbox_out.x2 = clampInt((int)std::lround(x2), 0, input_params_.width);
        bbox_out.y2 = clampInt((int)std::lround(y2), 0, input_params_.height);
        if (bbox_out.x2 < bbox_out.x1) std::swap(bbox_out.x1, bbox_out.x2);
        if (bbox_out.y2 < bbox_out.y1) std::swap(bbox_out.y1, bbox_out.y2);
        return bbox_out.x2 > bbox_out.x1 && bbox_out.y2 > bbox_out.y1;
    }

    int viewportChromaXAlign() const {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(input_params_.realPixelFormat().get());
        if (!desc || desc->log2_chroma_w < 0) return 1;
        return 1 << desc->log2_chroma_w;
    }

    int viewportChromaYAlign() const {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(input_params_.realPixelFormat().get());
        if (!desc || desc->log2_chroma_h < 0) return 1;
        return 1 << desc->log2_chroma_h;
    }

    static int alignViewportCropCoord(int value, int align) {
        if (align <= 1) return value;
        return value & ~(align - 1);
    }

    int clampViewportCropX(int x, int dst_w) const {
        const int max_x = std::max(0, input_params_.width - dst_w);
        const int clamped = std::max(0, std::min(x, max_x));
        return alignViewportCropCoord(clamped, viewportChromaXAlign());
    }

    int clampViewportCropY(int y, int dst_h) const {
        const int max_y = std::max(0, input_params_.height - dst_h);
        const int clamped = std::max(0, std::min(y, max_y));
        return alignViewportCropCoord(clamped, viewportChromaYAlign());
    }

    std::pair<int, int> centerViewportCrop(int dst_w, int dst_h) const {
        return {
            clampViewportCropX((input_params_.width - dst_w) / 2, dst_w),
            clampViewportCropY((input_params_.height - dst_h) / 2, dst_h)
        };
    }

    static bool metadataHasViewportDstDims(const Parameters &md) {
        return md.contains("viewport_dst_width") && md["viewport_dst_width"].is_number()
            && md.contains("viewport_dst_height") && md["viewport_dst_height"].is_number();
    }

    // Same interpretation as crop_metadata_cuda::parseCropPosition (metadata + crop size).
    bool parseViewportCropPositionFromMd(const Parameters &md, int dst_w, int dst_h, int &x_out, int &y_out) const {
        double center_x = NAN;
        double center_y = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
            const auto &bbox = md["viewport_bbox"];
            const double x1 = bbox[0].get<double>();
            const double y1 = bbox[1].get<double>();
            const double x2 = bbox[2].get<double>();
            const double y2 = bbox[3].get<double>();
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else if (md.contains("viewport_center_x")) {
            center_x = md["viewport_center_x"].get<double>();
            center_y = input_params_.height * 0.5;
        } else if (md.contains("bbox_norm") && md["bbox_norm"].is_array() && md["bbox_norm"].size() >= 4) {
            const auto &bbox = md["bbox_norm"];
            const double fw = md.value("full_frame_width", input_params_.width);
            const double fh = md.value("full_frame_height", input_params_.height);
            const double x1 = bbox[0].get<double>() * fw;
            const double y1 = bbox[1].get<double>() * fh;
            const double x2 = bbox[2].get<double>() * fw;
            const double y2 = bbox[3].get<double>() * fh;
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else {
            return false;
        }

        if (!std::isfinite(center_x) || !std::isfinite(center_y)) return false;

        x_out = clampViewportCropX((int)std::lround(center_x - (double)dst_w * 0.5), dst_w);
        y_out = clampViewportCropY((int)std::lround(center_y - (double)dst_h * 0.5), dst_h);
        return true;
    }

    bool tryParseViewportCropFromFrame(const av::VideoFrame &frm, int dst_w, int dst_h, int &x_out, int &y_out) const {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        for (const std::string &key : metadata_keys_) {
            AVDictionaryEntry *entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
            if (!entry || !entry->value) continue;
            try {
                Parameters md = Parameters::parse(entry->value);
                if (parseViewportCropPositionFromMd(md, dst_w, dst_h, x_out, y_out)) return true;
            } catch (const std::exception &) {
                continue;
            }
        }
        return false;
    }

    bool tryReadViewportDstDims(const av::VideoFrame &frm, int &w_out, int &h_out) const {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        for (const std::string &key : metadata_keys_) {
            AVDictionaryEntry *entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
            if (!entry || !entry->value) continue;
            try {
                Parameters md = Parameters::parse(entry->value);
                if (!metadataHasViewportDstDims(md)) continue;
                const int w = md["viewport_dst_width"].get<int>();
                const int h = md["viewport_dst_height"].get<int>();
                if (w <= 0 || h <= 0) continue;
                if ((w & 1) || (h & 1)) continue;
                if (w > input_params_.width || h > input_params_.height) continue;
                w_out = w;
                h_out = h;
                return true;
            } catch (const std::exception &) {
                continue;
            }
        }
        return false;
    }

    void updateViewportCropPosition(const av::VideoFrame &frm, int dst_w, int dst_h, int &x_out, int &y_out) {
        int next_x = 0;
        int next_y = 0;
        const bool parsed = tryParseViewportCropFromFrame(frm, dst_w, dst_h, next_x, next_y);
        if (!parsed) {
            if (have_last_viewport_crop_) {
                next_x = last_viewport_crop_x_;
                next_y = last_viewport_crop_y_;
            } else {
                std::tie(next_x, next_y) = centerViewportCrop(dst_w, dst_h);
            }
        }
        last_viewport_crop_x_ = next_x;
        last_viewport_crop_y_ = next_y;
        have_last_viewport_crop_ = true;
        x_out = next_x;
        y_out = next_y;
    }

    bool parseSingleBBoxMetadata(const Parameters &md, BBox &bbox_out) const {
        double x1 = NAN;
        double y1 = NAN;
        double x2 = NAN;
        double y2 = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
            if (metadataHasViewportDstDims(md)) {
                return false;
            }
            const auto &bbox = md["viewport_bbox"];
            const double fw = md.value("full_frame_width", (double)input_params_.width);
            const double fh = md.value("full_frame_height", (double)input_params_.height);
            const double sx = fw > 0.0 ? (double)input_params_.width / fw : 1.0;
            const double sy = fh > 0.0 ? (double)input_params_.height / fh : 1.0;
            x1 = bbox[0].get<double>() * sx;
            y1 = bbox[1].get<double>() * sy;
            x2 = bbox[2].get<double>() * sx;
            y2 = bbox[3].get<double>() * sy;
        } else if (md.contains("bbox_norm") && md["bbox_norm"].is_array() && md["bbox_norm"].size() >= 4) {
            const auto &bbox = md["bbox_norm"];
            x1 = bbox[0].get<double>() * (double)input_params_.width;
            y1 = bbox[1].get<double>() * (double)input_params_.height;
            x2 = bbox[2].get<double>() * (double)input_params_.width;
            y2 = bbox[3].get<double>() * (double)input_params_.height;
        } else {
            return false;
        }

        return scaleAndClampBBox(x1, y1, x2, y2, bbox_out);
    }

    bool yoloDetectionAllowed(const Parameters& det) const {
        const double conf = det.value("conf", 0.0);
        if (conf < min_conf_) return false;

        if (allowed_classes_.empty() && allowed_labels_.empty()) {
            return true;
        }

        bool class_match = false;
        bool label_match = false;
        if (!allowed_classes_.empty() && det.contains("cls")) {
            class_match = allowed_classes_.count(det["cls"].get<int>()) > 0;
        }
        if (!allowed_labels_.empty() && det.contains("label") && det["label"].is_string()) {
            label_match = allowed_labels_.count(det["label"].get<std::string>()) > 0;
        }
        return class_match || label_match;
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

    // Appends to boxes_out. Returns true if at least one bbox was added from this JSON blob.
    bool parseMetadataBlob(const Parameters& md, std::vector<BBox>& boxes_out) const {
        BBox single_bbox;
        if (parseSingleBBoxMetadata(md, single_bbox)) {
            boxes_out.push_back(single_bbox);
            return true;
        }
        const size_t before = boxes_out.size();
        parseYoloDetections(md, boxes_out);
        return boxes_out.size() > before;
    }

    void parseYoloDetections(const Parameters& md, std::vector<BBox>& boxes_out) const {
        if (!md.contains("detections") || !md["detections"].is_array()) return;

        const std::string coord_space = md.value("coord_space", std::string("model"));
        const double model_w = md.value("model_width", (double)input_params_.width);
        const double model_h = md.value("model_height", (double)input_params_.height);

        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (!yoloDetectionAllowed(det)) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            const auto& xyxy = det["xyxy"];
            double x1 = xyxy[0].get<double>();
            double y1 = xyxy[1].get<double>();
            double x2 = xyxy[2].get<double>();
            double y2 = xyxy[3].get<double>();
            if (coord_space == "model") {
                if (!remapModelCoord(x1, y1, model_w, model_h, x1, y1)) continue;
                if (!remapModelCoord(x2, y2, model_w, model_h, x2, y2)) continue;
            }
            BBox bbox;
            if (scaleAndClampBBox(x1, y1, x2, y2, bbox)) {
                const DrawColor color = resolveModelColor(det);
                bbox.y_color = color.y;
                bbox.u_color = color.u;
                bbox.v_color = color.v;
                boxes_out.push_back(bbox);
            }
        }
    }

    bool parseBBoxes(const av::VideoFrame &frm, std::vector<BBox> &boxes_out) const {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata)
            return false;

        boxes_out.clear();
        bool any = false;
        for (const std::string &key : metadata_keys_) {
            AVDictionaryEntry *entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
            if (!entry || !entry->value)
                continue;
            try {
                Parameters md = Parameters::parse(entry->value);
                if (parseMetadataBlob(md, boxes_out))
                    any = true;
            } catch (const std::exception &) {
                continue;
            }
        }
        return any && !boxes_out.empty();
    }

    bool drawBBoxOnFrame(av::VideoFrame &frm, const BBox &bbox) {
        if (bbox_thickness_ <= 0) return true;

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
        int x1 = bbox.x1;
        int y1 = bbox.y1;
        int x2 = bbox.x2;
        int y2 = bbox.y2;
        int thickness = bbox_thickness_;
        int y_color = bbox.y_color;
        int u_color = bbox.u_color;
        int v_color = bbox.v_color;

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&x1, (void*)&y1, (void*)&x2, (void*)&y2,
            (void*)&thickness, (void*)&y_color
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
            (void*)&x1, (void*)&y1, (void*)&x2, (void*)&y2,
            (void*)&thickness, (void*)&u_color, (void*)&v_color
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

    void maybeLogFrame(const std::vector<BBox>& boxes) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        if (boxes.empty()) {
            logstream << "draw_bbox: frame=" << frame_counter_ << " no bbox metadata";
            return;
        }
        const BBox& bbox = boxes.front();
        logstream << "draw_bbox: frame=" << frame_counter_
                  << " boxes=" << boxes.size()
                  << " first_bbox=[" << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 << "," << bbox.y2 << "]"
                  << " thickness=" << bbox_thickness_;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_bbox_ptx, avpl_draw_bbox_ptx_len,
                         "kDrawBBoxNV12Luma", "kDrawBBoxNV12Chroma")) {
            throw Error("draw_bbox: failed to initialize CUDA kernels");
        }

        int vdw = 0;
        int vdh = 0;
        if (tryReadViewportDstDims(input, vdw, vdh)) {
            if (input_params_.width != last_viewport_input_w_ || input_params_.height != last_viewport_input_h_) {
                have_last_viewport_crop_ = false;
                last_viewport_input_w_ = input_params_.width;
                last_viewport_input_h_ = input_params_.height;
            }
            if (vdw != last_viewport_dst_w_ || vdh != last_viewport_dst_h_) {
                have_last_viewport_crop_ = false;
                last_viewport_dst_w_ = vdw;
                last_viewport_dst_h_ = vdh;
            }
            int vx = 0;
            int vy = 0;
            updateViewportCropPosition(input, vdw, vdh, vx, vy);
            DrawColor white{};
            if (!cuda_overlay::tryParseNamedColor("white", white)) {
                white = DrawColor{235, 128, 128};
            }
            BBox viewport{};
            viewport.x1 = clampInt(vx, 0, input_params_.width);
            viewport.y1 = clampInt(vy, 0, input_params_.height);
            viewport.x2 = clampInt(vx + vdw, 0, input_params_.width);
            viewport.y2 = clampInt(vy + vdh, 0, input_params_.height);
            viewport.y_color = white.y;
            viewport.u_color = white.u;
            viewport.v_color = white.v;
            if (viewport.x2 > viewport.x1 && viewport.y2 > viewport.y1) {
                if (!drawBBoxOnFrame(output, viewport)) {
                    throw Error("draw_bbox: failed drawing viewport bbox");
                }
                if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                    logstream << "draw_bbox: frame=" << frame_counter_
                              << " viewport=[" << viewport.x1 << "," << viewport.y1
                              << "," << viewport.x2 << "," << viewport.y2 << "]";
                }
            }
        } else {
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "draw_bbox: frame=" << frame_counter_ << " no viewport dims found";
            }
        }

        std::vector<BBox> boxes;
        const bool have_bbox = parseBBoxes(input, boxes);
        if (have_bbox) {
            for (const BBox& bbox : boxes) {
                if (!drawBBoxOnFrame(output, bbox)) {
                    throw Error("draw_bbox: failed drawing bbox");
                }
            }
        }
        maybeLogFrame(boxes);
    }

public:
    DrawBBox(std::unique_ptr<Source<av::VideoFrame>> &&source,
             std::unique_ptr<Sink<av::VideoFrame>> &&sink,
             std::vector<std::string> metadata_keys,
             int bbox_thickness,
             double min_conf,
             std::unordered_set<int> allowed_classes,
             std::unordered_set<std::string> allowed_labels,
             std::unordered_map<int, DrawColor> model_colors,
             double model_content_width,
             double model_content_height,
             double model_content_offset_x,
             double model_content_offset_y,
             VideoParameters input_params,
             av::Rational frame_rate,
             av::Rational timebase,
             int debug_log_every_n)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          metadata_keys_(std::move(metadata_keys)),
          bbox_thickness_(bbox_thickness),
          debug_log_every_n_(debug_log_every_n),
          min_conf_(min_conf),
          allowed_classes_(std::move(allowed_classes)),
          allowed_labels_(std::move(allowed_labels)),
          model_colors_(std::move(model_colors)),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
        if (metadata_keys_.empty()) {
            throw Error("draw_bbox: metadata_keys must be non-empty (or pass metadata_key)");
        }
        if (bbox_thickness_ <= 0) {
            throw Error("draw_bbox: bbox_thickness must be positive");
        }
        if ((model_content_width_ > 0.0 || model_content_height_ > 0.0)
                && !(model_content_width_ > 0.0 && model_content_height_ > 0.0)) {
            throw Error("draw_bbox: model_content_width and model_content_height must both be positive when set");
        }
        if (model_content_offset_x_ < 0.0 || model_content_offset_y_ < 0.0) {
            throw Error("draw_bbox: model_content offsets must be non-negative");
        }
    }

    static std::shared_ptr<DrawBBox> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        std::vector<std::string> metadata_keys;
        if (params.count("metadata_keys") && params["metadata_keys"].is_array()) {
            for (const auto &item : params["metadata_keys"]) {
                if (!item.is_string())
                    throw Error("draw_bbox: metadata_keys entries must be strings");
                metadata_keys.push_back(item.get<std::string>());
            }
        }
        if (metadata_keys.empty()) {
            metadata_keys.push_back(params.value("metadata_key", std::string("reframer_bbox")));
        } else {
            std::unordered_set<std::string> seen;
            std::vector<std::string> unique_keys;
            unique_keys.reserve(metadata_keys.size());
            for (std::string &k : metadata_keys) {
                if (seen.insert(k).second)
                    unique_keys.push_back(std::move(k));
            }
            metadata_keys = std::move(unique_keys);
        }
        const int bbox_thickness = params.value("bbox_thickness", 2);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const double min_conf = params.value("min_conf", 0.0);
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);
        std::unordered_set<int> allowed_classes;
        std::unordered_set<std::string> allowed_labels;
        std::unordered_map<int, DrawColor> model_colors;
        if (params.count("allowed_classes") && params["allowed_classes"].is_array()) {
            for (const auto& item : params["allowed_classes"]) {
                allowed_classes.insert(item.get<int>());
            }
        }
        if (params.count("allowed_labels") && params["allowed_labels"].is_array()) {
            for (const auto& item : params["allowed_labels"]) {
                allowed_labels.insert(item.get<std::string>());
            }
        }
        if (params.count("model_colors") && params["model_colors"].is_object()) {
            for (auto it = params["model_colors"].begin(); it != params["model_colors"].end(); ++it) {
                DrawColor color;
                if (!it.value().is_string() || !cuda_overlay::tryParseNamedColor(it.value().get<std::string>(), color)) {
                    throw Error("draw_bbox: model_colors values must be one of: red, green, light_blue");
                }
                model_colors[std::stoi(it.key())] = color;
            }
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawBBox>(
            edges, params, std::move(metadata_keys), bbox_thickness, min_conf,
            std::move(allowed_classes), std::move(allowed_labels), std::move(model_colors),
            model_content_width, model_content_height, model_content_offset_x, model_content_offset_y,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_bbox, DrawBBox)
