#include "../node_common.hpp"
#include "../../video_parameters.hpp"
#include "../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/draw_bbox.ptx.h"

static int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "draw_bbox: cuda function " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}
#define CHECK_CU(x) check_cu((x), #x)

namespace {
struct BBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};
}

class DrawBBox : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                 public IVideoFormatSource,
                 public IFrameRateSource,
                 public ITimeBaseSource {
private:
    std::string metadata_key_ = "reframer_bbox";
    int bbox_thickness_ = 2;
    int debug_log_every_n_ = 0;
    double min_conf_ = 0.0;
    std::unordered_set<int> allowed_classes_;
    std::unordered_set<std::string> allowed_labels_;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;

    VideoParameters input_params_{};
    av::Rational frame_rate_{0, 0};
    av::Rational timebase_{0, 0};

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule draw_module_ = nullptr;
    CUfunction draw_luma_kernel_ = nullptr;
    CUfunction draw_chroma_kernel_ = nullptr;

    uint64_t frame_counter_ = 0;

    void unloadKernels() {
        if (draw_module_ && cu_ctx_) {
            CHECK_CU(cuCtxSetCurrent(cu_ctx_));
            CHECK_CU(cuModuleUnload(draw_module_));
        }
        draw_module_ = nullptr;
        draw_luma_kernel_ = nullptr;
        draw_chroma_kernel_ = nullptr;
    }

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "draw_bbox: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "draw_bbox: missing device_ctx/hwctx in frame";
            return false;
        }
        AVCUDADeviceContext* next_dev_ctx = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next_dev_ctx || !next_dev_ctx->cuda_ctx) {
            logstream << "draw_bbox: missing CUDA context in frame";
            return false;
        }
        if (cu_ctx_ && cu_ctx_ != next_dev_ctx->cuda_ctx) {
            unloadKernels();
        }
        cuda_dev_ctx_ = next_dev_ctx;
        cu_ctx_ = next_dev_ctx->cuda_ctx;
        return CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernels() {
        if (draw_module_ && draw_luma_kernel_ && draw_chroma_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;

        const std::string ptx_str(avpl_draw_bbox_ptx, avpl_draw_bbox_ptx + avpl_draw_bbox_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&draw_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "draw_bbox: failed to load PTX module";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&draw_luma_kernel_, draw_module_, "kDrawBBoxNV12Luma"))) {
            logstream << "draw_bbox: failed to get luma kernel";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&draw_chroma_kernel_, draw_module_, "kDrawBBoxNV12Chroma"))) {
            logstream << "draw_bbox: failed to get chroma kernel";
            return false;
        }
        return true;
    }

    static int clampInt(int value, int lo, int hi) {
        return std::max(lo, std::min(hi, value));
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

    bool parseSingleBBoxMetadata(const Parameters &md, BBox &bbox_out) const {
        double x1 = NAN;
        double y1 = NAN;
        double x2 = NAN;
        double y2 = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
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
            if (scaleAndClampBBox(
                    x1,
                    y1,
                    x2,
                    y2,
                    bbox)) {
                boxes_out.push_back(bbox);
            }
        }
    }

    bool parseBBoxes(const av::VideoFrame &frm, std::vector<BBox> &boxes_out) const {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry *entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            boxes_out.clear();

            BBox single_bbox;
            if (parseSingleBBoxMetadata(md, single_bbox)) {
                boxes_out.push_back(single_bbox);
            } else {
                parseYoloDetections(md, boxes_out);
            }
            return !boxes_out.empty();
        } catch (const std::exception &) {
            return false;
        }
    }

    bool copyPlane(CUdeviceptr dst, size_t dst_pitch,
                   CUdeviceptr src, size_t src_pitch,
                   size_t width_bytes, size_t height) const {
        CUDA_MEMCPY2D cpy{};
        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.srcDevice = src;
        cpy.srcPitch = src_pitch;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = dst;
        cpy.dstPitch = dst_pitch;
        cpy.WidthInBytes = width_bytes;
        cpy.Height = height;
        return CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream)) == 0;
    }

    bool copyInputFrame(const av::VideoFrame &frm, av::VideoFrame &out) {
        out.raw()->format = frm.raw()->format;
        out.raw()->width = frm.width();
        out.raw()->height = frm.height();

        int ret = av_hwframe_get_buffer(frm.raw()->hw_frames_ctx, out.raw(), 0);
        if (ret < 0) {
            logstream << "draw_bbox: av_hwframe_get_buffer failed: " << av::error2string(ret);
            return false;
        }

        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[0], (size_t)out.raw()->linesize[0],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[0], (size_t)frm.raw()->linesize[0],
                       (size_t)frm.width(), (size_t)frm.height())) {
            logstream << "draw_bbox: luma plane copy failed";
            return false;
        }

        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[1], (size_t)out.raw()->linesize[1],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[1], (size_t)frm.raw()->linesize[1],
                       (size_t)frm.width(), (size_t)((frm.height() + 1) / 2))) {
            logstream << "draw_bbox: chroma plane copy failed";
            return false;
        }

        ret = av_frame_copy_props(out.raw(), frm.raw());
        if (ret < 0) {
            logstream << "draw_bbox: av_frame_copy_props failed: " << av::error2string(ret);
            return false;
        }
        return true;
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

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&x1, (void*)&y1, (void*)&x2, (void*)&y2,
            (void*)&thickness
        };
        if (CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
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
            (void*)&thickness
        };
        if (CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_bbox: failed launching chroma kernel";
            return false;
        }

        return CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
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

public:
    DrawBBox(std::unique_ptr<Source<av::VideoFrame>> &&source,
             std::unique_ptr<Sink<av::VideoFrame>> &&sink,
             std::string metadata_key,
             int bbox_thickness,
             double min_conf,
             std::unordered_set<int> allowed_classes,
             std::unordered_set<std::string> allowed_labels,
             double model_content_width,
             double model_content_height,
             double model_content_offset_x,
             double model_content_offset_y,
             VideoParameters input_params,
             av::Rational frame_rate,
             av::Rational timebase,
             int debug_log_every_n)
        : NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          bbox_thickness_(bbox_thickness),
          debug_log_every_n_(debug_log_every_n),
          min_conf_(min_conf),
          allowed_classes_(std::move(allowed_classes)),
          allowed_labels_(std::move(allowed_labels)),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y),
          input_params_(input_params),
          frame_rate_(frame_rate),
          timebase_(timebase) {
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

    ~DrawBBox() override {
        unloadKernels();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            throw Error("draw_bbox: non-CUDA frame received");
        }

        input_params_ = VideoParameters(frm);
        timebase_ = frm.timeBase();
        if (frame_rate_.getNumerator() == 0 || frame_rate_.getDenominator() == 0) {
            if (timebase_.getNumerator() > 0 && timebase_.getDenominator() > 0) {
                frame_rate_ = av::Rational(timebase_.getDenominator(), timebase_.getNumerator());
            }
            if (frame_rate_.getNumerator() == 0 || frame_rate_.getDenominator() == 0) {
                frame_rate_ = av::Rational(30, 1);
            }
        }

        if (input_params_.realPixelFormat() != AV_PIX_FMT_NV12) {
            throw Error("draw_bbox: only NV12 CUDA frames are supported");
        }
        if (!initCudaContextFromFrame(frm) || !loadKernels()) {
            throw Error("draw_bbox: failed to initialize CUDA kernels");
        }

        av::VideoFrame out;
        if (!copyInputFrame(frm, out)) {
            throw Error("draw_bbox: failed to copy input frame");
        }

        std::vector<BBox> boxes;
        const bool have_bbox = parseBBoxes(frm, boxes);
        if (have_bbox) {
            for (const BBox& bbox : boxes) {
                if (!drawBBoxOnFrame(out, bbox)) {
                    throw Error("draw_bbox: failed drawing bbox");
                }
            }
        }

        out.setTimeBase(frm.timeBase());
        out.setComplete(true);
        maybeLogFrame(boxes);
        this->sink_->put(out);
    }

    int width() override {
        return input_params_.width;
    }

    int height() override {
        return input_params_.height;
    }

    av::PixelFormat pixelFormat() override {
        return input_params_.pixel_format == AV_PIX_FMT_NONE ? av::PixelFormat(AV_PIX_FMT_CUDA) : input_params_.pixel_format;
    }

    av::PixelFormat realPixelFormat() override {
        return input_params_.real_pixel_format == AV_PIX_FMT_NONE ? pixelFormat() : input_params_.real_pixel_format;
    }

    av::Rational frameRate() override {
        return frame_rate_;
    }

    av::Rational timeBase() override {
        return timebase_;
    }

    static std::shared_ptr<DrawBBox> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto video_format_src = src_edge->findNodeUp<IVideoFormatSource>();
        auto frame_rate_src = src_edge->findNodeUp<IFrameRateSource>();
        auto timebase_src = src_edge->findNodeUp<ITimeBaseSource>();

        const std::string metadata_key = params.value("metadata_key", std::string("reframer_bbox"));
        const int bbox_thickness = params.value("bbox_thickness", 2);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const double min_conf = params.value("min_conf", 0.0);
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);
        std::unordered_set<int> allowed_classes;
        std::unordered_set<std::string> allowed_labels;
        VideoParameters input_params;
        if (video_format_src) {
            input_params.width = video_format_src->width();
            input_params.height = video_format_src->height();
            input_params.pixel_format = video_format_src->pixelFormat();
            input_params.real_pixel_format = video_format_src->realPixelFormat();
        }
        if (params.count("width")) input_params.width = params["width"];
        if (params.count("height")) input_params.height = params["height"];
        if (params.count("pixel_format")) input_params.pixel_format = av::PixelFormat(params["pixel_format"].get<std::string>());
        if (params.count("real_pixel_format")) input_params.real_pixel_format = av::PixelFormat(params["real_pixel_format"].get<std::string>());
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
        const av::Rational frame_rate = frame_rate_src ? frame_rate_src->frameRate() : av::Rational{0, 0};
        const av::Rational timebase = timebase_src ? timebase_src->timeBase() : av::Rational{0, 0};

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawBBox>(
            edges, params, metadata_key, bbox_thickness, min_conf,
            std::move(allowed_classes), std::move(allowed_labels),
            model_content_width, model_content_height, model_content_offset_x, model_content_offset_y,
            input_params, frame_rate, timebase, debug_log_every_n);
    }
};

DECLNODE(draw_bbox, DrawBBox);
