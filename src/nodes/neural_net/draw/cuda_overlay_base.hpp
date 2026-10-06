#pragma once

#include "../../node_common.hpp"
#include "../../../video_parameters.hpp"
#include "../../../hwaccel.hpp"
#include "../../hwaccel/cuda_rect_frame.hpp"
#include "picture_frame.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuda_overlay {

struct DrawColor {
    int y = 173;
    int u = 42;
    int v = 26;
};

inline int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}

#define CUDA_OVERLAY_CHECK_CU(x) cuda_overlay::check_cu((x), #x)

inline std::string normalizeColorName(std::string color_name) {
    std::transform(color_name.begin(), color_name.end(), color_name.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ') return '_';
        return (char)std::tolower(c);
    });
    return color_name;
}

inline bool tryParseNamedColor(const std::string& color_name, DrawColor& color_out) {
    const std::string normalized = normalizeColorName(color_name);
    if (normalized == "white") {
        color_out = DrawColor{235, 128, 128};
        return true;
    }
    if (normalized == "light_gray" || normalized == "light_grey") {
        color_out = DrawColor{200, 128, 128};
        return true;
    }
    if (normalized == "gray" || normalized == "grey") {
        color_out = DrawColor{128, 128, 128};
        return true;
    }
    if (normalized == "black") {
        color_out = DrawColor{16, 128, 128};
        return true;
    }
    if (normalized == "green") {
        color_out = DrawColor{173, 42, 26};
        return true;
    }
    if (normalized == "red") {
        color_out = DrawColor{81, 90, 240};
        return true;
    }
    if (normalized == "light_blue") {
        color_out = DrawColor{169, 166, 16};
        return true;
    }
    if (normalized == "yellow") {
        color_out = DrawColor{210, 16, 146};
        return true;
    }
    if (normalized == "orange") {
        color_out = DrawColor{156, 44, 200};
        return true;
    }
    if (normalized == "magenta" || normalized == "pink") {
        color_out = DrawColor{105, 212, 235};
        return true;
    }
    if (normalized == "cyan") {
        color_out = DrawColor{188, 154, 16};
        return true;
    }
    if (normalized == "purple") {
        color_out = DrawColor{76, 184, 230};
        return true;
    }
    return false;
}

enum class GlyphPreset {
    k5x7 = 0,
    k10x14 = 1,
};

bool tryParseGlyphPreset(const std::string& preset_name, GlyphPreset& preset_out);
int glyphBaseWidth(GlyphPreset preset);
int glyphBaseHeight(GlyphPreset preset);
int glyphAdvance(GlyphPreset preset);

struct YoloParseConfig {
    int frame_width = 0;
    int frame_height = 0;
    double min_conf = 0.0;
    const std::unordered_set<int>* allowed_classes = nullptr;
    const std::unordered_set<std::string>* allowed_labels = nullptr;
    double model_content_width = 0.0;
    double model_content_height = 0.0;
    double model_content_offset_x = 0.0;
    double model_content_offset_y = 0.0;
};

struct ParsedYoloDetection {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int cls = -1;
    bool has_cls = false;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    int model_index = -1;
    bool has_model_index = false;
    int track_id = -1;
    bool has_track_id = false;
    bool predicted = false;
    bool has_predicted = false;
    bool has_velocity = false;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
};

bool scaleAndClampBBox(double x1, double y1, double x2, double y2, int frame_width, int frame_height,
                       int& out_x1, int& out_y1, int& out_x2, int& out_y2);
bool remapModelCoord(const YoloParseConfig& cfg, double x, double y,
                     double model_w, double model_h, double& out_x, double& out_y);
bool yoloDetectionAllowed(const Parameters& det, const YoloParseConfig& cfg);
void parseYoloDetections(const Parameters& md, const YoloParseConfig& cfg,
                         std::vector<ParsedYoloDetection>& detections_out);

} // namespace cuda_overlay

class CudaOverlayBase : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                         public IVideoFormatSource,
                         public IFrameRateSource,
                         public ITimeBaseSource,
                         public ReportsFinishByFlag,
                         public IReturnsObjects {
protected:
    VideoParameters input_params_{};
    av::Rational frame_rate_{0, 0};
    av::Rational timebase_{0, 0};

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule draw_module_ = nullptr;
    CUfunction draw_luma_kernel_ = nullptr;
    CUfunction draw_chroma_kernel_ = nullptr;

    uint64_t frame_counter_ = 0;

    // The node's own pool of the pictures it draws on. An input that is not a picture of a draw
    // node above is copied into a frame of it. A CUarray input is a decoder surface, which nothing
    // may draw on: the rect compositor's array reader reads it into a frame of this pool.
    std::unique_ptr<avp::mixer::CudaRectDraw> picture_draw_;
    avp::AvBufferRef picture_frames_;
    // Frames by where the picture the node drew on came from; see getObject.
    std::atomic<uint64_t> from_array_{0}, copied_{0}, in_place_{0};

    using NodeSISO::NodeSISO;

    void unloadKernels() {
        if (draw_module_ && cu_ctx_) {
            CUDA_OVERLAY_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
            CUDA_OVERLAY_CHECK_CU(cuModuleUnload(draw_module_));
        }
        draw_module_ = nullptr;
        draw_luma_kernel_ = nullptr;
        draw_chroma_kernel_ = nullptr;
        onKernelsUnloaded();
    }

    // Override to clean up extra kernel state (e.g. device symbols).
    virtual void onKernelsUnloaded() {}

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << nodeName() << ": missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << nodeName() << ": missing device_ctx/hwctx in frame";
            return false;
        }
        AVCUDADeviceContext* next_dev_ctx = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next_dev_ctx || !next_dev_ctx->cuda_ctx) {
            logstream << nodeName() << ": missing CUDA context in frame";
            return false;
        }
        if (cu_ctx_ && cu_ctx_ != next_dev_ctx->cuda_ctx) {
            unloadKernels();
            // The array reader unloads under the context it was loaded in, which it makes current.
            picture_draw_.reset();
        }
        cuda_dev_ctx_ = next_dev_ctx;
        cu_ctx_ = next_dev_ctx->cuda_ctx;
        return CUDA_OVERLAY_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernels(const unsigned char* ptx_data, unsigned int ptx_len,
                     const char* luma_kernel_name, const char* chroma_kernel_name) {
        if (draw_module_ && draw_luma_kernel_ && draw_chroma_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CUDA_OVERLAY_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;

        const std::string ptx_str(ptx_data, ptx_data + ptx_len);
        if (CUDA_OVERLAY_CHECK_CU(cuModuleLoadDataEx(&draw_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << nodeName() << ": failed to load PTX module";
            return false;
        }
        if (CUDA_OVERLAY_CHECK_CU(cuModuleGetFunction(&draw_luma_kernel_, draw_module_, luma_kernel_name))) {
            logstream << nodeName() << ": failed to get luma kernel";
            return false;
        }
        if (CUDA_OVERLAY_CHECK_CU(cuModuleGetFunction(&draw_chroma_kernel_, draw_module_, chroma_kernel_name))) {
            logstream << nodeName() << ": failed to get chroma kernel";
            return false;
        }
        return true;
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
        return CUDA_OVERLAY_CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream)) == 0;
    }

    /// Makes picture_frames_ the node's own marked pool of linear NV12 frames of `width` x `height`
    /// on the device of `src`. The array reader is made for the pool's device and size, so it goes
    /// with the pool it was made for.
    void ensurePictureFrames(const AVHWFramesContext* src, int width, int height) {
        const AVHWFramesContext* own = picture_frames_ ? (const AVHWFramesContext*)picture_frames_->data : nullptr;
        if (own && own->device_ctx == src->device_ctx && own->width == width && own->height == height) return;
        picture_draw_.reset();
        picture_frames_.reset();
        HWAccelDevice hw(av_buffer_ref(src->device_ref));
        picture_frames_ = avp::mixer::allocCanvasFrames(hw, {width, height, AV_PIX_FMT_NV12}, nodeName());
        cuda_overlay::markPicturePool(picture_frames_.get());
    }

    bool copyInputFrame(const av::VideoFrame& frm, av::VideoFrame& out) {
        // The copy is a frame of a picture pool, so that the draw nodes below draw on it instead of
        // copying it again. A picture that is still referenced elsewhere is copied within the pool
        // it came from, which keeps a chain's output in one frames context. Any other input is
        // copied into the node's own pool, which has the device, format and size of the input's
        // pool and so gives the frame that pool would have given.
        AVBufferRef* frames = frm.raw()->hw_frames_ctx;
        if (!cuda_overlay::isPicturePool(frames)) {
            const AVHWFramesContext* src = (const AVHWFramesContext*)frames->data;
            ensurePictureFrames(src, src->width, src->height);
            frames = picture_frames_.get();
        }
        out.raw()->format = frm.raw()->format;
        out.raw()->width = frm.width();
        out.raw()->height = frm.height();

        int ret = av_hwframe_get_buffer(frames, out.raw(), 0);
        if (ret < 0) {
            logstream << nodeName() << ": av_hwframe_get_buffer failed: " << av::error2string(ret);
            return false;
        }

        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[0], (size_t)out.raw()->linesize[0],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[0], (size_t)frm.raw()->linesize[0],
                       (size_t)frm.width(), (size_t)frm.height())) {
            logstream << nodeName() << ": luma plane copy failed";
            return false;
        }

        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[1], (size_t)out.raw()->linesize[1],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[1], (size_t)frm.raw()->linesize[1],
                       (size_t)frm.width(), (size_t)((frm.height() + 1) / 2))) {
            logstream << nodeName() << ": chroma plane copy failed";
            return false;
        }

        ret = cuda_overlay::copyFrameProps(out.raw(), frm.raw());
        if (ret < 0) {
            logstream << nodeName() << ": copying frame properties failed: " << av::error2string(ret);
            return false;
        }
        return true;
    }

    /// The linear picture of a CUarray frame: one whole-frame layer drawn 1:1 by the rect
    /// compositor's draw into a frame of the node's own pool, on the stream of the frame's own
    /// device, which is the stream the node's kernels use. The pool is marked, so the draw nodes
    /// below draw on the picture itself (isPrivatePicture). The read of `frm` is only queued, as
    /// the last thing that can fail here: the caller synchronizes the stream before it releases
    /// `frm`, also when it fails.
    void linearPicture(const av::VideoFrame& frm, av::VideoFrame& out) {
        const AVHWFramesContext* src = (const AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        ensurePictureFrames(src, frm.width(), frm.height());
        if (!picture_draw_) {
            picture_draw_ = std::make_unique<avp::mixer::CudaRectDraw>(
                std::make_shared<HWAccelDevice>(av_buffer_ref(src->device_ref)),
                avp::mixer::CudaRectDraw::Canvas{frm.width(), frm.height(), AV_PIX_FMT_NV12}, 1);
        }
        out = avp::mixer::canvasFrame(picture_frames_.get(), nodeName());
        const int ret = cuda_overlay::copyFrameProps(out.raw(), frm.raw());
        if (ret < 0) {
            throw Error(std::string(nodeName()) + ": copying frame properties failed: " + av::error2string(ret));
        }
        const auto ops = avp::mixer::resolveDrawOps({&frm}, {avp::mixer::LayerSpec{}},
                                                    frm.width(), frm.height(), AV_PIX_FMT_NV12);
        picture_draw_->draw(cuda_dev_ctx_->stream, ops, out.raw(), frm.raw());
    }

    // Subclass must return its node type name for log messages.
    virtual const char* nodeName() const = 0;

    // Subclass implements the actual drawing on the output frame, which is a copy of the
    // input or, for a private picture, the input frame itself. Called before the frame is put
    // to the sink.
    virtual void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) = 0;

    // Common video params parsing for create() factories.
    struct UpstreamInfo {
        VideoParameters input_params;
        av::Rational frame_rate{0, 0};
        av::Rational timebase{0, 0};
    };

    static UpstreamInfo resolveUpstreamInfo(const std::shared_ptr<Edge<av::VideoFrame>>& src_edge,
                                            const Parameters& params) {
        UpstreamInfo info;
        auto video_format_src = src_edge->template findNodeUp<IVideoFormatSource>();
        auto frame_rate_src = src_edge->template findNodeUp<IFrameRateSource>();
        auto timebase_src = src_edge->template findNodeUp<ITimeBaseSource>();

        if (video_format_src) {
            info.input_params.width = video_format_src->width();
            info.input_params.height = video_format_src->height();
            info.input_params.pixel_format = video_format_src->pixelFormat();
            info.input_params.real_pixel_format = video_format_src->realPixelFormat();
        }
        if (params.count("width")) info.input_params.width = params["width"];
        if (params.count("height")) info.input_params.height = params["height"];
        if (params.count("pixel_format")) info.input_params.pixel_format = av::PixelFormat(params["pixel_format"].get<std::string>());
        if (params.count("real_pixel_format")) info.input_params.real_pixel_format = av::PixelFormat(params["real_pixel_format"].get<std::string>());
        info.frame_rate = frame_rate_src ? frame_rate_src->frameRate() : av::Rational{0, 0};
        info.timebase = timebase_src ? timebase_src->timeBase() : av::Rational{0, 0};
        return info;
    }

public:
    ~CudaOverlayBase() override {
        unloadKernels();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        const AVPixelFormat format = (AVPixelFormat)frm.raw()->format;
        if (!avp::mixer::CudaRectDraw::frameSupported(format)) {
            throw Error(std::string(nodeName()) + ": input must be a CUDA device frame or CUarray");
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
            throw Error(std::string(nodeName()) + ": only NV12 frames are supported");
        }
        if (!initCudaContextFromFrame(frm)) {
            throw Error(std::string(nodeName()) + ": failed to init CUDA context");
        }

        const bool array = format != AV_PIX_FMT_CUDA;
        const bool in_place = !array && cuda_overlay::isPrivatePicture(frm.raw());
        av::VideoFrame picture;
        if (array) {
            linearPicture(frm, picture);
        } else if (!in_place && !copyInputFrame(frm, picture)) {
            throw Error(std::string(nodeName()) + ": failed to copy input frame");
        }
        av::VideoFrame& out = in_place ? frm : picture;
        ++(array ? from_array_ : in_place ? in_place_ : copied_);

        try {
            drawOnFrame(frm, out);
        } catch (...) {
            // The picture's read of the surface may still be queued, and unwinding releases frm.
            if (array) CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
            throw;
        }

        out.setTimeBase(frm.timeBase());
        out.setComplete(true);
        if (array) {
            // frm is a decoder surface. The picture's read of it must have finished before the
            // decoder gets it back, and it must not stay referenced while put waits.
            if (CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
                throw Error(std::string(nodeName()) + ": cuStreamSynchronize failed");
            }
            frm = av::VideoFrame();
        }
        this->sink_->put(out);
    }

    /// `pictures`: how many frames the node drew on a linear picture it made of a CUarray
    /// (`from_array`), on a copy of its input (`copied`) and on its input frame (`in_place`).
    Parameters getObject(const std::string name) override {
        if (name != "pictures") {
            throw Error(std::string(nodeName()) + ": unknown object " + name);
        }
        return {{"from_array", from_array_.load()}, {"copied", copied_.load()}, {"in_place", in_place_.load()}};
    }

    int width() override { return input_params_.width; }
    int height() override { return input_params_.height; }

    // The output is linear CUDA whatever the input's storage: a CUarray is declared as the
    // frame it becomes.
    av::PixelFormat pixelFormat() override {
        const AVPixelFormat format = input_params_.pixel_format.get();
        return format == AV_PIX_FMT_NONE || avp::mixer::CudaRectDraw::frameSupported(format)
            ? av::PixelFormat(AV_PIX_FMT_CUDA) : input_params_.pixel_format;
    }

    av::PixelFormat realPixelFormat() override {
        return input_params_.real_pixel_format == AV_PIX_FMT_NONE ? pixelFormat() : input_params_.real_pixel_format;
    }

    av::Rational frameRate() override { return frame_rate_; }
    av::Rational timeBase() override { return timebase_; }
};
