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
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "../../../objs/src/nodes/hwaccel/draw_text.ptx.h"

static int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "draw_text: cuda function " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}
#define CHECK_CU(x) check_cu((x), #x)

namespace {

constexpr int kMaxOverlayChars = 96;

struct DrawColor {
    int y = 235;
    int u = 128;
    int v = 128;
};

struct OverlayText {
    std::array<char, kMaxOverlayChars> line1{};
    std::array<char, kMaxOverlayChars> line2{};
    int line1_len = 0;
    int line2_len = 0;
    int origin_x = 48;
    int origin_y = 48;
    int font_scale = 5;
    int line_spacing = 16;
    int bg_x = 0;
    int bg_y = 0;
    int bg_w = 0;
    int bg_h = 0;
};

static std::string normalizeColorName(std::string color_name) {
    std::transform(color_name.begin(), color_name.end(), color_name.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ') return '_';
        return (char)std::tolower(c);
    });
    return color_name;
}

static bool tryParseNamedColor(const std::string& color_name, DrawColor& color_out) {
    const std::string normalized = normalizeColorName(color_name);
    if (normalized == "white") {
        color_out = DrawColor{235, 128, 128};
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
    return false;
}

static std::string uppercaseAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        if (c == '-') return '_';
        return (char)std::toupper(c);
    });
    return text;
}

static int boundedStringLength(const std::array<char, kMaxOverlayChars>& text) {
    int len = 0;
    while (len < (int)text.size() && text[(size_t)len] != '\0') {
        ++len;
    }
    return len;
}

static OverlayText makeDefaultOverlay(int origin_x, int origin_y, int font_scale, int line_spacing) {
    OverlayText overlay;
    overlay.origin_x = origin_x;
    overlay.origin_y = origin_y;
    overlay.font_scale = font_scale;
    overlay.line_spacing = line_spacing;
    std::snprintf(overlay.line1.data(), overlay.line1.size(), "SHOTS: 0");
    overlay.line2[0] = '\0';
    overlay.line1_len = boundedStringLength(overlay.line1);
    overlay.line2_len = boundedStringLength(overlay.line2);
    return overlay;
}

} // namespace

class DrawText : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                 public IVideoFormatSource,
                 public IFrameRateSource,
                 public ITimeBaseSource {
private:
    std::string metadata_key_ = "basketball_analysis_v1";
    std::string analysis_object_key_ = "basketball_analysis";
    int origin_x_ = 48;
    int origin_y_ = 48;
    int font_scale_ = 5;
    int line_spacing_ = 16;
    int debug_log_every_n_ = 0;
    DrawColor text_color_{235, 128, 128};
    DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;

    VideoParameters input_params_{};
    av::Rational frame_rate_{0, 0};
    av::Rational timebase_{0, 0};

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule draw_module_ = nullptr;
    CUfunction draw_luma_kernel_ = nullptr;
    CUfunction draw_chroma_kernel_ = nullptr;
    CUdeviceptr line1_symbol_ = 0;
    CUdeviceptr line2_symbol_ = 0;
    size_t line_symbol_size_ = 0;

    uint64_t frame_counter_ = 0;

    void unloadKernels() {
        if (draw_module_ && cu_ctx_) {
            CHECK_CU(cuCtxSetCurrent(cu_ctx_));
            CHECK_CU(cuModuleUnload(draw_module_));
        }
        draw_module_ = nullptr;
        draw_luma_kernel_ = nullptr;
        draw_chroma_kernel_ = nullptr;
        line1_symbol_ = 0;
        line2_symbol_ = 0;
        line_symbol_size_ = 0;
    }

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "draw_text: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "draw_text: missing device_ctx/hwctx in frame";
            return false;
        }
        AVCUDADeviceContext* next_dev_ctx = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next_dev_ctx || !next_dev_ctx->cuda_ctx) {
            logstream << "draw_text: missing CUDA context in frame";
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

        const std::string ptx_str(avpl_draw_text_ptx, avpl_draw_text_ptx + avpl_draw_text_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&draw_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "draw_text: failed to load PTX module";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&draw_luma_kernel_, draw_module_, "kDrawTextNV12Luma"))) {
            logstream << "draw_text: failed to get luma kernel";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&draw_chroma_kernel_, draw_module_, "kDrawTextNV12Chroma"))) {
            logstream << "draw_text: failed to get chroma kernel";
            return false;
        }

        size_t line1_bytes = 0;
        size_t line2_bytes = 0;
        if (CHECK_CU(cuModuleGetGlobal(&line1_symbol_, &line1_bytes, draw_module_, "gDrawTextLine1"))) {
            logstream << "draw_text: failed to get line1 symbol";
            return false;
        }
        if (CHECK_CU(cuModuleGetGlobal(&line2_symbol_, &line2_bytes, draw_module_, "gDrawTextLine2"))) {
            logstream << "draw_text: failed to get line2 symbol";
            return false;
        }
        if (line1_bytes < kMaxOverlayChars || line2_bytes < kMaxOverlayChars) {
            logstream << "draw_text: PTX line buffers are too small";
            return false;
        }
        line_symbol_size_ = std::min(line1_bytes, line2_bytes);
        return true;
    }

    static const Parameters* findObjectChild(const Parameters* parent, const char* key) {
        if (!parent || !parent->contains(key)) return nullptr;
        const auto& child = (*parent)[key];
        if (!child.is_object()) return nullptr;
        return &child;
    }

    static int readIntOrDefault(const Parameters* parent, const char* key, int fallback) {
        if (!parent || !parent->contains(key)) return fallback;
        const auto& value = (*parent)[key];
        if (value.is_number_integer()) return value.get<int>();
        if (value.is_number_float()) return (int)std::lround(value.get<double>());
        return fallback;
    }

    void finalizeOverlayLayout(OverlayText& overlay) const {
        overlay.line1_len = boundedStringLength(overlay.line1);
        overlay.line2_len = boundedStringLength(overlay.line2);

        const int char_advance = std::max(1, 6 * overlay.font_scale);
        const int line_height = std::max(1, 7 * overlay.font_scale);
        const int text_w = std::max(overlay.line1_len, overlay.line2_len) * char_advance;
        const int text_h = line_height * (overlay.line2_len > 0 ? 2 : 1)
                         + (overlay.line2_len > 0 ? overlay.line_spacing : 0);
        const int pad_x = std::max(10, overlay.font_scale * 2);
        const int pad_y = std::max(8, overlay.font_scale);

        overlay.bg_x = overlay.origin_x - pad_x;
        overlay.bg_y = overlay.origin_y - pad_y;
        overlay.bg_w = text_w + pad_x * 2;
        overlay.bg_h = text_h + pad_y * 2;
    }

    OverlayText buildOverlayText(const av::VideoFrame& frm) const {
        OverlayText overlay = makeDefaultOverlay(origin_x_, origin_y_, font_scale_, line_spacing_);

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            finalizeOverlayLayout(overlay);
            return overlay;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            finalizeOverlayLayout(overlay);
            return overlay;
        }

        try {
            const Parameters md = Parameters::parse(entry->value);
            const Parameters* analysis_root = findObjectChild(&md, analysis_object_key_.c_str());
            const Parameters* analysis = findObjectChild(analysis_root, "analysis");
            const Parameters* totals = findObjectChild(analysis, "totals");

            const int shots = readIntOrDefault(totals, "shots", 0);

            std::snprintf(overlay.line1.data(), overlay.line1.size(),
                          "SHOTS: %d", shots);
            overlay.line2[0] = '\0';
        } catch (const std::exception&) {
            // Keep the default overlay when metadata is missing or malformed.
        }

        finalizeOverlayLayout(overlay);
        return overlay;
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

    bool copyInputFrame(const av::VideoFrame& frm, av::VideoFrame& out) {
        out.raw()->format = frm.raw()->format;
        out.raw()->width = frm.width();
        out.raw()->height = frm.height();

        const int ret = av_hwframe_get_buffer(frm.raw()->hw_frames_ctx, out.raw(), 0);
        if (ret < 0) {
            logstream << "draw_text: av_hwframe_get_buffer failed: " << av::error2string(ret);
            return false;
        }

        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[0], (size_t)out.raw()->linesize[0],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[0], (size_t)frm.raw()->linesize[0],
                       (size_t)frm.width(), (size_t)frm.height())) {
            logstream << "draw_text: luma plane copy failed";
            return false;
        }
        if (!copyPlane((CUdeviceptr)(uintptr_t)out.raw()->data[1], (size_t)out.raw()->linesize[1],
                       (CUdeviceptr)(uintptr_t)frm.raw()->data[1], (size_t)frm.raw()->linesize[1],
                       (size_t)frm.width(), (size_t)((frm.height() + 1) / 2))) {
            logstream << "draw_text: chroma plane copy failed";
            return false;
        }

        const int ret_props = av_frame_copy_props(out.raw(), frm.raw());
        if (ret_props < 0) {
            logstream << "draw_text: av_frame_copy_props failed: " << av::error2string(ret_props);
            return false;
        }
        return true;
    }

    bool copyOverlayTextToGpu(const OverlayText& overlay) const {
        if (!line1_symbol_ || !line2_symbol_ || line_symbol_size_ < kMaxOverlayChars) {
            return false;
        }
        if (CHECK_CU(cuMemcpyHtoDAsync(line1_symbol_, overlay.line1.data(),
                                       std::min(line_symbol_size_, overlay.line1.size()),
                                       cuda_dev_ctx_->stream))) {
            logstream << "draw_text: failed copying line1 text";
            return false;
        }
        if (CHECK_CU(cuMemcpyHtoDAsync(line2_symbol_, overlay.line2.data(),
                                       std::min(line_symbol_size_, overlay.line2.size()),
                                       cuda_dev_ctx_->stream))) {
            logstream << "draw_text: failed copying line2 text";
            return false;
        }
        return true;
    }

    bool drawTextOnFrame(av::VideoFrame& frm, const OverlayText& overlay) {
        const unsigned int block_x = 32;
        const unsigned int block_y = 8;

        const int launch_w = std::max(1, overlay.bg_w);
        const int launch_h = std::max(1, overlay.bg_h);
        const unsigned int grid_x = (unsigned int)(launch_w + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(launch_h + (int)block_y - 1) / block_y;
        const unsigned int uv_grid_x = (unsigned int)(((launch_w + 1) / 2) + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(((launch_h + 1) / 2) + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        size_t pitch_y = (size_t)frm.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        size_t pitch_uv = (size_t)frm.raw()->linesize[1];
        int width = frm.width();
        int height = frm.height();
        int origin_x = overlay.origin_x;
        int origin_y = overlay.origin_y;
        int font_scale = overlay.font_scale;
        int line_spacing = overlay.line_spacing;
        int line1_len = overlay.line1_len;
        int line2_len = overlay.line2_len;
        int bg_x = overlay.bg_x;
        int bg_y = overlay.bg_y;
        int bg_w = overlay.bg_w;
        int bg_h = overlay.bg_h;
        int draw_background = draw_background_ ? 1 : 0;
        int text_y = text_color_.y;
        int text_u = text_color_.u;
        int text_v = text_color_.v;
        int bg_y_color = background_color_.y;
        int bg_u = background_color_.u;
        int bg_v = background_color_.v;

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&origin_x, (void*)&origin_y,
            (void*)&font_scale, (void*)&line_spacing,
            (void*)&line1_len, (void*)&line2_len,
            (void*)&bg_x, (void*)&bg_y, (void*)&bg_w, (void*)&bg_h,
            (void*)&draw_background,
            (void*)&text_y, (void*)&bg_y_color
        };
        if (CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_text: failed launching luma kernel";
            return false;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&origin_x, (void*)&origin_y,
            (void*)&font_scale, (void*)&line_spacing,
            (void*)&line1_len, (void*)&line2_len,
            (void*)&bg_x, (void*)&bg_y, (void*)&bg_w, (void*)&bg_h,
            (void*)&draw_background,
            (void*)&text_u, (void*)&text_v,
            (void*)&bg_u, (void*)&bg_v
        };
        if (CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_text: failed launching chroma kernel";
            return false;
        }

        return CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void maybeLogOverlay(const OverlayText& overlay) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "draw_text: frame=" << frame_counter_
                  << " metadata_key=" << metadata_key_
                  << " analysis_object_key=" << analysis_object_key_
                  << " line1=\"" << overlay.line1.data() << "\""
                  << " line2=\"" << overlay.line2.data() << "\"";
    }

public:
    DrawText(std::unique_ptr<Source<av::VideoFrame>>&& source,
             std::unique_ptr<Sink<av::VideoFrame>>&& sink,
             std::string metadata_key,
             std::string analysis_object_key,
             int origin_x,
             int origin_y,
             int font_scale,
             int line_spacing,
             DrawColor text_color,
             DrawColor background_color,
             bool draw_background,
             VideoParameters input_params,
             av::Rational frame_rate,
             av::Rational timebase,
             int debug_log_every_n)
        : NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          analysis_object_key_(std::move(analysis_object_key)),
          origin_x_(origin_x),
          origin_y_(origin_y),
          font_scale_(font_scale),
          line_spacing_(line_spacing),
          debug_log_every_n_(debug_log_every_n),
          text_color_(text_color),
          background_color_(background_color),
          draw_background_(draw_background),
          input_params_(input_params),
          frame_rate_(frame_rate),
          timebase_(timebase) {
        if (origin_x_ < 0 || origin_y_ < 0) {
            throw Error("draw_text: origin_x and origin_y must be non-negative");
        }
        if (font_scale_ <= 0) {
            throw Error("draw_text: font_scale must be positive");
        }
        if (line_spacing_ < 0) {
            throw Error("draw_text: line_spacing must be non-negative");
        }
        if (metadata_key_.empty()) {
            throw Error("draw_text: metadata_key must not be empty");
        }
        if (analysis_object_key_.empty()) {
            throw Error("draw_text: analysis_object_key must not be empty");
        }
    }

    ~DrawText() override {
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
            throw Error("draw_text: non-CUDA frame received");
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
            throw Error("draw_text: only NV12 CUDA frames are supported");
        }
        if (!initCudaContextFromFrame(frm) || !loadKernels()) {
            throw Error("draw_text: failed to initialize CUDA kernels");
        }

        const OverlayText overlay = buildOverlayText(frm);

        av::VideoFrame out;
        if (!copyInputFrame(frm, out)) {
            throw Error("draw_text: failed to copy input frame");
        }
        if (!copyOverlayTextToGpu(overlay)) {
            throw Error("draw_text: failed to upload overlay text");
        }
        if (!drawTextOnFrame(out, overlay)) {
            throw Error("draw_text: failed drawing text");
        }

        out.setTimeBase(frm.timeBase());
        out.setComplete(true);
        maybeLogOverlay(overlay);
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

    static std::shared_ptr<DrawText> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto video_format_src = src_edge->findNodeUp<IVideoFormatSource>();
        auto frame_rate_src = src_edge->findNodeUp<IFrameRateSource>();
        auto timebase_src = src_edge->findNodeUp<ITimeBaseSource>();

        const std::string metadata_key = params.value("metadata_key", std::string("basketball_analysis_v1"));
        const std::string analysis_object_key = params.value("analysis_object_key", std::string("basketball_analysis"));
        const int origin_x = params.value("origin_x", 48);
        const int origin_y = params.value("origin_y", 48);
        const int font_scale = params.value("font_scale", 5);
        const int line_spacing = params.value("line_spacing", 16);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);

        DrawColor text_color{235, 128, 128};
        if (params.count("text_color")) {
            if (!params["text_color"].is_string() || !tryParseNamedColor(params["text_color"].get<std::string>(), text_color)) {
                throw Error("draw_text: text_color must be one of: white, black, red, green, yellow, light_blue");
            }
        }

        DrawColor background_color{16, 128, 128};
        bool draw_background = true;
        if (params.count("background_color")) {
            if (!params["background_color"].is_string()) {
                throw Error("draw_text: background_color must be a string color name or \"none\"");
            }
            const std::string background_name = normalizeColorName(params["background_color"].get<std::string>());
            if (background_name == "none") {
                draw_background = false;
            } else if (!tryParseNamedColor(background_name, background_color)) {
                throw Error("draw_text: background_color must be one of: white, black, red, green, yellow, light_blue, none");
            }
        }

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

        const av::Rational frame_rate = frame_rate_src ? frame_rate_src->frameRate() : av::Rational{0, 0};
        const av::Rational timebase = timebase_src ? timebase_src->timeBase() : av::Rational{0, 0};

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawText>(
            edges, params, metadata_key, analysis_object_key, origin_x, origin_y,
            font_scale, line_spacing, text_color, background_color, draw_background,
            input_params, frame_rate, timebase, debug_log_every_n);
    }
};

DECLNODE(draw_text, DrawText);
