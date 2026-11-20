#include "node_common.hpp"

#if HAVE_CUDA && HAVE_NVJPEG2K

#include "../hwaccel.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <nvjpeg2k.h>

namespace {

std::string nvjpeg2kStatusString(nvjpeg2kStatus_t st) {
    switch (st) {
        case NVJPEG2K_STATUS_SUCCESS: return "success";
        case NVJPEG2K_STATUS_NOT_INITIALIZED: return "not initialized";
        case NVJPEG2K_STATUS_INVALID_PARAMETER: return "invalid parameter";
        case NVJPEG2K_STATUS_BAD_JPEG: return "bad jpeg2000";
        case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED: return "jpeg2000 not supported";
        case NVJPEG2K_STATUS_ALLOCATOR_FAILURE: return "allocator failure";
        case NVJPEG2K_STATUS_EXECUTION_FAILED: return "execution failed";
        case NVJPEG2K_STATUS_ARCH_MISMATCH: return "arch mismatch";
        case NVJPEG2K_STATUS_INTERNAL_ERROR: return "internal error";
        case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "implementation not supported";
        case NVJPEG2K_STATUS_TILE_TOO_LARGE: return "tile too large";
        default: return "status " + std::to_string(static_cast<int>(st));
    }
}

void checkNvjpeg2k(nvjpeg2kStatus_t st, const char *what) {
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        throw Error(std::string("nvjpeg2k_dec ") + what + " failed: " + nvjpeg2kStatusString(st));
    }
}

void checkCuda(cudaError_t st, const char *what) {
    if (st != cudaSuccess) {
        throw Error(std::string("nvjpeg2k_dec ") + what + " failed: " + cudaGetErrorString(st));
    }
}

AVPixelFormat formatForPrecision(int precision) {
    if (precision <= 8) return AV_PIX_FMT_YUV444P;
    if (precision <= 10) return AV_PIX_FMT_YUV444P10LE;
    if (precision <= 12) return AV_PIX_FMT_YUV444P12LE;
    if (precision <= 16) return AV_PIX_FMT_YUV444P16LE;
    throw Error("nvjpeg2k_dec unsupported component precision " + std::to_string(precision));
}

AVPixelFormat parsePixelFormat(const std::string &name) {
    const AVPixelFormat fmt = av_get_pix_fmt(name.c_str());
    if (fmt == AV_PIX_FMT_NONE) {
        throw Error("nvjpeg2k_dec unknown real_pixel_format: " + name);
    }
    return fmt;
}

bool isSupportedOutputFormat(AVPixelFormat fmt) {
    return fmt == AV_PIX_FMT_YUV444P || fmt == AV_PIX_FMT_YUV444P10LE ||
           fmt == AV_PIX_FMT_YUV444P12LE || fmt == AV_PIX_FMT_YUV444P16LE;
}

} // namespace

class Nvjpeg2kDecoderNode : public NodeSISO<av::Packet, av::VideoFrame>,
                            public IVideoFormatSource,
                            public ITimeBaseSource,
                            public IFrameRateSource,
                            public ReportsFinishByFlag {
    nvjpeg2kHandle_t handle_ = nullptr;
    nvjpeg2kDecodeState_t decode_state_ = nullptr;
    nvjpeg2kStream_t stream_ = nullptr;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef *hw_frames_ctx_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    AVPixelFormat output_format_ = AV_PIX_FMT_YUV444P;
    av::Rational timebase_{0, 1};
    av::Rational framerate_{0, 1};
    int64_t frames_seen_ = 0;

    void ensureDecoder() {
        if (handle_) return;
        checkNvjpeg2k(nvjpeg2kCreateSimple(&handle_), "create");
        checkNvjpeg2k(nvjpeg2kDecodeStateCreate(handle_, &decode_state_), "decode state create");
        checkNvjpeg2k(nvjpeg2kStreamCreate(&stream_), "stream create");
    }

    void destroyDecoder() {
        if (stream_) {
            nvjpeg2kStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (decode_state_) {
            nvjpeg2kDecodeStateDestroy(decode_state_);
            decode_state_ = nullptr;
        }
        if (handle_) {
            nvjpeg2kDestroy(handle_);
            handle_ = nullptr;
        }
    }

    void ensureFrames(int width, int height, AVPixelFormat format) {
        if (hw_frames_ctx_ && width == width_ && height == height_ && format == output_format_) return;
        if (hw_frames_ctx_) {
            av_buffer_unref(&hw_frames_ctx_);
        }
        hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
        if (!hw_frames_ctx_) throw Error("nvjpeg2k_dec failed to allocate hw frames context");
        auto *frmctx = reinterpret_cast<AVHWFramesContext *>(hw_frames_ctx_->data);
        frmctx->format = AV_PIX_FMT_CUDA;
        frmctx->sw_format = format;
        frmctx->width = width;
        frmctx->height = height;
        const int r = av_hwframe_ctx_init(hw_frames_ctx_);
        if (r < 0) {
            throw Error("nvjpeg2k_dec av_hwframe_ctx_init failed: " + av::error2string(r));
        }
        width_ = width;
        height_ = height;
        output_format_ = format;
        logstream << "nvjpeg2k_dec frames"
                  << " size=" << width_ << "x" << height_
                  << " format=" << av_get_pix_fmt_name(output_format_);
    }

public:
    Nvjpeg2kDecoderNode(std::unique_ptr<Source<av::Packet>> &&source,
                        std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                        std::shared_ptr<HWAccelDevice> hwaccel,
                        int width,
                        int height,
                        AVPixelFormat output_format,
                        av::Rational timebase,
                        av::Rational framerate)
        : NodeSISO<av::Packet, av::VideoFrame>(std::move(source), std::move(sink)),
          hwaccel_(std::move(hwaccel)),
          width_(width),
          height_(height),
          output_format_(output_format),
          timebase_(timebase),
          framerate_(framerate) {
        this->auto_eof_ = false;
    }

    ~Nvjpeg2kDecoderNode() override {
        destroyDecoder();
        if (hw_frames_ctx_) av_buffer_unref(&hw_frames_ctx_);
    }

    int width() override { return width_; }
    int height() override { return height_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(output_format_); }
    av::Rational timeBase() override { return timebase_; }
    av::Rational frameRate() override { return framerate_; }

    void process() override {
        av::Packet pkt = this->source_->get();
        if (!pkt.isComplete()) {
            this->sink_->put(av::VideoFrame());
            finished_ = true;
            return;
        }
        if (!pkt.data() || !pkt.size()) return;

        ensureDecoder();
        checkNvjpeg2k(nvjpeg2kStreamParse(handle_, pkt.data(), pkt.size(), 0, 0, stream_), "stream parse");
        nvjpeg2kImageInfo_t info = {};
        checkNvjpeg2k(nvjpeg2kStreamGetImageInfo(stream_, &info), "image info");
        if (info.num_components != 3) {
            throw Error("nvjpeg2k_dec requires 3 components, got " + std::to_string(info.num_components));
        }

        nvjpeg2kImageComponentInfo_t comp[3] = {};
        for (int c = 0; c < 3; ++c) {
            checkNvjpeg2k(nvjpeg2kStreamGetImageComponentInfo(stream_, &comp[c], c), "component info");
        }
        if (comp[0].component_width != comp[1].component_width ||
            comp[0].component_width != comp[2].component_width ||
            comp[0].component_height != comp[1].component_height ||
            comp[0].component_height != comp[2].component_height) {
            throw Error("nvjpeg2k_dec supports 4:4:4 inputs only");
        }

        const int precision = std::max({comp[0].precision, comp[1].precision, comp[2].precision});
        ensureFrames(static_cast<int>(info.image_width), static_cast<int>(info.image_height),
                     formatForPrecision(precision));

        av::VideoFrame out;
        const int r = av_hwframe_get_buffer(hw_frames_ctx_, out.raw(), 0);
        if (r < 0) {
            throw Error("nvjpeg2k_dec av_hwframe_get_buffer failed: " + av::error2string(r));
        }

        nvjpeg2kImage_t img = {};
        for (int c = 0; c < 3; ++c) {
            img.pixel_data[c] = out.data(c);
            img.pitch_in_bytes[c] = static_cast<size_t>(out.raw()->linesize[c]);
        }
        img.pixel_type = precision > 8 ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;
        img.num_components = 3;

        auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(out.raw()->hw_frames_ctx->data);
        auto *cuda_ctx = reinterpret_cast<AVCUDADeviceContext *>(frames_ctx->device_ctx->hwctx);
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(cuda_ctx->stream);
        checkNvjpeg2k(nvjpeg2kDecode(handle_, decode_state_, stream_, &img, cuda_stream), "decode");
        checkCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");

        out.raw()->color_range = AVCOL_RANGE_JPEG;
        out.setTimeBase(pkt.timeBase());
        out.setPts(pkt.pts());
        out.setComplete(true);
        this->sink_->put(out);

        ++frames_seen_;
        if (frames_seen_ == 1) {
            logstream << "nvjpeg2k_dec first frame"
                      << " size=" << width_ << "x" << height_
                      << " precision=" << precision;
        }
    }

    static std::shared_ptr<Nvjpeg2kDecoderNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto src_edge = edges.find<av::Packet>(params["src"]);
        auto dst_edge = edges.find<av::VideoFrame>(params["dst"]);
        if (!params.count("hwaccel")) {
            throw Error("nvjpeg2k_dec requires hwaccel");
        }
        auto hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);

        auto video_md = src_edge->findNodeUp<IVideoFormatSource>();
        auto tb_md = src_edge->findNodeUp<ITimeBaseSource>();
        auto fr_md = src_edge->findNodeUp<IFrameRateSource>();
        if (!tb_md) throw Error("nvjpeg2k_dec requires upstream timebase metadata");

        int width = params.value("width", video_md ? video_md->width() : 0);
        int height = params.value("height", video_md ? video_md->height() : 0);
        if (width <= 0 || height <= 0) {
            throw Error("nvjpeg2k_dec needs width/height parameters or upstream video metadata");
        }
        av::PixelFormat real_pf = video_md ? video_md->realPixelFormat() : av::PixelFormat(AV_PIX_FMT_YUV444P);
        if (!isSupportedOutputFormat(real_pf.get())) {
            real_pf = av::PixelFormat(AV_PIX_FMT_YUV444P);
        }
        if (params.count("real_pixel_format")) {
            real_pf = av::PixelFormat(parsePixelFormat(params["real_pixel_format"].get<std::string>()));
            if (!isSupportedOutputFormat(real_pf.get())) {
                throw Error("nvjpeg2k_dec real_pixel_format must be yuv444p, yuv444p10le, "
                            "yuv444p12le or yuv444p16le");
            }
        }

        return std::make_shared<Nvjpeg2kDecoderNode>(
            src_edge->makeSource(), dst_edge->makeSink(), hwaccel, width, height, real_pf.get(),
            tb_md->timeBase(), fr_md ? fr_md->frameRate() : av::Rational{0, 0});
    }
};

DECLNODE(nvjpeg2k_dec, Nvjpeg2kDecoderNode)

#else

class Nvjpeg2kDecoderNode : public NodeSISO<av::Packet, av::VideoFrame>,
                            public IVideoFormatSource,
                            public ITimeBaseSource,
                            public IFrameRateSource,
                            public ReportsFinishByFlag {
public:
    Nvjpeg2kDecoderNode(std::unique_ptr<Source<av::Packet>> &&source,
                        std::unique_ptr<Sink<av::VideoFrame>> &&sink)
        : NodeSISO<av::Packet, av::VideoFrame>(std::move(source), std::move(sink)) {}
    void process() override {
        throw Error("nvjpeg2k_dec requested, but avplumber was built with HAVE_NVJPEG2K=0");
    }
    int width() override { return 0; }
    int height() override { return 0; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(AV_PIX_FMT_YUV444P); }
    av::Rational timeBase() override { return {0, 1}; }
    av::Rational frameRate() override { return {0, 1}; }
    static std::shared_ptr<Nvjpeg2kDecoderNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return std::make_shared<Nvjpeg2kDecoderNode>(
            edges.find<av::Packet>(params["src"])->makeSource(),
            edges.find<av::VideoFrame>(params["dst"])->makeSink());
    }
};

DECLNODE(nvjpeg2k_dec, Nvjpeg2kDecoderNode)

#endif
