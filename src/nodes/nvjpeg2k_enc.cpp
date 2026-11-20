#include "node_common.hpp"

#if HAVE_CUDA && HAVE_NVJPEG2K

#include <algorithm>
#include <avcpp/codec.h>
#include <chrono>
#include <cstring>
#include <sstream>
#include <vector>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/mem.h>
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
        throw Error(std::string("nvjpeg2k_enc ") + what + " failed: " + nvjpeg2kStatusString(st));
    }
}

void checkCuda(cudaError_t st, const char *what) {
    if (st != cudaSuccess) {
        throw Error(std::string("nvjpeg2k_enc ") + what + " failed: " + cudaGetErrorString(st));
    }
}

int componentPrecision(AVPixelFormat fmt) {
    switch (fmt) {
        case AV_PIX_FMT_YUV444P: return 8;
        case AV_PIX_FMT_YUV444P10LE: return 10;
        case AV_PIX_FMT_YUV444P12LE: return 12;
        case AV_PIX_FMT_YUV444P16LE: return 16;
        default:
            throw Error(std::string("nvjpeg2k_enc unsupported pixel format: ") + av_get_pix_fmt_name(fmt));
    }
}

int chooseResolutions(int width, int height, int requested) {
    int min_dim = std::min(width, height);
    int maximum = 1;
    while (min_dim > 1 && maximum < NVJPEG2K_MAXRES) {
        min_dim >>= 1;
        ++maximum;
    }
    int result = requested > 0 ? requested : std::min(6, maximum);
    return std::max(1, std::min(result, maximum));
}

AVPixelFormat parsePixelFormat(const std::string &name) {
    const AVPixelFormat fmt = av_get_pix_fmt(name.c_str());
    if (fmt == AV_PIX_FMT_NONE) {
        throw Error("nvjpeg2k_enc unknown real_pixel_format: " + name);
    }
    return fmt;
}

} // namespace

class Nvjpeg2kEncoderNode : public NodeSISO<av::VideoFrame, av::Packet>,
                            public IEncoder,
                            public IVideoFormatSource,
                            public ITimeBaseSource,
                            public IFrameRateSource,
                            public ReportsFinishByFlag {
    nvjpeg2kEncoder_t enc_handle_ = nullptr;
    nvjpeg2kEncodeState_t enc_state_ = nullptr;
    nvjpeg2kEncodeParams_t enc_params_ = nullptr;
    nvjpeg2kEncodeConfig_t enc_config_ = {};
    nvjpeg2kImageComponentInfo_t comp_info_[3] = {};
    av::Codec codec_;
    AVCodecParameters *codecpar_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    AVPixelFormat input_format_ = AV_PIX_FMT_YUV444P;
    int precision_ = 8;
    av::Rational timebase_;
    av::Rational framerate_;
    bool lossless_ = true;
    double psnr_ = 50.0;
    int requested_resolutions_ = 6;
    int code_block_w_ = 64;
    int code_block_h_ = 64;
    bool use_jp2_ = false;
    int64_t frames_seen_ = 0;
    int64_t bytes_seen_ = 0;

    AVCUDADeviceContext *cudaDeviceContext(const av::VideoFrame &frm) {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->hw_frames_ctx || !raw->hw_frames_ctx->data) {
            throw Error("nvjpeg2k_enc requires a CUDA frame with hw_frames_ctx");
        }
        auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(raw->hw_frames_ctx->data);
        if (!frames_ctx || frames_ctx->format != AV_PIX_FMT_CUDA || frames_ctx->sw_format != input_format_) {
            std::ostringstream os;
            os << "nvjpeg2k_enc requires CUDA/" << av_get_pix_fmt_name(input_format_) << " frames, got hw="
               << (frames_ctx ? av_get_pix_fmt_name(frames_ctx->format) : "null")
               << " sw=" << (frames_ctx ? av_get_pix_fmt_name(frames_ctx->sw_format) : "null");
            throw Error(os.str());
        }
        if (!frames_ctx->device_ctx || !frames_ctx->device_ctx->hwctx) {
            throw Error("nvjpeg2k_enc missing CUDA device context");
        }
        auto *cuda_ctx = reinterpret_cast<AVCUDADeviceContext *>(frames_ctx->device_ctx->hwctx);
        if (!cuda_ctx || !cuda_ctx->cuda_ctx) {
            throw Error("nvjpeg2k_enc missing CUDA context");
        }
        return cuda_ctx;
    }

    void ensureInitialized() {
        if (enc_handle_) return;
        checkNvjpeg2k(nvjpeg2kEncoderCreateSimple(&enc_handle_), "encoder create");
        checkNvjpeg2k(nvjpeg2kEncodeStateCreate(enc_handle_, &enc_state_), "encoder state create");
        checkNvjpeg2k(nvjpeg2kEncodeParamsCreate(&enc_params_), "encoder params create");

        for (int c = 0; c < 3; ++c) {
            comp_info_[c] = {};
            comp_info_[c].component_width = static_cast<uint32_t>(width_);
            comp_info_[c].component_height = static_cast<uint32_t>(height_);
            comp_info_[c].precision = static_cast<uint8_t>(precision_);
            comp_info_[c].sgn = 0;
        }

        enc_config_ = {};
        enc_config_.stream_type = use_jp2_ ? NVJPEG2K_STREAM_JP2 : NVJPEG2K_STREAM_J2K;
        enc_config_.color_space = NVJPEG2K_COLORSPACE_SYCC;
        enc_config_.image_width = static_cast<uint32_t>(width_);
        enc_config_.image_height = static_cast<uint32_t>(height_);
        enc_config_.num_components = 3;
        enc_config_.image_comp_info = comp_info_;
        enc_config_.prog_order = NVJPEG2K_LRCP;
        enc_config_.num_layers = 1;
        enc_config_.mct_mode = 0;
        enc_config_.num_resolutions = static_cast<uint32_t>(
            chooseResolutions(width_, height_, requested_resolutions_));
        enc_config_.code_block_w = static_cast<uint32_t>(code_block_w_);
        enc_config_.code_block_h = static_cast<uint32_t>(code_block_h_);
        enc_config_.irreversible = lossless_ ? 0 : 1;

        checkNvjpeg2k(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params_, &enc_config_), "set encode config");
        if (!lossless_) {
            checkNvjpeg2k(nvjpeg2kEncodeParamsSpecifyQuality(
                              enc_params_, NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, psnr_),
                          "set quality");
        }
        logstream << "nvjpeg2k_enc initialized"
                  << " mode=" << (lossless_ ? "lossless" : "lossy")
                  << " psnr=" << (lossless_ ? 0.0 : psnr_)
                  << " stream=" << (use_jp2_ ? "jp2" : "j2k")
                  << " format=cuda/" << av_get_pix_fmt_name(input_format_)
                  << " precision=" << precision_
                  << " resolutions=" << enc_config_.num_resolutions
                  << " size=" << width_ << "x" << height_;
    }

    void destroyNvjpeg2k() {
        if (enc_params_) {
            nvjpeg2kEncodeParamsDestroy(enc_params_);
            enc_params_ = nullptr;
        }
        if (enc_state_) {
            nvjpeg2kEncodeStateDestroy(enc_state_);
            enc_state_ = nullptr;
        }
        if (enc_handle_) {
            nvjpeg2kEncoderDestroy(enc_handle_);
            enc_handle_ = nullptr;
        }
    }

public:
    Nvjpeg2kEncoderNode(std::unique_ptr<Source<av::VideoFrame>> &&source,
                        std::unique_ptr<Sink<av::Packet>> &&sink,
                        int width,
                        int height,
                        AVPixelFormat input_format,
                        av::Rational timebase,
                        av::Rational framerate)
        : NodeSISO<av::VideoFrame, av::Packet>(std::move(source), std::move(sink)),
          codec_(av::findEncodingCodec("jpeg2000")),
          width_(width),
          height_(height),
          input_format_(input_format),
          precision_(componentPrecision(input_format)),
          timebase_(timebase),
          framerate_(framerate) {
        codecpar_ = avcodec_parameters_alloc();
        if (!codecpar_) throw Error("nvjpeg2k_enc failed to allocate codec parameters");
        codecpar_->codec_type = AVMEDIA_TYPE_VIDEO;
        codecpar_->codec_id = AV_CODEC_ID_JPEG2000;
        codecpar_->width = width_;
        codecpar_->height = height_;
        codecpar_->format = input_format_;
        this->auto_eof_ = false;
    }

    ~Nvjpeg2kEncoderNode() override {
        destroyNvjpeg2k();
        if (codecpar_) avcodec_parameters_free(&codecpar_);
    }

    av::Codec& encodingCodec() override { return codec_; }
    AVCodecParameters* codecParameters() override { return codecpar_; }

    void setOutput(av::Stream &stream, av::FormatContext &octx) override {
        if (codec_.isNull()) {
            throw Error("nvjpeg2k_enc has no libavcodec JPEG2000 encoder; use it with fabric/packet outputs");
        }
        if (!octx.outputFormat().codecSupported(codec_)) {
            throw Error(std::string("Codec ") + codec_.name() + " not supported by container " +
                        octx.outputFormat().name());
        }
        stream.setTimeBase(timebase_);
        avcodec_parameters_copy(stream.raw()->codecpar, codecpar_);
    }

    int width() override { return width_; }
    int height() override { return height_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(input_format_); }
    av::Rational timeBase() override { return timebase_; }
    av::Rational frameRate() override { return framerate_; }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm.isComplete()) {
            this->sink_->put(createEofPacket());
            finished_ = true;
            return;
        }
        if (frm.pixelFormat().get() != AV_PIX_FMT_CUDA) {
            throw Error("nvjpeg2k_enc requires AV_PIX_FMT_CUDA input");
        }
        if (frm.width() != width_ || frm.height() != height_) {
            throw Error("nvjpeg2k_enc input size changed");
        }

        AVCUDADeviceContext *cuda_ctx = cudaDeviceContext(frm);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_ctx->stream);
        ensureInitialized();

        nvjpeg2kImage_t img = {};
        for (int c = 0; c < 3; ++c) {
            img.pixel_data[c] = frm.data(c);
            img.pitch_in_bytes[c] = static_cast<size_t>(frm.raw()->linesize[c]);
            if (!img.pixel_data[c] || !img.pitch_in_bytes[c]) {
                throw Error(std::string("nvjpeg2k_enc missing plane ") + std::to_string(c));
            }
        }
        img.pixel_type = precision_ > 8 ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;
        img.num_components = 3;

        checkNvjpeg2k(nvjpeg2kEncode(enc_handle_, enc_state_, enc_params_, &img, stream), "encode");
        size_t j2k_size = 0;
        checkNvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(
                          enc_handle_, enc_state_, nullptr, &j2k_size, stream),
                      "retrieve size");
        uint8_t *j2k = static_cast<uint8_t *>(av_malloc(j2k_size + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!j2k) throw Error("nvjpeg2k_enc failed to allocate AVPacket payload");
        checkNvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(
                          enc_handle_, enc_state_, j2k, &j2k_size, stream),
                      "retrieve bitstream");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        std::memset(j2k + j2k_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        av::Packet pkt(j2k, j2k_size, av::Packet::wrap_data{});
        pkt.setTimeBase(frm.timeBase());
        pkt.setPts(frm.pts());
        pkt.setDts(frm.pts());
        pkt.setKeyPacket(true);
        this->sink_->put(pkt);

        ++frames_seen_;
        bytes_seen_ += static_cast<int64_t>(j2k_size);
        if (frames_seen_ == 1) {
            logstream << "nvjpeg2k_enc first frame bytes=" << j2k_size;
        }
    }

    static std::shared_ptr<Nvjpeg2kEncoderNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = edges.find<av::Packet>(params["dst"]);
        auto video_md = src_edge->findNodeUp<IVideoFormatSource>();
        auto tb_md = src_edge->findNodeUp<ITimeBaseSource>();
        auto fr_md = src_edge->findNodeUp<IFrameRateSource>();
        if (!video_md) throw Error("nvjpeg2k_enc requires upstream video metadata");
        if (!tb_md) throw Error("nvjpeg2k_enc requires upstream timebase metadata");

        int width = params.value("width", video_md->width());
        int height = params.value("height", video_md->height());
        av::PixelFormat real_pf = video_md->realPixelFormat();
        if (params.count("real_pixel_format")) {
            real_pf = av::PixelFormat(parsePixelFormat(params["real_pixel_format"].get<std::string>()));
        }
        if (video_md->pixelFormat().get() != AV_PIX_FMT_CUDA) {
            throw Error("nvjpeg2k_enc requires upstream pixel_format=cuda");
        }
        const AVPixelFormat input_format = real_pf.get();
        componentPrecision(input_format); // validate before the node starts

        auto node = std::make_shared<Nvjpeg2kEncoderNode>(
            src_edge->makeSource(), dst_edge->makeSink(), width, height, input_format,
            tb_md->timeBase(), fr_md ? fr_md->frameRate() : av::Rational{0, 0});
        if (params.count("lossless")) node->lossless_ = params["lossless"];
        if (params.count("psnr")) node->psnr_ = params["psnr"];
        if (params.count("num_resolutions")) node->requested_resolutions_ = params["num_resolutions"];
        if (params.count("code_block_w")) node->code_block_w_ = params["code_block_w"];
        if (params.count("code_block_h")) node->code_block_h_ = params["code_block_h"];
        if (params.count("stream_type")) {
            const std::string st = params["stream_type"];
            if (st != "j2k" && st != "jp2") throw Error("nvjpeg2k_enc stream_type must be j2k or jp2");
            node->use_jp2_ = st == "jp2";
        }
        if (node->code_block_w_ != 32 && node->code_block_w_ != 64 && node->code_block_w_ != 128) {
            throw Error("nvjpeg2k_enc code_block_w must be 32, 64 or 128");
        }
        if (node->code_block_h_ != 32 && node->code_block_h_ != 64) {
            throw Error("nvjpeg2k_enc code_block_h must be 32 or 64");
        }
        if (!node->lossless_ && node->psnr_ <= 0.0) {
            throw Error("nvjpeg2k_enc lossless=false needs a positive psnr");
        }
        return node;
    }
};

DECLNODE(nvjpeg2k_enc, Nvjpeg2kEncoderNode)

#else

class Nvjpeg2kEncoderNode : public NodeSISO<av::VideoFrame, av::Packet>,
                            public IEncoder,
                            public IVideoFormatSource,
                            public ITimeBaseSource,
                            public IFrameRateSource,
                            public ReportsFinishByFlag {
public:
    Nvjpeg2kEncoderNode(std::unique_ptr<Source<av::VideoFrame>> &&source,
                        std::unique_ptr<Sink<av::Packet>> &&sink)
        : NodeSISO<av::VideoFrame, av::Packet>(std::move(source), std::move(sink)) {}
    void process() override {
        throw Error("nvjpeg2k_enc requested, but avplumber was built with HAVE_NVJPEG2K=0");
    }
    av::Codec& encodingCodec() override { throw Error("nvjpeg2k_enc unavailable"); }
    AVCodecParameters* codecParameters() override { return nullptr; }
    void setOutput(av::Stream&, av::FormatContext&) override {}
    int width() override { return 0; }
    int height() override { return 0; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(AV_PIX_FMT_YUV444P); }
    av::Rational timeBase() override { return {0, 1}; }
    av::Rational frameRate() override { return {0, 1}; }
    static std::shared_ptr<Nvjpeg2kEncoderNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return std::make_shared<Nvjpeg2kEncoderNode>(
            edges.find<av::VideoFrame>(params["src"])->makeSource(),
            edges.find<av::Packet>(params["dst"])->makeSink());
    }
};

DECLNODE(nvjpeg2k_enc, Nvjpeg2kEncoderNode)

#endif
