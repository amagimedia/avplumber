#include "node_common.hpp"

#if HAVE_CUDA && HAVE_NVJPEG2K

#include "nvjpeg2k_common.hpp"

#include <algorithm>
#include <avcpp/codec.h>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/mem.h>
}

namespace {

constexpr const char *kWho = "nvjpeg2k_enc";

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

struct AvFree {
    void operator()(uint8_t *p) const { av_free(p); }
};
/// An AVPacket payload until the packet that frees it exists.
using AvBuffer = std::unique_ptr<uint8_t, AvFree>;

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
    CUcontext cuda_ctx_ = nullptr;   // the context the encoder's resources were created in
    AVBufferRef *device_ref_ = nullptr;   // keeps that context alive until they are freed
    avp::nvjpeg2k::WorkStream work_;
    int width_ = 0;
    int height_ = 0;
    AVPixelFormat input_format_ = AV_PIX_FMT_YUV444P;
    int precision_ = 8;
    av::Rational timebase_;
    av::Rational framerate_;
    bool lossless_ = true;
    bool high_throughput_ = false;
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
            os << "nvjpeg2k_enc requires CUDA/" << avp::nvjpeg2k::pixFmtName(input_format_)
               << " frames, got hw="
               << (frames_ctx ? avp::nvjpeg2k::pixFmtName(frames_ctx->format) : "null")
               << " sw=" << (frames_ctx ? avp::nvjpeg2k::pixFmtName(frames_ctx->sw_format) : "null");
            throw Error(os.str());
        }
        if (!frames_ctx->device_ctx || !frames_ctx->device_ctx->hwctx) {
            throw Error("nvjpeg2k_enc missing CUDA device context");
        }
        auto *cuda_ctx = reinterpret_cast<AVCUDADeviceContext *>(frames_ctx->device_ctx->hwctx);
        if (!cuda_ctx || !cuda_ctx->cuda_ctx) {
            throw Error("nvjpeg2k_enc missing CUDA context");
        }
        if (!device_ref_) {
            // Hold the device: the encoder's own GPU resources live in its CUDA context and are
            // freed by the destructor, which runs long after the frames that brought us here are
            // gone. Nothing else keeps the device for us, and when the context is destroyed first
            // the SDK's teardown fails from inside a destructor.
            device_ref_ = av_buffer_ref(frames_ctx->device_ref);
            if (!device_ref_) throw Error("nvjpeg2k_enc cannot reference the CUDA device");
        }
        return cuda_ctx;
    }

    void ensureInitialized() {
        if (enc_handle_) return;
        avp::nvjpeg2k::check(nvjpeg2kEncoderCreateSimple(&enc_handle_), kWho, "encoder create");
        avp::nvjpeg2k::check(nvjpeg2kEncodeStateCreate(enc_handle_, &enc_state_), kWho,
                             "encoder state create");
        avp::nvjpeg2k::check(nvjpeg2kEncodeParamsCreate(&enc_params_), kWho, "encoder params create");

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
        if (high_throughput_) {
            // HTJ2K (ISO/IEC 15444-15): the same wavelet and codestream structure with the fast
            // block coder instead of Part 1's arithmetic one. Only an HT-capable decoder reads it
            // -- nvjpeg2k_dec takes it from the codestream, a Part 1 decoder cannot read it at all.
            enc_config_.rsiz = NVJPEG2K_RSIZ_HT;
            enc_config_.encode_modes = NVJPEG2K_MODE_HT;
        }

        avp::nvjpeg2k::check(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params_, &enc_config_), kWho,
                             "set encode config");
        if (!lossless_) {
            avp::nvjpeg2k::check(nvjpeg2kEncodeParamsSpecifyQuality(
                                     enc_params_, NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, psnr_),
                                 kWho, "set quality");
        }
        logstream << "nvjpeg2k_enc initialized"
                  << " mode=" << (lossless_ ? "lossless" : "lossy")
                  << " psnr=" << (lossless_ ? 0.0 : psnr_)
                  << " coder=" << (high_throughput_ ? "ht" : "part1")
                  << " stream=" << (use_jp2_ ? "jp2" : "j2k")
                  << " code_block=" << code_block_w_ << "x" << code_block_h_
                  << " format=cuda/" << avp::nvjpeg2k::pixFmtName(input_format_)
                  << " precision=" << precision_
                  << " resolutions=" << enc_config_.num_resolutions
                  << " size=" << width_ << "x" << height_;
    }

    /// Frees the SDK's resources with the context they were created in current. Reports a failure
    /// rather than propagating it: this runs from the destructor, and nvjpeg2k throws an exception
    /// type of its own (not derived from std::exception) that must not escape one.
    void destroyNvjpeg2k() noexcept {
        try {
            std::optional<avp::nvjpeg2k::ContextGuard> ctx;
            if (cuda_ctx_) ctx.emplace(cuda_ctx_, kWho);
            work_.destroy();   // its stream and event belong to this context too
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
        } catch (const std::exception &e) {
            logstream << "nvjpeg2k_enc: " << e.what() << "; the encoder may not be fully freed";
        } catch (...) {
            logstream << "nvjpeg2k_enc: nvjpeg2k threw while freeing the encoder";
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
          precision_(avp::nvjpeg2k::componentPrecision(input_format, kWho)),
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
        destroyNvjpeg2k();   // before the device reference that keeps its CUDA context alive
        if (device_ref_) av_buffer_unref(&device_ref_);
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
        if (!frm.isComplete()) {   // the frame EOF marker is an empty frame, so this catches it
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

        AvBuffer j2k;
        size_t j2k_size = 0;
        {
            avp::nvjpeg2k::ContextGuard ctx(cuda_ctx->cuda_ctx, kWho);
            cuda_ctx_ = cuda_ctx->cuda_ctx;   // recorded first: the destructor frees in it
            ensureInitialized();
            // Read the frame only once the work that produced it has finished, then let the host
            // wait for our own encode instead of for the whole device.
            work_.ensure(kWho);
            work_.waitFor(cuda_ctx->stream, kWho);
            const cudaStream_t stream = work_.get();

            // nvjpeg2kImage_t points at the caller's per-component arrays; it holds none itself.
            void *planes[3] = {};
            size_t pitches[3] = {};
            for (int c = 0; c < 3; ++c) {
                planes[c] = frm.data(c);
                pitches[c] = static_cast<size_t>(frm.raw()->linesize[c]);
                if (!planes[c] || !pitches[c]) {
                    throw Error(std::string("nvjpeg2k_enc missing plane ") + std::to_string(c));
                }
            }
            nvjpeg2kImage_t img = {};
            img.pixel_data = planes;
            img.pitch_in_bytes = pitches;
            img.pixel_type = precision_ > 8 ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;
            img.num_components = 3;

            avp::nvjpeg2k::check(nvjpeg2kEncode(enc_handle_, enc_state_, enc_params_, &img, stream),
                                 kWho, "encode");
            avp::nvjpeg2k::check(nvjpeg2kEncodeRetrieveBitstream(
                                     enc_handle_, enc_state_, nullptr, &j2k_size, stream),
                                 kWho, "retrieve size");
            j2k.reset(static_cast<uint8_t *>(av_malloc(j2k_size + AV_INPUT_BUFFER_PADDING_SIZE)));
            if (!j2k) throw Error("nvjpeg2k_enc failed to allocate AVPacket payload");
            avp::nvjpeg2k::check(nvjpeg2kEncodeRetrieveBitstream(
                                     enc_handle_, enc_state_, j2k.get(), &j2k_size, stream),
                                 kWho, "retrieve bitstream");
            work_.synchronize(kWho);
        }
        std::memset(j2k.get() + j2k_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        // The packet owns the payload from here; anything that threw above freed it instead.
        av::Packet pkt(j2k.get(), j2k_size, av::Packet::wrap_data{});
        j2k.release();
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
            real_pf = av::PixelFormat(avp::nvjpeg2k::parsePixelFormat(
                params["real_pixel_format"].get<std::string>(), kWho));
        }
        if (video_md->pixelFormat().get() != AV_PIX_FMT_CUDA) {
            throw Error("nvjpeg2k_enc requires upstream pixel_format=cuda");
        }
        const AVPixelFormat input_format = real_pf.get();
        avp::nvjpeg2k::componentPrecision(input_format, kWho); // validate before the node starts

        auto node = std::make_shared<Nvjpeg2kEncoderNode>(
            src_edge->makeSource(), dst_edge->makeSink(), width, height, input_format,
            tb_md->timeBase(), fr_md ? fr_md->frameRate() : av::Rational{0, 0});
        if (params.count("lossless")) node->lossless_ = params["lossless"];
        if (params.count("high_throughput")) node->high_throughput_ = params["high_throughput"];
        if (params.count("psnr")) node->psnr_ = params["psnr"];
        if (params.count("num_resolutions")) node->requested_resolutions_ = params["num_resolutions"];
        if (params.count("code_block_w")) node->code_block_w_ = params["code_block_w"];
        if (params.count("code_block_h")) node->code_block_h_ = params["code_block_h"];
        if (params.count("stream_type")) {
            const std::string st = params["stream_type"];
            if (st != "j2k" && st != "jp2") throw Error("nvjpeg2k_enc stream_type must be j2k or jp2");
            node->use_jp2_ = st == "jp2";
        }
        // 32 and 64 are the only widths this node can reach. The SDK also lists 128, but takes it
        // only on the HT coder with NLT type 3, which carries signed-magnitude float samples
        // (FP16/FP32) and not the integer ones encoded here; it accepts such a config and then
        // fails the first encode with "jpeg2000 not supported". Rejecting it here keeps that a
        // node.add error rather than one that surfaces after group.start. Within 32 and 64 the
        // spec's 4096-sample code-block limit (xcb + ycb <= 12) cannot be exceeded.
        if (node->code_block_w_ != 32 && node->code_block_w_ != 64) {
            throw Error("nvjpeg2k_enc code_block_w must be 32 or 64; nvJPEG2000 takes 128 only for "
                        "HT with NLT type 3 float samples, which this node does not encode");
        }
        if (node->code_block_h_ != 32 && node->code_block_h_ != 64) {
            throw Error("nvjpeg2k_enc code_block_h must be 32 or 64");
        }
        if (!node->lossless_ && node->psnr_ <= 0.0) {
            throw Error("nvjpeg2k_enc lossless=false needs a positive psnr");
        }
        // nvJPEG2000 0.12 rejects nvjpeg2kEncodeParamsSpecifyQuality() on an HT codestream, so
        // the two cannot be combined. Caught here rather than as a "set quality failed: invalid
        // parameter" on the first frame, after group.start.
        if (!node->lossless_ && node->high_throughput_) {
            throw Error("nvjpeg2k_enc high_throughput=true needs lossless=true; nvJPEG2000 has no "
                        "target-PSNR rate control for HT codestreams");
        }
        return node;
    }
};

DECLNODE(nvjpeg2k_enc, Nvjpeg2kEncoderNode)

#endif
