#include "node_common.hpp"

#if HAVE_CUDA

#include <avcpp/codec.h>
#include <chrono>
#include <cstring>
#include <deque>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/mem.h>
}

#include <nvjpeg.h>

namespace {

std::string nvjpegStatusString(nvjpegStatus_t st) {
    switch (st) {
        case NVJPEG_STATUS_SUCCESS: return "success";
        case NVJPEG_STATUS_NOT_INITIALIZED: return "not initialized";
        case NVJPEG_STATUS_INVALID_PARAMETER: return "invalid parameter";
        case NVJPEG_STATUS_BAD_JPEG: return "bad jpeg";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "jpeg not supported";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "allocator failure";
        case NVJPEG_STATUS_EXECUTION_FAILED: return "execution failed";
        case NVJPEG_STATUS_ARCH_MISMATCH: return "arch mismatch";
        case NVJPEG_STATUS_INTERNAL_ERROR: return "internal error";
        case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "implementation not supported";
        case NVJPEG_STATUS_INCOMPLETE_BITSTREAM: return "incomplete bitstream";
        default: return "unknown";
    }
}

void checkNvjpeg(nvjpegStatus_t st, const char *what) {
    if (st != NVJPEG_STATUS_SUCCESS) {
        throw Error(std::string("nvjpeg_enc ") + what + " failed: " + nvjpegStatusString(st));
    }
}

void checkCuda(cudaError_t st, const char *what) {
    if (st != cudaSuccess) {
        throw Error(std::string("nvjpeg_enc ") + what + " failed: " + cudaGetErrorString(st));
    }
}

double elapsedMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

class NvjpegEncoderNode : public NodeSISO<av::VideoFrame, av::Packet>,
                          public IEncoder,
                          public IVideoFormatSource,
                          public ITimeBaseSource,
                          public IFrameRateSource,
                          public ReportsFinishByFlag {
    nvjpegHandle_t handle_ = nullptr;
    nvjpegEncoderState_t state_ = nullptr;
    nvjpegEncoderParams_t enc_params_ = nullptr;
    av::Codec codec_;
    AVCodecParameters *codecpar_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    av::Rational timebase_;
    av::Rational framerate_;
    int quality_ = 95;
    bool optimized_huffman_ = true;
    bool latency_measure_ = false;
    int latency_warmup_frames_ = 30;
    int latency_report_frames_ = 300;
    int64_t frames_seen_ = 0;
    int64_t bytes_seen_ = 0;
    std::vector<double> encode_ms_;

    void ensureInitialized(cudaStream_t stream) {
        if (handle_) return;
        checkNvjpeg(nvjpegCreateSimple(&handle_), "create");
        checkNvjpeg(nvjpegEncoderStateCreate(handle_, &state_, stream), "encoder state create");
        checkNvjpeg(nvjpegEncoderParamsCreate(handle_, &enc_params_, stream), "encoder params create");
        checkNvjpeg(nvjpegEncoderParamsSetQuality(enc_params_, quality_, stream), "set quality");
        checkNvjpeg(nvjpegEncoderParamsSetOptimizedHuffman(enc_params_, optimized_huffman_ ? 1 : 0, stream), "set optimized huffman");
        checkNvjpeg(nvjpegEncoderParamsSetSamplingFactors(enc_params_, NVJPEG_CSS_420, stream), "set sampling");
        logstream << "nvjpeg_enc initialized quality=" << quality_
                  << " optimized_huffman=" << optimized_huffman_
                  << " format=cuda/nv12 size=" << width_ << "x" << height_;
    }

    void destroyNvjpeg() {
        if (enc_params_) {
            nvjpegEncoderParamsDestroy(enc_params_);
            enc_params_ = nullptr;
        }
        if (state_) {
            nvjpegEncoderStateDestroy(state_);
            state_ = nullptr;
        }
        if (handle_) {
            nvjpegDestroy(handle_);
            handle_ = nullptr;
        }
    }

    void reportLatency() {
        if (!latency_measure_ || encode_ms_.empty()) return;
        auto sorted = encode_ms_;
        std::sort(sorted.begin(), sorted.end());
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double avg = sum / sorted.size();
        size_t p95_i = std::min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 95 / 100));
        double mbit = 0.0;
        if (frames_seen_ > 0) {
            double seconds = framerate_.getNumerator() > 0 && framerate_.getDenominator() > 0
                ? static_cast<double>(frames_seen_) * framerate_.getDenominator() / framerate_.getNumerator()
                : 0.0;
            if (seconds > 0.0) mbit = static_cast<double>(bytes_seen_) * 8.0 / seconds / 1000000.0;
        }
        logstream << "NVJPEG_ENCODER_LATENCY_SUMMARY samples=" << encode_ms_.size()
                  << " warmup_frames=" << latency_warmup_frames_
                  << " quality=" << quality_
                  << " encode_ms_min=" << sorted.front()
                  << " encode_ms_avg=" << avg
                  << " encode_ms_p95=" << sorted[p95_i]
                  << " encode_ms_max=" << sorted.back()
                  << " avg_mbit=" << mbit;
        encode_ms_.clear();
    }

    AVCUDADeviceContext *cudaDeviceContext(const av::VideoFrame &frm) {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->hw_frames_ctx || !raw->hw_frames_ctx->data) {
            throw Error("nvjpeg_enc requires CUDA AVFrame with hw_frames_ctx");
        }
        auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(raw->hw_frames_ctx->data);
        if (!frames_ctx || frames_ctx->format != AV_PIX_FMT_CUDA || frames_ctx->sw_format != AV_PIX_FMT_NV12) {
            std::ostringstream os;
            os << "nvjpeg_enc requires CUDA/NV12 frames, got hw="
               << (frames_ctx ? av_get_pix_fmt_name(frames_ctx->format) : "null")
               << " sw=" << (frames_ctx ? av_get_pix_fmt_name(frames_ctx->sw_format) : "null");
            throw Error(os.str());
        }
        if (!frames_ctx->device_ctx || !frames_ctx->device_ctx->hwctx) {
            throw Error("nvjpeg_enc missing CUDA device context");
        }
        auto *cuda_ctx = reinterpret_cast<AVCUDADeviceContext *>(frames_ctx->device_ctx->hwctx);
        if (!cuda_ctx || !cuda_ctx->cuda_ctx) {
            throw Error("nvjpeg_enc missing CUDA context");
        }
        return cuda_ctx;
    }

public:
    NvjpegEncoderNode(std::unique_ptr<Source<av::VideoFrame>> &&source,
                      std::unique_ptr<Sink<av::Packet>> &&sink,
                      int width,
                      int height,
                      av::Rational timebase,
                      av::Rational framerate)
        : NodeSISO<av::VideoFrame, av::Packet>(std::move(source), std::move(sink)),
          codec_(av::findEncodingCodec("mjpeg")),
          width_(width),
          height_(height),
          timebase_(timebase),
          framerate_(framerate) {
        codecpar_ = avcodec_parameters_alloc();
        if (!codecpar_) throw Error("nvjpeg_enc failed to allocate codec parameters");
        codecpar_->codec_type = AVMEDIA_TYPE_VIDEO;
        codecpar_->codec_id = AV_CODEC_ID_MJPEG;
        codecpar_->width = width_;
        codecpar_->height = height_;
        codecpar_->format = AV_PIX_FMT_YUVJ420P;
        codecpar_->color_range = AVCOL_RANGE_JPEG;
        this->auto_eof_ = false;
    }

    ~NvjpegEncoderNode() override {
        reportLatency();
        destroyNvjpeg();
        if (codecpar_) avcodec_parameters_free(&codecpar_);
    }

    av::Codec& encodingCodec() override { return codec_; }
    AVCodecParameters* codecParameters() override { return codecpar_; }

    void setOutput(av::Stream &stream, av::FormatContext &octx) override {
        if (!octx.outputFormat().codecSupported(codec_)) {
            throw Error(std::string("Codec ") + codec_.name() + " not supported by container " + octx.outputFormat().name());
        }
        stream.setTimeBase(timebase_);
        avcodec_parameters_copy(stream.raw()->codecpar, codecpar_);
    }

    int width() override { return width_; }
    int height() override { return height_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(AV_PIX_FMT_NV12); }
    av::Rational timeBase() override { return timebase_; }
    av::Rational frameRate() override { return framerate_; }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm.isComplete()) {
            reportLatency();
            this->sink_->put(createEofPacket());
            finished_ = true;
            return;
        }
        if (frm.pixelFormat().get() != AV_PIX_FMT_CUDA) {
            throw Error("nvjpeg_enc requires AV_PIX_FMT_CUDA input");
        }
        if (frm.width() != width_ || frm.height() != height_) {
            throw Error("nvjpeg_enc input size changed");
        }

        AVCUDADeviceContext *cuda_ctx = cudaDeviceContext(frm);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_ctx->stream);
        ensureInitialized(stream);

        nvjpegImage_t img = {};
        img.channel[0] = frm.data(0);
        img.pitch[0] = static_cast<size_t>(frm.raw()->linesize[0]);
        img.channel[1] = frm.data(1);
        img.pitch[1] = static_cast<size_t>(frm.raw()->linesize[1]);
        if (!img.channel[0] || !img.channel[1] || !img.pitch[0] || !img.pitch[1]) {
            throw Error("nvjpeg_enc missing NV12 Y/UV planes");
        }

        auto t0 = std::chrono::steady_clock::now();
        checkNvjpeg(nvjpegEncode(handle_, state_, enc_params_, &img, NVJPEG_CSS_420, NVJPEG_INPUT_NV12, width_, height_, stream), "encode");
        size_t jpeg_size = 0;
        checkNvjpeg(nvjpegEncodeRetrieveBitstream(handle_, state_, nullptr, &jpeg_size, stream), "retrieve size");
        uint8_t *jpeg = static_cast<uint8_t *>(av_malloc(jpeg_size + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!jpeg) throw Error("nvjpeg_enc failed to allocate AVPacket payload");
        checkNvjpeg(nvjpegEncodeRetrieveBitstream(handle_, state_, jpeg, &jpeg_size, stream), "retrieve bitstream");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        auto t1 = std::chrono::steady_clock::now();
        std::memset(jpeg + jpeg_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        av::Packet pkt(jpeg, jpeg_size, av::Packet::wrap_data{});
        pkt.setTimeBase(frm.timeBase());
        pkt.setPts(frm.pts());
        pkt.setDts(frm.pts());
        pkt.setKeyPacket(true);
        this->sink_->put(pkt);

        ++frames_seen_;
        bytes_seen_ += static_cast<int64_t>(jpeg_size);
        if (latency_measure_ && frames_seen_ > latency_warmup_frames_) {
            encode_ms_.push_back(elapsedMs(t0, t1));
            if (static_cast<int>(encode_ms_.size()) >= latency_report_frames_) {
                reportLatency();
            }
        }
    }

    static std::shared_ptr<NvjpegEncoderNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = edges.find<av::Packet>(params["dst"]);
        auto video_md = src_edge->findNodeUp<IVideoFormatSource>();
        auto tb_md = src_edge->findNodeUp<ITimeBaseSource>();
        auto fr_md = src_edge->findNodeUp<IFrameRateSource>();
        if (!video_md) throw Error("nvjpeg_enc requires upstream video metadata");
        if (!tb_md) throw Error("nvjpeg_enc requires upstream timebase metadata");
        int width = video_md->width();
        int height = video_md->height();
        auto pf = video_md->pixelFormat();
        auto real_pf = video_md->realPixelFormat();
        if (pf.get() != AV_PIX_FMT_CUDA || real_pf.get() != AV_PIX_FMT_NV12) {
            throw Error(std::string("nvjpeg_enc requires upstream pixel_format=cuda real_pixel_format=nv12, got ") +
                        pf.name() + "/" + real_pf.name());
        }
        auto node = std::make_shared<NvjpegEncoderNode>(
            src_edge->makeSource(), dst_edge->makeSink(), width, height, tb_md->timeBase(),
            fr_md ? fr_md->frameRate() : av::Rational{0, 0});
        if (params.count("quality")) node->quality_ = params["quality"];
        if (params.count("optimized_huffman")) node->optimized_huffman_ = params["optimized_huffman"];
        if (params.count("latency_measure")) node->latency_measure_ = params["latency_measure"];
        if (params.count("latency_warmup_frames")) node->latency_warmup_frames_ = params["latency_warmup_frames"];
        if (params.count("latency_report_frames")) node->latency_report_frames_ = params["latency_report_frames"];
        if (node->quality_ < 1 || node->quality_ > 100) {
            throw Error("nvjpeg_enc quality must be 1..100");
        }
        return node;
    }
};

DECLNODE(nvjpeg_enc, NvjpegEncoderNode);

#endif
