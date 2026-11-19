#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../../objs/src/nodes/hwaccel/v210_pack.ptx.h"

#include <avcpp/codec.h>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}

#include <cstring>
#include <limits>

namespace {

AVCUDADeviceContext* cuda(AVBufferRef* device) {
    return static_cast<AVCUDADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(device->data)->hwctx);
}

// Never throws, because libavutil's buffer callbacks use it too: check `ok`.
struct ContextScope {
    const bool ok;
    explicit ContextScope(AVBufferRef* device) : ok(!CHECK_CU(cuCtxPushCurrent(cuda(device)->cuda_ctx))) {}
    ~ContextScope() {
        CUcontext previous;
        if (ok) CHECK_CU(cuCtxPopCurrent(&previous));
    }
};

// Download buffers outlive the node: a packet can still sit in a queue, or in
// the muxer, when the graph is torn down. The pool's opaque is therefore its
// own reference to the CUDA device, released after the last buffer, so
// cuMemFreeHost always runs against a live context.
void freePinned(void* device, uint8_t* data) {
    if (ContextScope scope(static_cast<AVBufferRef*>(device)); scope.ok) CHECK_CU(cuMemFreeHost(data));
}

AVBufferRef* allocPinned(void* device, size_t size) {
    void* host = nullptr;
    if (ContextScope scope(static_cast<AVBufferRef*>(device)); !scope.ok || CHECK_CU(cuMemHostAlloc(&host, size, 0)))
        return nullptr;
    auto* data = static_cast<uint8_t*>(host);
    // Only the payload is written per frame; zero the AVPacket padding once.
    std::memset(data + size - AV_INPUT_BUFFER_PADDING_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    AVBufferRef* buffer = av_buffer_create(data, size, freePinned, device, 0);
    if (!buffer) freePinned(device, data);
    return buffer;
}

// Everything the pack holds on one device, released in one place so that a
// failed setup leaves nothing behind.
struct Gpu {
    AVBufferRef* device = nullptr; // Keeps the CUDA context alive.
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr packed = 0;
    AVBufferPool* pool = nullptr;
    // The pack's own stream: on the device's stream (the legacy NULL stream in
    // the mixer) the download would delay every later kernel of the context.
    CUstream stream = nullptr;
    CUevent produced = nullptr;

    ~Gpu() {
        // Buffers still referenced by packets survive this; the pool waits.
        av_buffer_pool_uninit(&pool);
        if (!device) return;
        if (ContextScope scope(device); scope.ok) {
            if (packed) CHECK_CU(cuMemFree(packed));
            if (module) CHECK_CU(cuModuleUnload(module));
            if (produced) CHECK_CU(cuEventDestroy(produced));
            if (stream) CHECK_CU(cuStreamDestroy(stream));
        }
        av_buffer_unref(&device);
    }
};

bool positive(const av::Rational& ratio) {
    return ratio.getNumerator() > 0 && ratio.getDenominator() > 0;
}

} // namespace

// The mirror of FFmpeg's hwupload_cuda=v210_width unpack: packs CUDA 10-bit
// 4:2:2 frames into v210 packets with a PTX kernel and DMAs each one straight
// into the buffer the muxer will read, which replaces both a hwdownload and
// the libavcodec v210 encoder.
//
// It implements IEncoder without an encoder: v210 is a memory layout, so the
// muxer only needs codec parameters and a stream rate, and those are known at
// node creation — which is when the output node opens the format context.
class CudaToV210 : public NodeSISO<av::VideoFrame, av::Packet>, public IEncoder,
                   public ReportsFinishByFlag {
    // Packed v210 carries no header, so a frame is exactly stride * height bytes.
    // The kernel reads these as int arguments.
    int width_, height_, stride_;
    av::Rational fps_, timebase_;
    av::Codec codec_ = av::findEncodingCodec("v210");
    AVCodecParameters* codecpar_ = avcodec_parameters_alloc();
    std::unique_ptr<Gpu> gpu_;

    size_t size() const { return size_t(stride_) * height_; }

    // Frames may arrive from another device at any time, so this runs per frame.
    void useDevice(AVBufferRef* device) {
        if (gpu_ && gpu_->device->data == device->data) return;
        auto gpu = std::make_unique<Gpu>();
        if (!(gpu->device = av_buffer_ref(device))) throw Error("cuda_to_v210: cannot reference the CUDA device");
        AVBufferRef* pool_device = av_buffer_ref(device);
        if (pool_device)
            gpu->pool = av_buffer_pool_init2(size() + AV_INPUT_BUFFER_PADDING_SIZE, pool_device, allocPinned, [](void* opaque) {
                auto* owned = static_cast<AVBufferRef*>(opaque);
                av_buffer_unref(&owned);
            });
        if (!gpu->pool) {
            av_buffer_unref(&pool_device);
            throw Error("cuda_to_v210: cannot create the download buffer pool");
        }
        const std::string image(avpl_v210_pack_ptx, avpl_v210_pack_ptx + avpl_v210_pack_ptx_len);
        // The kernel writes whole words and never the row padding, so zeroing
        // once keeps that padding defined for every frame.
        if (ContextScope scope(device); !scope.ok ||
            CHECK_CU(cuMemAlloc(&gpu->packed, size())) || CHECK_CU(cuMemsetD8(gpu->packed, 0, size())) ||
            CHECK_CU(cuModuleLoadDataEx(&gpu->module, image.c_str(), 0, nullptr, nullptr)) ||
            CHECK_CU(cuModuleGetFunction(&gpu->kernel, gpu->module, "pack_v210")) ||
            // The bundled dynamic-loader header omits CU_STREAM_NON_BLOCKING (0x1).
            CHECK_CU(cuStreamCreate(&gpu->stream, 0x1)) ||
            CHECK_CU(cuEventCreate(&gpu->produced, CU_EVENT_DISABLE_TIMING)))
            throw Error("cuda_to_v210: cannot set up the pack on the CUDA device");
        gpu_ = std::move(gpu);
    }

    av::Packet pack(av::VideoFrame& frame, int semiplanar) {
        av::Packet packet;
        AVPacket* raw = packet.raw();
        // The packet owns the pinned buffer from here, including on failure.
        if (!(raw->buf = av_buffer_pool_get(gpu_->pool)))
            throw Error("cuda_to_v210: cannot allocate a pinned download buffer");
        raw->data = raw->buf->data;
        raw->size = static_cast<int>(size());

        AVFrame* input = frame.raw();
        void* args[] = {&gpu_->packed, &stride_, &width_, &height_,
                        &input->data[0], &input->linesize[0],
                        &input->data[1], &input->linesize[1],
                        &input->data[2], &input->linesize[2], &semiplanar};
        const int blocks = (width_ + 5) / 6;
        // Whatever produced this frame queued it on the device context's
        // stream: the pack waits for that point only, on its own stream.
        const CUstream stream = gpu_->stream;
        ContextScope scope(gpu_->device);
        if (!scope.ok) throw Error("cuda_to_v210: cannot enter the CUDA context");
        // Straight into the packet's payload: the muxer's copy into the grain
        // is the only time a CPU touches these bytes.
        const bool queued =
            !CHECK_CU(cuEventRecord(gpu_->produced, cuda(gpu_->device)->stream)) &&
            !CHECK_CU(cuStreamWaitEvent(stream, gpu_->produced, 0)) &&
            !CHECK_CU(cuLaunchKernel(gpu_->kernel, (blocks + 31) / 32, (height_ + 7) / 8, 1, 32, 8, 1, 0,
                                     stream, args, nullptr)) &&
            !CHECK_CU(cuMemcpyDtoHAsync(raw->data, gpu_->packed, size(), stream));
        // Also after a failure: the buffer must not be recycled while a queued operation writes it.
        if (CHECK_CU(cuStreamSynchronize(stream)) || !queued) throw Error("cuda_to_v210: cannot pack the frame");

        raw->duration = av_rescale_q(input->duration, frame.timeBase(), timebase_);
        packet.setTimeBase(timebase_);
        packet.setPts(frame.pts());
        packet.setDts(frame.pts()); // Uncompressed: no reordering, every frame a key frame.
        packet.setKeyPacket(true);
        packet.setComplete(true);
        return packet;
    }

public:
    CudaToV210(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
               int width, int height, int stride, av::Rational fps, av::Rational timebase)
        : NodeSISO(std::move(source), std::move(sink)), width_(width), height_(height), stride_(stride),
          fps_(fps), timebase_(timebase) {
        if (!codecpar_) throw Error("cuda_to_v210: cannot allocate codec parameters");
        codecpar_->codec_type = AVMEDIA_TYPE_VIDEO;
        codecpar_->codec_id = AV_CODEC_ID_V210;
        codecpar_->codec_tag = MKTAG('v', '2', '1', '0');
        codecpar_->width = width_;
        codecpar_->height = height_;
        // The layout libavcodec's v210 decoder produces, which is what a reader
        // of these packets gets; the packets themselves are packed, not planar.
        codecpar_->format = AV_PIX_FMT_YUV422P10LE;
        codecpar_->bits_per_coded_sample = 20;
        codecpar_->bits_per_raw_sample = 10;
        codecpar_->field_order = AV_FIELD_PROGRESSIVE;
        // EOF has to reach the muxer as a packet, so this node consumes the
        // marker itself instead of letting the framework swallow it.
        this->auto_eof_ = false;
    }

    ~CudaToV210() override { avcodec_parameters_free(&codecpar_); }

    av::Codec& encodingCodec() override { return codec_; }
    AVCodecParameters* codecParameters() override { return codecpar_; }

    void setOutput(av::Stream& stream, av::FormatContext& octx) override {
        if (!codec_.isNull() && !octx.outputFormat().codecSupported(codec_))
            throw Error(std::string("cuda_to_v210: codec v210 not supported by container ") +
                        octx.outputFormat().name());
        stream.setTimeBase(timebase_);
        // The MXL muxer reads the flow's grain rate off the stream, so leaving
        // the frame rate unset would publish a flow nothing can consume.
        stream.setAverageFrameRate(fps_);
        stream.setFrameRate(fps_);
        if (avcodec_parameters_copy(stream.raw()->codecpar, codecpar_) < 0)
            throw Error("cuda_to_v210: cannot set the stream codec parameters");
    }

    void process() override {
        av::VideoFrame frame = source_->get();
        if (frame.isNull()) return; // Input queue interrupted during shutdown.
        if (isEofMarker(frame)) {
            sink_->put(createEofPacket());
            markFinished();
            return;
        }
        // Nothing is remembered from earlier frames: each one names its own layout and device.
        const AVFrame* raw = frame.raw();
        const auto* frames = raw->hw_frames_ctx ? reinterpret_cast<AVHWFramesContext*>(raw->hw_frames_ctx->data) : nullptr;
        if (raw->format != AV_PIX_FMT_CUDA || !frames ||
            (frames->sw_format != AV_PIX_FMT_P210LE && frames->sw_format != AV_PIX_FMT_YUV422P10LE))
            throw Error("cuda_to_v210: requires CUDA p210le or yuv422p10le frames");
        if (raw->width != width_ || raw->height != height_)
            throw Error("cuda_to_v210: input geometry changed, expected " +
                        std::to_string(width_) + "x" + std::to_string(height_));
        if (!positive(frame.timeBase())) throw Error("cuda_to_v210: input frame needs a positive time base");
        useDevice(frames->device_ref);
        sink_->put(pack(frame, frames->sw_format == AV_PIX_FMT_P210LE));
    }

    static std::shared_ptr<CudaToV210> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto src_edge = nci.edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = nci.edges.find<av::Packet>(params["dst"]);
        // The muxer asks for geometry, rate and time base while the output node
        // is still being created, so none of them can wait for the first frame.
        // Parameters override the upstream metadata, and stand in for it when a
        // node that reports none sits above.
        const bool sized = params.count("width") && params.count("height");
        auto video = src_edge->findNodeUp<IVideoFormatSource>();
        if (!sized && !video) throw Error("cuda_to_v210: no upstream video format, pass width and height");
        const int width = sized ? params.at("width").get<int>() : video->width();
        const int height = sized ? params.at("height").get<int>() : video->height();
        if (width <= 0 || height <= 0 || width % 2 || av_image_check_size(width, height, 0, nullptr) < 0)
            throw Error("cuda_to_v210: requires valid dimensions and an even width");
        // Six pixels per 16 bytes. The default pads rows to 128 bytes, the pitch
        // that SDI-style hardware and the libavcodec v210 codecs use; the minimum
        // is the packed width alone, which is what libavcodec writes before it
        // pads the row out.
        int64_t stride = params.value("stride", int64_t(0));
        if (stride <= 0) stride = (width + 47) / 48 * 128;
        if (stride < (int64_t(width) * 2 + 2) / 3 * 4 || stride % 4 || stride > std::numeric_limits<int>::max() / height)
            throw Error("cuda_to_v210: stride must be a multiple of four, fit a v210 row and an AVPacket");

        auto rate = src_edge->findNodeUp<IFrameRateSource>();
        const av::Rational fps = params.count("fps") ? parseRatio(params.at("fps")) : rate ? rate->frameRate() : av::Rational();
        if (!positive(fps)) throw Error("cuda_to_v210: no positive frame rate from above, pass fps");
        auto tb = src_edge->findNodeUp<ITimeBaseSource>();
        av::Rational timebase = params.count("timebase") ? parseRatio(params.at("timebase")) : tb ? tb->timeBase() : av::Rational();
        if (!positive(timebase)) {
            if (params.count("timebase")) throw Error("cuda_to_v210: timebase must be positive");
            timebase = av::Rational(fps.getDenominator(), fps.getNumerator());
        }
        return std::make_shared<CudaToV210>(src_edge->makeSource(), dst_edge->makeSink(), width, height,
                                            static_cast<int>(stride), fps, timebase);
    }
};

DECLNODE(cuda_to_v210, CudaToV210)
