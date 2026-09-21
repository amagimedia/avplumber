#include "v210_cuda.hpp"

#include <avcpp/codec.h>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>
#include <memory>
#include "../../../objs/src/nodes/hwaccel/v210_pack.ptx.h"

namespace {

constexpr const char* kNode = "cuda_to_v210";

// Download buffers outlive the node: a packet can still sit in a queue, or in
// the muxer, when the graph is torn down. The pool therefore keeps its own
// reference to the CUDA device context, so cuMemFreeHost always runs against a
// live context.
struct PinnedPool {
    CUcontext context = nullptr;
    AVBufferRef* device_ref = nullptr;
    size_t payload = 0;
};

void releasePinned(void* opaque, uint8_t* data) {
    auto* pool = static_cast<PinnedPool*>(opaque);
    if (CHECK_CU(cuCtxPushCurrent(pool->context))) return;
    CHECK_CU(cuMemFreeHost(data));
    CUcontext previous;
    CHECK_CU(cuCtxPopCurrent(&previous));
}

AVBufferRef* allocPinned(void* opaque, size_t size) {
    auto* pool = static_cast<PinnedPool*>(opaque);
    if (CHECK_CU(cuCtxPushCurrent(pool->context))) return nullptr;
    AVBufferRef* buffer = nullptr;
    void* data = nullptr;
    if (!CHECK_CU(cuMemHostAlloc(&data, size, 0))) {
        // Only the payload is written per frame; zero the AVPacket padding once.
        std::memset(static_cast<uint8_t*>(data) + pool->payload, 0, size - pool->payload);
        buffer = av_buffer_create(static_cast<uint8_t*>(data), size, releasePinned, opaque, 0);
        if (!buffer) CHECK_CU(cuMemFreeHost(data));
    }
    CUcontext previous;
    CHECK_CU(cuCtxPopCurrent(&previous));
    return buffer;
}

void destroyPinnedPool(void* opaque) {
    auto* pool = static_cast<PinnedPool*>(opaque);
    av_buffer_unref(&pool->device_ref);
    delete pool;
}

struct PacketDeleter {
    void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

// Everything the pack needs on the device, released in one place so that a
// failed initialization leaves nothing behind for the next attempt.
struct PackResources {
    CUcontext context = nullptr;
    CUstream stream = nullptr; // Borrowed: the frames arrive on it already.
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr packed = 0;
    AVBufferPool* pool = nullptr;
    // Keeps the CUDA device alive for as long as this node holds resources.
    AVBufferRef* device_ref = nullptr;

    void release() {
        // Buffers still referenced by packets survive this; the pool waits.
        if (pool) av_buffer_pool_uninit(&pool);
        if (context && !CHECK_CU(cuCtxPushCurrent(context))) {
            if (stream) CHECK_CU(cuStreamSynchronize(stream));
            if (packed) CHECK_CU(cuMemFree(packed));
            if (module) CHECK_CU(cuModuleUnload(module));
            CUcontext previous;
            CHECK_CU(cuCtxPopCurrent(&previous));
        }
        packed = 0;
        module = nullptr;
        kernel = nullptr;
        stream = nullptr;
        context = nullptr;
        if (device_ref) av_buffer_unref(&device_ref);
    }

    ~PackResources() { release(); }
};

} // namespace

// The mirror of v210_to_cuda: packs CUDA 10-bit 4:2:2 frames into v210 packets
// with a PTX kernel and DMAs each one straight into the buffer the muxer will
// read, which replaces both a hwdownload and the libavcodec v210 encoder.
//
// It implements IEncoder without an encoder: v210 is a memory layout, so the
// muxer only needs codec parameters and a stream rate, and those are known at
// node creation — which is when the output node opens the format context.
class CudaToV210 : public NodeSISO<av::VideoFrame, av::Packet>, public IEncoder,
                   public ReportsFinishByFlag {
    v210cuda::Layout layout_;
    av::Rational fps_, timebase_;
    av::Codec codec_;
    AVCodecParameters* codecpar_ = nullptr;
    int semiplanar_ = 0;
    PackResources gpu_;

    void initialize(const av::VideoFrame& frame) {
        const AVFrame* raw = frame.raw();
        if (!raw || !raw->hw_frames_ctx)
            throw Error(std::string(kNode) + ": requires CUDA frames with a hardware frames context");
        auto* frames = reinterpret_cast<AVHWFramesContext*>(raw->hw_frames_ctx->data);
        if (frames->format != AV_PIX_FMT_CUDA || !v210cuda::isSupportedFormat(frames->sw_format))
            throw Error(std::string(kNode) + ": requires cuda/p210le or cuda/yuv422p10le frames, got " +
                        av_get_pix_fmt_name(frames->format) + "/" + av_get_pix_fmt_name(frames->sw_format));
        if (!frames->device_ref || !frames->device_ctx || !frames->device_ctx->hwctx)
            throw Error(std::string(kNode) + ": frames carry no CUDA device context");
        semiplanar_ = v210cuda::isSemiplanar(frames->sw_format) ? 1 : 0;

        try {
            gpu_.device_ref = av_buffer_ref(frames->device_ref);
            if (!gpu_.device_ref) throw Error(std::string(kNode) + ": cannot reference the CUDA device");
            auto* device = static_cast<AVCUDADeviceContext*>(frames->device_ctx->hwctx);
            gpu_.context = device->cuda_ctx;
            // Pack on the device context's stream. Whatever produced this frame
            // used it too, so the ordering needs no extra synchronization.
            gpu_.stream = device->stream;

            v210cuda::CurrentContext context(gpu_.context, kNode);
            v210cuda::check(cuMemAlloc(&gpu_.packed, layout_.size), kNode, "allocate packed buffer");
            // The kernel writes whole words and never the row padding, so
            // zeroing once keeps that padding defined for every frame.
            v210cuda::check(cuMemsetD8(gpu_.packed, 0, layout_.size), kNode, "clear packed buffer");

            auto* pool = new PinnedPool{gpu_.context, av_buffer_ref(gpu_.device_ref), layout_.size};
            gpu_.pool = av_buffer_pool_init2(layout_.size + AV_INPUT_BUFFER_PADDING_SIZE, pool,
                                             allocPinned, destroyPinnedPool);
            if (!gpu_.pool) {
                destroyPinnedPool(pool);
                throw Error(std::string(kNode) + ": cannot create the download buffer pool");
            }

            const std::string image(avpl_v210_pack_ptx, avpl_v210_pack_ptx + avpl_v210_pack_ptx_len);
            v210cuda::check(cuModuleLoadDataEx(&gpu_.module, image.c_str(), 0, nullptr, nullptr),
                            kNode, "load kernel");
            v210cuda::check(cuModuleGetFunction(&gpu_.kernel, gpu_.module, "pack_v210"), kNode, "find kernel");
        } catch (...) {
            gpu_.release();
            throw;
        }
    }

    av::Packet pack(av::VideoFrame& frame) {
        AVBufferRef* buffer = av_buffer_pool_get(gpu_.pool);
        if (!buffer) throw Error(std::string(kNode) + ": cannot allocate a pinned download buffer");
        std::unique_ptr<AVPacket, PacketDeleter> raw(av_packet_alloc());
        if (!raw) {
            av_buffer_unref(&buffer);
            throw Error(std::string(kNode) + ": cannot allocate a packet");
        }
        // The packet owns the pinned buffer from here, including on failure.
        raw->buf = buffer;
        raw->data = buffer->data;
        raw->size = static_cast<int>(layout_.size);

        {
            AVFrame* input = frame.raw();
            void* args[] = {&gpu_.packed, &layout_.stride, &layout_.width, &layout_.height,
                            &input->data[0], &input->linesize[0],
                            &input->data[1], &input->linesize[1],
                            &input->data[2], &input->linesize[2], &semiplanar_};
            const int blocks = (layout_.width + 5) / 6;
            v210cuda::CurrentContext context(gpu_.context, kNode);
            try {
                v210cuda::check(cuLaunchKernel(gpu_.kernel, (blocks + 31) / 32, (layout_.height + 7) / 8, 1,
                                               32, 8, 1, 0, gpu_.stream, args, nullptr), kNode, "pack");
                // Straight into the packet's payload: the muxer's copy into the
                // grain is the only time a CPU touches these bytes.
                v210cuda::check(cuMemcpyDtoHAsync(raw->data, gpu_.packed, layout_.size, gpu_.stream),
                                kNode, "download");
                v210cuda::check(cuStreamSynchronize(gpu_.stream), kNode, "complete frame");
            } catch (...) {
                // Do not recycle the buffer while a queued operation writes it.
                CHECK_CU(cuStreamSynchronize(gpu_.stream));
                throw;
            }
        }

        raw->duration = av_rescale_q(frame.raw()->duration, frame.timeBase(), timebase_);
        av::Packet packet(raw.get()); // References the pinned payload, no copy.
        packet.setTimeBase(timebase_);
        packet.setPts(frame.pts());
        packet.setDts(frame.pts()); // Uncompressed: no reordering, every frame a key frame.
        packet.setKeyPacket(true);
        packet.setComplete(true);
        return packet;
    }

public:
    CudaToV210(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
               const v210cuda::Layout& layout, av::Rational fps, av::Rational timebase)
        : NodeSISO(std::move(source), std::move(sink)), layout_(layout), fps_(fps), timebase_(timebase),
          codec_(av::findEncodingCodec("v210")) {
        codecpar_ = avcodec_parameters_alloc();
        if (!codecpar_) throw Error(std::string(kNode) + ": cannot allocate codec parameters");
        codecpar_->codec_type = AVMEDIA_TYPE_VIDEO;
        codecpar_->codec_id = AV_CODEC_ID_V210;
        codecpar_->codec_tag = MKTAG('v', '2', '1', '0');
        codecpar_->width = layout_.width;
        codecpar_->height = layout_.height;
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

    ~CudaToV210() override {
        if (codecpar_) avcodec_parameters_free(&codecpar_);
    }

    av::Codec& encodingCodec() override { return codec_; }
    AVCodecParameters* codecParameters() override { return codecpar_; }

    void setOutput(av::Stream& stream, av::FormatContext& octx) override {
        if (!codec_.isNull() && !octx.outputFormat().codecSupported(codec_))
            throw Error(std::string(kNode) + ": codec v210 not supported by container " +
                        octx.outputFormat().name());
        stream.setTimeBase(timebase_);
        // The MXL muxer reads the flow's grain rate off the stream, so leaving
        // the frame rate unset would publish a flow nothing can consume.
        stream.setAverageFrameRate(fps_);
        stream.setFrameRate(fps_);
        if (avcodec_parameters_copy(stream.raw()->codecpar, codecpar_) < 0)
            throw Error(std::string(kNode) + ": cannot set the stream codec parameters");
    }

    void process() override {
        av::VideoFrame frame = source_->get();
        if (frame.isNull()) return; // Input queue interrupted during shutdown.
        if (isEofMarker(frame)) {
            onEofConsumed();
            sink_->put(createEofPacket());
            markFinished();
            return;
        }
        if (frame.pixelFormat().get() != AV_PIX_FMT_CUDA)
            throw Error(std::string(kNode) + ": requires CUDA frames, got " + frame.pixelFormat().name());
        if (frame.width() != layout_.width || frame.height() != layout_.height)
            throw Error(std::string(kNode) + ": input geometry changed, expected " +
                        std::to_string(layout_.width) + "x" + std::to_string(layout_.height));
        if (frame.timeBase().getNumerator() <= 0 || frame.timeBase().getDenominator() <= 0)
            throw Error(std::string(kNode) + ": input frame needs a positive time base");
        if (!gpu_.kernel) initialize(frame);
        sink_->put(pack(frame));
    }

    static std::shared_ptr<CudaToV210> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto src_edge = nci.edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = nci.edges.find<av::Packet>(params["dst"]);
        // The muxer asks for geometry, rate and time base while the output node
        // is still being created, so none of them can wait for the first frame.
        // Parameters override the upstream metadata, and stand in for it when a
        // node that reports none sits above.
        int width = 0, height = 0;
        if (params.count("width") && params.count("height")) {
            width = params.at("width").get<int>();
            height = params.at("height").get<int>();
        } else if (auto video = src_edge->findNodeUp<IVideoFormatSource>()) {
            width = video->width();
            height = video->height();
        } else {
            throw Error(std::string(kNode) + ": no upstream video format, pass width and height");
        }
        av::Rational fps;
        if (params.count("fps")) {
            fps = v210cuda::positiveRatio(params.at("fps"), kNode);
        } else if (auto rate = src_edge->findNodeUp<IFrameRateSource>()) {
            fps = rate->frameRate();
        }
        if (fps.getNumerator() <= 0 || fps.getDenominator() <= 0)
            throw Error(std::string(kNode) + ": no upstream frame rate, pass fps");
        av::Rational timebase;
        if (params.count("timebase")) {
            timebase = v210cuda::positiveRatio(params.at("timebase"), kNode);
        } else if (auto tb = src_edge->findNodeUp<ITimeBaseSource>()) {
            timebase = tb->timeBase();
        }
        if (timebase.getNumerator() <= 0 || timebase.getDenominator() <= 0)
            timebase = av::Rational(fps.getDenominator(), fps.getNumerator());
        const auto layout = v210cuda::makeLayout(kNode, width, height, params.value("stride", int64_t(0)));
        return std::make_shared<CudaToV210>(src_edge->makeSource(), dst_edge->makeSink(),
                                            layout, fps, timebase);
    }
};

DECLNODE(cuda_to_v210, CudaToV210)
