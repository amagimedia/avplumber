#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../hwaccel.hpp"
#include "../../avbuffer.hpp"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>
#include <limits>
#include "../../../objs/src/nodes/hwaccel/v210_unpack.ptx.h"

// Headerless raw pictures, one per packet, uploaded straight into CUDA frames:
// v210_to_cuda unpacks packed v210 on the GPU, raw_to_cuda copies the planes of a
// CUDA-storable pixel format (nv12, p010le, ...) as they are.
namespace {

void checkCuda(CUresult result, const char* node, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* description = nullptr;
    if (cuGetErrorString) cuGetErrorString(result, &description);
    throw Error(std::string(node) + ": " + operation + ": " +
                (description ? description : std::to_string(result)));
}

class CurrentContext {
public:
    CurrentContext(CUcontext context, const char* node) { checkCuda(cuCtxPushCurrent(context), node, "push context"); }
    ~CurrentContext() {
        CUcontext previous;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
};

// A private stream and bounded staging allocation isolate this node's work
// (module, kernel and packed are v210_to_cuda's unpacking resources).
// Destruction also covers partial initialization and failed kernel launches.
struct UploadResources {
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUevent readers_done = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr packed = 0;
    void* staging = nullptr;
    avp::AvBufferRef frames;

    ~UploadResources() {
        if (!context) return;
        if (CHECK_CU(cuCtxPushCurrent(context))) return;
        if (stream) CHECK_CU(cuStreamSynchronize(stream));
        frames.reset();
        if (packed) CHECK_CU(cuMemFree(packed));
        if (staging) CHECK_CU(cuMemFreeHost(staging));
        if (module) CHECK_CU(cuModuleUnload(module));
        if (readers_done) CHECK_CU(cuEventDestroy(readers_done));
        if (stream) CHECK_CU(cuStreamDestroy(stream));
        CUcontext previous;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
};

av::Rational positiveRatio(const std::string& value, const char* node) {
    auto ratio = parseRatio(value);
    if (ratio.getNumerator() <= 0 || ratio.getDenominator() <= 0)
        throw Error(std::string(node) + ": ratios must be positive");
    return ratio;
}

int colorOption(const Parameters& params, const char* node, const char* name, int fallback,
                int (*parse)(const char*)) {
    if (!params.count(name)) return fallback;
    const auto value = params.at(name).get<std::string>();
    const int result = parse(value.c_str());
    if (result < 0) throw Error(std::string(node) + ": invalid " + name + ": " + value);
    return result;
}

} // namespace

// Each packet is copied into pinned staging, so the host-to-device copy is a DMA
// on the node's own non-blocking stream. FFmpeg hwupload instead copies from
// pageable memory on the device context's legacy stream, which the driver
// stages itself, contending on the context shared by every mixer thread.
// The stream is synchronized before the frame is published: consumers see an
// ordinary, complete CUDA frame.
// FFmpeg filters read these frames on the device context's stream and release
// them without synchronizing, so the pool can hand back a picture a queued
// kernel still reads. Each upload waits for the work queued on that stream so
// far; nothing ever waits for an upload.
class PacketToCuda : public NodeSISO<av::Packet, av::VideoFrame>, public ReportsFinishByFlag,
                     public IVideoFormatSource, public IFrameRateSource, public ITimeBaseSource {
    av::Rational fps_, timebase_, aspect_;
    AVColorRange range_;
    AVColorSpace matrix_;
    AVColorPrimaries primaries_;
    AVColorTransferCharacteristic transfer_;
    AVChromaLocation chroma_;
    // Keep the FFmpeg device alive until all CUDA resources have been released.
    std::shared_ptr<HWAccelDevice> device_;
    CUstream device_stream_ = nullptr;   // the FFmpeg device context's stream

    void initialize() {
        if (global_cuda.has_errors || !device_ || device_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error(std::string(node_) + ": requires an initialized CUDA hwaccel device");
        auto* device = reinterpret_cast<AVHWDeviceContext*>(device_->deviceContext()->data);
        gpu_.context = static_cast<AVCUDADeviceContext*>(device->hwctx)->cuda_ctx;
        device_stream_ = static_cast<AVCUDADeviceContext*>(device->hwctx)->stream;
        CurrentContext context(gpu_.context, node_);

        gpu_.frames.reset(av_hwframe_ctx_alloc(device_->deviceContext()));
        if (!gpu_.frames) throw Error(std::string(node_) + ": cannot allocate CUDA frame pool");
        auto* frames = reinterpret_cast<AVHWFramesContext*>(gpu_.frames->data);
        frames->format = AV_PIX_FMT_CUDA;
        frames->sw_format = format_;
        frames->width = width_;
        frames->height = height_;
        const int result = av_hwframe_ctx_init(gpu_.frames.get());
        if (result < 0)
            throw Error(std::string(node_) + ": CUDA frames cannot store " + av_get_pix_fmt_name(format_) +
                        " (P210 needs FFmpeg 8.1): " + av::error2string(result));

        // CU_STREAM_NON_BLOCKING is absent from the bundled dynlink declarations.
        constexpr unsigned kNonBlockingStream = 0x1;
        check(cuStreamCreate(&gpu_.stream, kNonBlockingStream), "create stream");
        check(cuMemHostAlloc(&gpu_.staging, packet_size_, 0), "allocate upload staging");
        check(cuEventCreate(&gpu_.readers_done, CU_EVENT_DISABLE_TIMING), "create event");
        initializeDevice();
    }

protected:
    const char* node_;
    int width_, height_;
    // Set by the derived constructor, before initialize().
    size_t packet_size_ = 0;
    AVPixelFormat format_ = AV_PIX_FMT_NONE;
    // Drop a picture shorter than packet_size_ instead of failing (raw_to_cuda).
    bool drop_truncated_ = false;
    UploadResources gpu_;

    void check(CUresult result, const char* operation) const { checkCuda(result, node_, operation); }
    // Further allocations, with the context current.
    virtual void initializeDevice() {}
    // Queue the work that turns gpu_.staging into `frame` on gpu_.stream.
    virtual void enqueue(AVFrame* frame) = 0;

    template <typename Child> static std::shared_ptr<Child> build(NodeCreationInfo& nci) {
        auto device = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, nci.params.at("hwaccel"));
        auto node = createCommon<Child>(nci.edges, nci.params, nci.params, device);
        node->initialize();
        return node;
    }

public:
    PacketToCuda(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
                 const Parameters& params, std::shared_ptr<HWAccelDevice> device, const char* node)
        : NodeSISO(std::move(source), std::move(sink)),
          fps_(positiveRatio(params.at("fps"), node)),
          timebase_(params.count("timebase") ? positiveRatio(params.at("timebase"), node)
                                            : av::Rational(fps_.getDenominator(), fps_.getNumerator())),
          aspect_(positiveRatio(params.value("sample_aspect_ratio", std::string("1/1")), node)),
          range_(static_cast<AVColorRange>(colorOption(params, node, "color_range", AVCOL_RANGE_UNSPECIFIED, av_color_range_from_name))),
          matrix_(static_cast<AVColorSpace>(colorOption(params, node, "colorspace", AVCOL_SPC_UNSPECIFIED, av_color_space_from_name))),
          primaries_(static_cast<AVColorPrimaries>(colorOption(params, node, "color_primaries", AVCOL_PRI_UNSPECIFIED, av_color_primaries_from_name))),
          transfer_(static_cast<AVColorTransferCharacteristic>(colorOption(params, node, "color_trc", AVCOL_TRC_UNSPECIFIED, av_color_transfer_from_name))),
          chroma_(static_cast<AVChromaLocation>(colorOption(params, node, "chroma_location", AVCHROMA_LOC_UNSPECIFIED, av_chroma_location_from_name))),
          device_(std::move(device)), node_(node),
          width_(params.at("width").get<int>()), height_(params.at("height").get<int>()) {
        if (width_ <= 0 || height_ <= 0 || av_image_check_size(width_, height_, 0, nullptr) < 0)
            throw Error(std::string(node_) + ": requires valid dimensions");
    }

    void process() override {
        av::Packet packet = source_->get();
        if (packet.isNull()) return; // Input queue interrupted during shutdown.
        if (isEofMarker(packet)) {
            onEofConsumed();
            markFinished();
            return;
        }
        if ((packet.size() != packet_size_ && !(drop_truncated_ && packet.size() < packet_size_)) || !packet.data())
            throw Error(std::string(node_) + ": expected exactly one " + std::to_string(packet_size_) +
                        "-byte picture per packet, got " + std::to_string(packet.size()));
        if (packet.size() < packet_size_) {
            // rawvideo passes on the partial picture at the end of a file whose size is
            // not a whole number of pictures; dropping it keeps a looped source running.
            logstream << node_ << ": dropping a truncated " << packet.size() << "-byte picture";
            return;
        }
        if (!packet.pts().isValid() || packet.timeBase().getNumerator() <= 0 ||
            packet.timeBase().getDenominator() <= 0)
            throw Error(std::string(node_) + ": input packet needs valid PTS and a positive time base");

        av::VideoFrame output;
        {
            CurrentContext context(gpu_.context, node_);
            const int result = av_hwframe_get_buffer(gpu_.frames.get(), output.raw(), 0);
            if (result < 0) throw Error(std::string(node_) + ": cannot allocate output: " + av::error2string(result));
            // Staging allows arbitrary AVPacket/MXL host buffers without registering their pages.
            std::memcpy(gpu_.staging, packet.data(), packet_size_);
            try {
                // After av_hwframe_get_buffer: the reads of a recycled frame were queued before its release.
                check(cuEventRecord(gpu_.readers_done, device_stream_), "record device stream");
                check(cuStreamWaitEvent(gpu_.stream, gpu_.readers_done, 0), "wait for device stream");
                enqueue(output.raw());
                check(cuStreamSynchronize(gpu_.stream), "complete frame");
            } catch (...) {
                // Do not recycle output/staging while a queued operation uses it.
                CHECK_CU(cuStreamSynchronize(gpu_.stream));
                throw;
            }
        }
        output.setTimeBase(timebase_);
        output.raw()->time_base = timebase_;
        output.raw()->pts = rescaleTS(packet.pts(), timebase_).timestamp();
        output.raw()->pkt_dts = rescaleTS(packet.dts(), timebase_).timestamp();
        output.raw()->duration = av_rescale_q(packet.raw()->duration, packet.timeBase(), timebase_);
        output.raw()->sample_aspect_ratio = aspect_;
        output.raw()->color_range = range_;
        output.raw()->colorspace = matrix_;
        output.raw()->color_primaries = primaries_;
        output.raw()->color_trc = transfer_;
        output.raw()->chroma_location = chroma_;
        output.setComplete(true);
        sink_->put(output);
    }

    int width() override { return width_; }
    int height() override { return height_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(format_); }
    av::Rational frameRate() override { return fps_; }
    av::Rational timeBase() override { return timebase_; }
};

class V210ToCuda : public PacketToCuda {
    int stride_;

    void initializeDevice() override {
        check(cuMemAlloc(&gpu_.packed, packet_size_), "allocate packed buffer");
        const std::string module(avpl_v210_unpack_ptx,
                                 avpl_v210_unpack_ptx + avpl_v210_unpack_ptx_len);
        check(cuModuleLoadDataEx(&gpu_.module, module.c_str(), 0, nullptr, nullptr), "load kernel");
        check(cuModuleGetFunction(&gpu_.kernel, gpu_.module, "unpack_v210"), "find kernel");
    }

    // Copy bytes only; unpacking is entirely on the GPU.
    void enqueue(AVFrame* frame) override {
        int semiplanar = format_ == AV_PIX_FMT_P210LE;
        void* args[] = {&gpu_.packed, &stride_, &width_, &height_,
                        &frame->data[0], &frame->linesize[0],
                        &frame->data[1], &frame->linesize[1],
                        &frame->data[2], &frame->linesize[2], &semiplanar};
        check(cuMemcpyHtoDAsync(gpu_.packed, gpu_.staging, packet_size_, gpu_.stream), "upload");
        check(cuLaunchKernel(gpu_.kernel, (width_ / 2 + 31) / 32, (height_ + 7) / 8, 1,
                             32, 8, 1, 0, gpu_.stream, args, nullptr), "unpack");
    }

public:
    V210ToCuda(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
               const Parameters& params, std::shared_ptr<HWAccelDevice> device)
        : PacketToCuda(std::move(source), std::move(sink), params, std::move(device), "v210_to_cuda") {
        if (width_ % 2) throw Error("v210_to_cuda: requires an even width");
        const int64_t minimum_stride = ((int64_t(width_) * 2 + 2) / 3) * 4;
        const int64_t stride = params.value("stride", ((int64_t(width_) + 47) / 48) * 128);
        if (stride < minimum_stride || stride % 4 ||
            stride > std::numeric_limits<int>::max() / height_)
            throw Error("v210_to_cuda: stride must be a multiple of four, fit a v210 row and an AVPacket");
        stride_ = static_cast<int>(stride);
        packet_size_ = size_t(stride_) * height_;
        const auto format = params.value("sw_format", std::string("p210le"));
        if (format != "p210le" && format != "yuv422p10le")
            throw Error("v210_to_cuda: sw_format must be p210le or yuv422p10le");
        format_ = av_get_pix_fmt(format.c_str());
        if (format_ == AV_PIX_FMT_NONE) throw Error("v210_to_cuda: output pixel format unavailable");
    }

    static std::shared_ptr<V210ToCuda> create(NodeCreationInfo& nci) { return build<V210ToCuda>(nci); }
};

// The packet holds the planes back to back without row padding (rawvideo's
// layout); each plane is copied as is into the frame's pitched plane.
class RawToCuda : public PacketToCuda {
    size_t offsets_[4] = {};
    int linesizes_[4] = {};
    size_t heights_[4] = {};

    void enqueue(AVFrame* frame) override {
        for (int i = 0; i < 4 && linesizes_[i]; ++i) {
            CUDA_MEMCPY2D copy = {};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = static_cast<uint8_t*>(gpu_.staging) + offsets_[i];
            copy.srcPitch = size_t(linesizes_[i]);
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = reinterpret_cast<CUdeviceptr>(frame->data[i]);
            copy.dstPitch = size_t(frame->linesize[i]);
            copy.WidthInBytes = size_t(linesizes_[i]);
            copy.Height = heights_[i];
            check(cuMemcpy2DAsync(&copy, gpu_.stream), "upload");
        }
    }

public:
    RawToCuda(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
              const Parameters& params, std::shared_ptr<HWAccelDevice> device)
        : PacketToCuda(std::move(source), std::move(sink), params, std::move(device), "raw_to_cuda") {
        drop_truncated_ = true;
        const auto format = params.at("pixel_format").get<std::string>();
        format_ = av_get_pix_fmt(format.c_str());
        // Whether CUDA frames can store the format is checked by av_hwframe_ctx_init.
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format_);
        if (!desc) throw Error("raw_to_cuda: unknown pixel_format: " + format);
        if (width_ % (1 << desc->log2_chroma_w) || height_ % (1 << desc->log2_chroma_h))
            throw Error("raw_to_cuda: dimensions must be multiples of the " + format + " chroma subsampling");
        ptrdiff_t pitches[4] = {};
        size_t sizes[4] = {};
        if (av_image_fill_linesizes(linesizes_, format_, width_) < 0) throw Error("raw_to_cuda: cannot lay out " + format);
        for (int i = 0; i < 4; ++i) pitches[i] = linesizes_[i];
        if (av_image_fill_plane_sizes(sizes, format_, height_, pitches) < 0)
            throw Error("raw_to_cuda: cannot lay out " + format);
        for (int i = 0; i < 4 && linesizes_[i]; ++i) {
            offsets_[i] = packet_size_;
            heights_[i] = sizes[i] / size_t(linesizes_[i]);
            packet_size_ += sizes[i];
        }
    }

    static std::shared_ptr<RawToCuda> create(NodeCreationInfo& nci) { return build<RawToCuda>(nci); }
};

DECLNODE(v210_to_cuda, V210ToCuda)
DECLNODE(raw_to_cuda, RawToCuda)
