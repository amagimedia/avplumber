#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../hwaccel.hpp"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>
#include <limits>
#include "../../../objs/src/nodes/hwaccel/v210_unpack.ptx.h"

namespace {

void checkCuda(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* description = nullptr;
    if (cuGetErrorString) cuGetErrorString(result, &description);
    throw Error(std::string("v210_to_cuda: ") + operation + ": " +
                (description ? description : std::to_string(result)));
}

class CurrentContext {
public:
    explicit CurrentContext(CUcontext context) { checkCuda(cuCtxPushCurrent(context), "push context"); }
    ~CurrentContext() {
        CUcontext previous;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
};

struct FramePoolDeleter {
    void operator()(AVBufferRef* ref) const { av_buffer_unref(&ref); }
};

// A private stream and bounded staging allocation isolate this node's work.
// Destruction also covers partial initialization and failed kernel launches.
struct UploadResources {
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr packed = 0;
    void* staging = nullptr;
    std::unique_ptr<AVBufferRef, FramePoolDeleter> frames;

    ~UploadResources() {
        if (!context) return;
        if (CHECK_CU(cuCtxPushCurrent(context))) return;
        if (stream) CHECK_CU(cuStreamSynchronize(stream));
        frames.reset();
        if (packed) CHECK_CU(cuMemFree(packed));
        if (staging) CHECK_CU(cuMemFreeHost(staging));
        if (module) CHECK_CU(cuModuleUnload(module));
        if (stream) CHECK_CU(cuStreamDestroy(stream));
        CUcontext previous;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
};

av::Rational positiveRatio(const std::string& value) {
    auto ratio = parseRatio(value);
    if (ratio.getNumerator() <= 0 || ratio.getDenominator() <= 0)
        throw Error("v210_to_cuda: ratios must be positive");
    return ratio;
}

int colorOption(const Parameters& params, const char* name, int fallback,
                int (*parse)(const char*)) {
    if (!params.count(name)) return fallback;
    const auto value = params.at(name).get<std::string>();
    const int result = parse(value.c_str());
    if (result < 0) throw Error(std::string("v210_to_cuda: invalid ") + name + ": " + value);
    return result;
}

} // namespace

class V210ToCuda : public NodeSISO<av::Packet, av::VideoFrame>, public ReportsFinishByFlag,
                   public IVideoFormatSource, public IFrameRateSource, public ITimeBaseSource {
    int width_, height_, stride_;
    size_t packed_size_;
    AVPixelFormat format_;
    av::Rational fps_, timebase_, aspect_;
    AVColorRange range_;
    AVColorSpace matrix_;
    AVColorPrimaries primaries_;
    AVColorTransferCharacteristic transfer_;
    AVChromaLocation chroma_;
    // Keep the FFmpeg device alive until all CUDA resources have been released.
    std::shared_ptr<HWAccelDevice> device_;
    UploadResources gpu_;

    void initialize() {
        if (global_cuda.has_errors || !device_ || device_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error("v210_to_cuda: requires an initialized CUDA hwaccel device");
        auto* device = reinterpret_cast<AVHWDeviceContext*>(device_->deviceContext()->data);
        gpu_.context = static_cast<AVCUDADeviceContext*>(device->hwctx)->cuda_ctx;
        CurrentContext context(gpu_.context);

        gpu_.frames.reset(av_hwframe_ctx_alloc(device_->deviceContext()));
        if (!gpu_.frames) throw Error("v210_to_cuda: cannot allocate CUDA frame pool");
        auto* frames = reinterpret_cast<AVHWFramesContext*>(gpu_.frames->data);
        frames->format = AV_PIX_FMT_CUDA;
        frames->sw_format = format_;
        frames->width = width_;
        frames->height = height_;
        const int result = av_hwframe_ctx_init(gpu_.frames.get());
        if (result < 0)
            throw Error("v210_to_cuda: CUDA output format requires FFmpeg 8.1 support: " +
                        av::error2string(result));

        // CU_STREAM_NON_BLOCKING is absent from the bundled dynlink declarations.
        constexpr unsigned non_blocking_stream = 0x1;
        checkCuda(cuStreamCreate(&gpu_.stream, non_blocking_stream), "create stream");
        checkCuda(cuMemAlloc(&gpu_.packed, packed_size_), "allocate packed buffer");
        checkCuda(cuMemHostAlloc(&gpu_.staging, packed_size_, 0), "allocate upload staging");
        const std::string module(avpl_v210_unpack_ptx,
                                 avpl_v210_unpack_ptx + avpl_v210_unpack_ptx_len);
        checkCuda(cuModuleLoadDataEx(&gpu_.module, module.c_str(), 0, nullptr, nullptr), "load kernel");
        checkCuda(cuModuleGetFunction(&gpu_.kernel, gpu_.module, "unpack_v210"), "find kernel");
    }

public:
    V210ToCuda(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
               const Parameters& params, std::shared_ptr<HWAccelDevice> device)
        : NodeSISO(std::move(source), std::move(sink)),
          width_(params.at("width").get<int>()), height_(params.at("height").get<int>()),
          fps_(positiveRatio(params.at("fps"))),
          timebase_(params.count("timebase") ? positiveRatio(params.at("timebase"))
                                            : av::Rational(fps_.getDenominator(), fps_.getNumerator())),
          aspect_(positiveRatio(params.value("sample_aspect_ratio", std::string("1/1")))),
          range_(static_cast<AVColorRange>(colorOption(params, "color_range", AVCOL_RANGE_UNSPECIFIED, av_color_range_from_name))),
          matrix_(static_cast<AVColorSpace>(colorOption(params, "colorspace", AVCOL_SPC_UNSPECIFIED, av_color_space_from_name))),
          primaries_(static_cast<AVColorPrimaries>(colorOption(params, "color_primaries", AVCOL_PRI_UNSPECIFIED, av_color_primaries_from_name))),
          transfer_(static_cast<AVColorTransferCharacteristic>(colorOption(params, "color_trc", AVCOL_TRC_UNSPECIFIED, av_color_transfer_from_name))),
          chroma_(static_cast<AVChromaLocation>(colorOption(params, "chroma_location", AVCHROMA_LOC_UNSPECIFIED, av_chroma_location_from_name))),
          device_(std::move(device)) {
        if (width_ <= 0 || height_ <= 0 || width_ % 2 ||
            av_image_check_size(width_, height_, 0, nullptr) < 0)
            throw Error("v210_to_cuda: requires valid dimensions and an even width");
        const int64_t minimum_stride = ((int64_t(width_) * 2 + 2) / 3) * 4;
        const int64_t stride = params.value("stride", ((int64_t(width_) + 47) / 48) * 128);
        if (stride < minimum_stride || stride % 4 ||
            stride > std::numeric_limits<int>::max() / height_)
            throw Error("v210_to_cuda: stride must be a multiple of four, fit a v210 row and an AVPacket");
        stride_ = static_cast<int>(stride);
        packed_size_ = size_t(stride_) * height_;
        const auto format = params.value("format", std::string("p210le"));
        if (format != "p210le" && format != "yuv422p10le")
            throw Error("v210_to_cuda: format must be p210le or yuv422p10le");
        format_ = av_get_pix_fmt(format.c_str());
        if (format_ == AV_PIX_FMT_NONE) throw Error("v210_to_cuda: output pixel format unavailable");
    }

    void process() override {
        av::Packet packet = source_->get();
        if (packet.isNull()) return; // Input queue interrupted during shutdown.
        if (isEofMarker(packet)) {
            onEofConsumed();
            markFinished();
            return;
        }
        if (packet.size() != packed_size_ || !packet.data())
            throw Error("v210_to_cuda: expected exactly one stride * height packed frame per packet");
        if (!packet.pts().isValid() || packet.timeBase().getNumerator() <= 0 ||
            packet.timeBase().getDenominator() <= 0)
            throw Error("v210_to_cuda: input packet needs valid PTS and a positive time base");

        av::VideoFrame output;
        {
            CurrentContext context(gpu_.context);
            const int result = av_hwframe_get_buffer(gpu_.frames.get(), output.raw(), 0);
            if (result < 0) throw Error("v210_to_cuda: cannot allocate output: " + av::error2string(result));
            // Copy bytes only. Unpacking is entirely on the GPU; staging allows
            // arbitrary AVPacket/MXL host buffers without registering their pages.
            std::memcpy(gpu_.staging, packet.data(), packed_size_);
            auto* frame = output.raw();
            int semiplanar = format_ == AV_PIX_FMT_P210LE;
            void* args[] = {&gpu_.packed, &stride_, &width_, &height_,
                            &frame->data[0], &frame->linesize[0],
                            &frame->data[1], &frame->linesize[1],
                            &frame->data[2], &frame->linesize[2], &semiplanar};
            try {
                checkCuda(cuMemcpyHtoDAsync(gpu_.packed, gpu_.staging, packed_size_, gpu_.stream), "upload");
                checkCuda(cuLaunchKernel(gpu_.kernel, (width_ / 2 + 31) / 32, (height_ + 7) / 8, 1,
                                        32, 8, 1, 0, gpu_.stream, args, nullptr), "unpack");
                checkCuda(cuStreamSynchronize(gpu_.stream), "complete frame");
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

    static std::shared_ptr<V210ToCuda> create(NodeCreationInfo& nci) {
        auto device = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, nci.params.at("hwaccel"));
        auto node = createCommon<V210ToCuda>(nci.edges, nci.params, nci.params, device);
        node->initialize();
        return node;
    }
};

DECLNODE(v210_to_cuda, V210ToCuda)
