#include "node_common.hpp"

#if HAVE_CUDA && HAVE_NVJPEG2K

#include "../frame_color_tags.hpp"
#include "../hwaccel.hpp"
#include "nvjpeg2k_common.hpp"

#include <algorithm>
#include <optional>
#include <sstream>

namespace {

constexpr const char *kWho = "nvjpeg2k_dec";

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
    avp::nvjpeg2k::WorkStream work_;
    AVBufferRef *hw_frames_ctx_ = nullptr;
    // What this node advertises, fixed at creation: width(), height() and realPixelFormat() are
    // read from other threads and by the nodes built below this one, so a bitstream that does not
    // match is an error rather than a new output format (see ensureFrames).
    const int width_;
    const int height_;
    const AVPixelFormat output_format_;
    av::Rational timebase_{0, 1};
    av::Rational framerate_{0, 1};
    // A JPEG 2000 codestream carries no colour tags, so the samples come out as they went in but
    // their meaning does not travel with them; this declares it, as `one_to_many` does.
    FrameColorTags color_;
    int64_t frames_seen_ = 0;

    void ensureDecoder() {
        if (handle_) return;
        avp::nvjpeg2k::check(nvjpeg2kCreateSimple(&handle_), kWho, "create");
        avp::nvjpeg2k::check(nvjpeg2kDecodeStateCreate(handle_, &decode_state_), kWho,
                             "decode state create");
        avp::nvjpeg2k::check(nvjpeg2kStreamCreate(&stream_), kWho, "stream create");
    }

    /// Frees the SDK's resources with the hwaccel's CUDA context current. Reports a failure rather
    /// than propagating it: this runs from the destructor, and nvjpeg2k throws an exception type of
    /// its own (not derived from std::exception) that must not escape one.
    void destroyDecoder() noexcept {
        try {
            avp::nvjpeg2k::ContextGuard ctx(
                avp::nvjpeg2k::deviceContext(hwaccel_->deviceContext(), kWho)->cuda_ctx, kWho);
            work_.destroy();   // its stream and event belong to this context too
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
        } catch (const std::exception &e) {
            logstream << "nvjpeg2k_dec: " << e.what() << "; the decoder may not be fully freed";
        } catch (...) {
            logstream << "nvjpeg2k_dec: nvjpeg2k threw while freeing the decoder";
        }
    }

    /// Allocates the frame pool, once, for the format the node was created with.
    ///
    /// The bitstream's own dimensions and precision are checked against that rather than adopted:
    /// width(), height() and realPixelFormat() are read when the nodes below this one are created,
    /// so changing them here would leave a downstream filter graph built for a format that no
    /// longer arrives -- and would race with the control and stats threads reading them.
    void ensureFrames(int width, int height, AVPixelFormat format) {
        if (width != width_ || height != height_ || format != output_format_) {
            std::ostringstream os;
            os << "nvjpeg2k_dec: the bitstream is " << width << "x" << height << " "
               << avp::nvjpeg2k::pixFmtName(format) << ", but this node was created for "
               << width_ << "x" << height_ << " " << avp::nvjpeg2k::pixFmtName(output_format_)
               << "; set width/height/real_pixel_format to match the producer";
            throw Error(os.str());
        }
        if (hw_frames_ctx_) return;
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
        logstream << "nvjpeg2k_dec frames"
                  << " size=" << width_ << "x" << height_
                  << " format=" << avp::nvjpeg2k::pixFmtName(output_format_);
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
        destroyDecoder();   // before the frames context, and before hwaccel_ drops the device
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
        if (!pkt.isComplete() || isEofMarker(pkt)) {
            this->sink_->put(av::VideoFrame());
            finished_ = true;
            return;
        }
        if (!pkt.data() || !pkt.size()) return;

        av::VideoFrame out;
        int precision = 0;
        {
            avp::nvjpeg2k::ContextGuard ctx(
                avp::nvjpeg2k::deviceContext(hwaccel_->deviceContext(), kWho)->cuda_ctx, kWho);

            ensureDecoder();
            avp::nvjpeg2k::check(nvjpeg2kStreamParse(handle_, pkt.data(), pkt.size(), 0, 0, stream_),
                                 kWho, "stream parse");
            nvjpeg2kImageInfo_t info = {};
            avp::nvjpeg2k::check(nvjpeg2kStreamGetImageInfo(stream_, &info), kWho, "image info");
            if (info.num_components != 3) {
                throw Error("nvjpeg2k_dec requires 3 components, got " +
                            std::to_string(info.num_components));
            }

            nvjpeg2kImageComponentInfo_t comp[3] = {};
            for (int c = 0; c < 3; ++c) {
                avp::nvjpeg2k::check(nvjpeg2kStreamGetImageComponentInfo(stream_, &comp[c], c), kWho,
                                     "component info");
            }
            if (comp[0].component_width != comp[1].component_width ||
                comp[0].component_width != comp[2].component_width ||
                comp[0].component_height != comp[1].component_height ||
                comp[0].component_height != comp[2].component_height) {
                throw Error("nvjpeg2k_dec supports 4:4:4 inputs only");
            }

            precision = std::max({comp[0].precision, comp[1].precision, comp[2].precision});
            ensureFrames(static_cast<int>(info.image_width), static_cast<int>(info.image_height),
                         avp::nvjpeg2k::formatForPrecision(precision, kWho));

            const int r = av_hwframe_get_buffer(hw_frames_ctx_, out.raw(), 0);
            if (r < 0) {
                throw Error("nvjpeg2k_dec av_hwframe_get_buffer failed: " + av::error2string(r));
            }

            // nvjpeg2kImage_t points at the caller's per-component arrays; it holds none itself.
            void *planes[3] = {};
            size_t pitches[3] = {};
            for (int c = 0; c < 3; ++c) {
                planes[c] = out.data(c);
                pitches[c] = static_cast<size_t>(out.raw()->linesize[c]);
                if (!planes[c] || !pitches[c]) {
                    throw Error(std::string("nvjpeg2k_dec missing plane ") + std::to_string(c));
                }
            }
            nvjpeg2kImage_t img = {};
            img.pixel_data = planes;
            img.pitch_in_bytes = pitches;
            img.pixel_type = precision > 8 ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;
            img.num_components = 3;

            auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(out.raw()->hw_frames_ctx->data);
            auto *cuda_ctx = reinterpret_cast<AVCUDADeviceContext *>(frames_ctx->device_ctx->hwctx);
            // The pool hands back memory whose refcount reached zero, which says nothing about
            // GPU work a downstream node may still have in flight on it. Wait for the frames'
            // stream before writing, then let the host wait only for our own decode.
            work_.ensure(kWho);
            work_.waitFor(cuda_ctx->stream, kWho);
            avp::nvjpeg2k::check(
                nvjpeg2kDecode(handle_, decode_state_, stream_, &img, work_.get()), kWho, "decode");
            work_.synchronize(kWho);
        }

        color_.apply(out.raw());
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
        if (!avp::nvjpeg2k::supportedFormat(real_pf.get())) {
            real_pf = av::PixelFormat(AV_PIX_FMT_YUV444P);
        }
        if (params.count("real_pixel_format")) {
            real_pf = av::PixelFormat(avp::nvjpeg2k::parsePixelFormat(
                params["real_pixel_format"].get<std::string>(), kWho));
        }

        auto node = std::make_shared<Nvjpeg2kDecoderNode>(
            src_edge->makeSource(), dst_edge->makeSink(), hwaccel, width, height, real_pf.get(),
            tb_md->timeBase(), fr_md ? fr_md->frameRate() : av::Rational{0, 0});
        if (params.count("frame_color")) {
            node->color_ = FrameColorTags::parse(params["frame_color"]);
        }
        return node;
    }
};

DECLNODE(nvjpeg2k_dec, Nvjpeg2kDecoderNode)

#endif
