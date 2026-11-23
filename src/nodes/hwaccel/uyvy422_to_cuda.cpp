#include "../node_common.hpp"
#include "../../frame_color_tags.hpp"
#include "CudaPinnedUploader.hpp"

#include <atomic>
#include <chrono>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

// Packed UYVY frames in host memory (8-bit 4:2:2: what an NDI receiver delivers, or a capture
// card) as NV12 CUDA frames. Each picture is copied once, as it is, into page-locked memory; the
// device takes it from there and unpacks it into the frame, so no CPU converts a sample. A filter
// graph for the same (`format=nv12,hwupload_cuda`) converts every picture with swscale first,
// which is most of what such a source costs. Timestamps, metadata and side data pass through;
// `frame_color` tags the frames as a `setparams` filter would.
class Uyvy422ToCuda: public NodeSISO<av::VideoFrame, av::VideoFrame>, public IVideoFormatSource,
                     public ReportsFinishByFlag, public IReturnsObjects {
protected:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    FrameColorTags color_;
    std::unique_ptr<CudaPinnedUploader> uploader_;
    int width_ = 0, height_ = 0;
    bool interlaced_ = false;
    std::atomic<uint64_t> frames_{0}, copy_us_{0}, upload_us_{0};

public:
    using NodeSISO::NodeSISO;

    void process() override {
        using Clock = std::chrono::steady_clock;
        av::VideoFrame in = this->source_->get();
        if (isEofMarker(in)) {
            this->sink_->put(in);
            markFinished();
            return;
        }
        if (!in) return;
        const AVFrame *src = in.raw();
        if (src->format != AV_PIX_FMT_UYVY422)
            throw Error(std::string("uyvy422_to_cuda: frames must be uyvy422, not ") +
                        av_get_pix_fmt_name(static_cast<AVPixelFormat>(src->format)));
        const bool interlaced = src->flags & AV_FRAME_FLAG_INTERLACED;
        if (!uploader_ || src->width != width_ || src->height != height_ || interlaced != interlaced_) {
            uploader_.reset();
            uploader_ = std::make_unique<CudaPinnedUploader>(hwaccel_, AV_PIX_FMT_NV12, src->width, src->height, false, 1, 0,
                                                             PackedPicture::Uyvy422, interlaced);
            width_ = src->width;
            height_ = src->height;
            interlaced_ = interlaced;
        }
        const auto started = Clock::now();
        av_image_copy_plane(uploader_->block(), uploader_->layout().row_bytes[0], src->data[0], src->linesize[0],
                            av_image_get_linesize(AV_PIX_FMT_UYVY422, width_, 0), height_);
        const auto copied = Clock::now();
        av::VideoFrame out = uploader_->upload();
        const auto uploaded = Clock::now();
        if (const int r = av_frame_copy_props(out.raw(), src); r < 0)
            throw Error("uyvy422_to_cuda: av_frame_copy_props: " + av::error2string(r));
        out.setTimeBase(in.timeBase());
        color_.apply(out.raw());
        out.setComplete(true);
        ++frames_;
        copy_us_ = std::chrono::duration_cast<std::chrono::microseconds>(copied - started).count();
        upload_us_ = std::chrono::duration_cast<std::chrono::microseconds>(uploaded - copied).count();
        this->sink_->put(out);
    }

    // The picture size is the upstream's: frames keep it.
    int width() override { return upstreamFormat()->width(); }
    int height() override { return upstreamFormat()->height(); }
    av::PixelFormat pixelFormat() override { return AV_PIX_FMT_CUDA; }
    av::PixelFormat realPixelFormat() override { return AV_PIX_FMT_NV12; }

    Parameters getObject(const std::string key) override {
        if (key != "stats") throw Error("uyvy422_to_cuda: unknown object " + key);
        return {{"frames", frames_.load()}, {"copy_us", copy_us_.load()}, {"upload_us", upload_us_.load()}};
    }

    static std::shared_ptr<Uyvy422ToCuda> create(NodeCreationInfo &nci) {
        const Parameters &params = nci.params;
        if (!params.count("hwaccel")) throw Error("uyvy422_to_cuda requires hwaccel");
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<Uyvy422ToCuda>(nci.edges, params);
        if (params.count("frame_color")) r->color_ = FrameColorTags::parse(params["frame_color"]);
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_ || r->hwaccel_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error("uyvy422_to_cuda: hwaccel must name a CUDA device");
        return r;
    }

private:
    std::shared_ptr<IVideoFormatSource> upstreamFormat() {
        auto format = this->findNodeUp<IVideoFormatSource>();
        if (!format) throw Error("uyvy422_to_cuda: nothing upstream tells the picture size");
        return format;
    }
};

DECLNODE(uyvy422_to_cuda, Uyvy422ToCuda);
