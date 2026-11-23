#include "../node_common.hpp"
#include "../../frame_color_tags.hpp"
#include "CudaPinnedUploader.hpp"

#include <atomic>
#include <chrono>

extern "C" {
#include <libavutil/csp.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

// Packed frames in host memory as CUDA frames of one `storage`: UYVY (8-bit 4:2:2), what an NDI
// receiver or a capture card delivers, and 8-bit RGB with a fourth byte (bgra, bgr0, rgba, rgb0),
// what an NDI receiver delivers for a sender with alpha. Each picture is copied once, as it is,
// into page-locked memory; the device takes it from there and unpacks it into the frame, so no CPU
// converts a sample. A filter graph for the same (`format=nv12,hwupload_cuda`) converts every
// picture with swscale first, which is most of what such a source costs.
//
// `storage` nv12 (the default) is for a source shown as video: RGB frames lose their alpha.
// `storage` bgra is for one keyed or blended by its alpha: bgra frames go to the device as they
// are, with no kernel, and frames without alpha (UYVY, bgr0, rgb0) become opaque bgra. What a
// sender delivers is known only from its frames, so the frames' format is the caller's choice and
// never the sender's.
//
// RGB and Y'CbCr are converted by the matrix and range `frame_color` declares (`colorspace`,
// `color_range`), BT.709 limited range without them. FFmpeg's convert_cuda filter (deps/ffmpeg)
// converts on the device too, but a filter graph is built for one input format and wants CUDA
// frames: this node takes each frame as it comes.
//
// Timestamps, metadata and side data pass through; `frame_color` tags the frames as a `setparams`
// filter would.
class PackedToCuda: public NodeSISO<av::VideoFrame, av::VideoFrame>, public IVideoFormatSource,
                    public ReportsFinishByFlag, public IReturnsObjects {
protected:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    FrameColorTags color_;
    std::unique_ptr<CudaPinnedUploader> uploader_;
    AVPixelFormat storage_ = AV_PIX_FMT_NV12;
    AVPixelFormat format_ = AV_PIX_FMT_NONE;
    int width_ = 0, height_ = 0;
    bool interlaced_ = false;
    std::atomic<uint64_t> frames_{0}, copy_us_{0}, upload_us_{0};

    // How frames of `format` are packed for the device: not at all when they are the storage's.
    PackedPicture packing(AVPixelFormat format) const {
        if (format == storage_) return PackedPicture::None;
        switch (format) {
        case AV_PIX_FMT_UYVY422: return PackedPicture::Uyvy422;
        case AV_PIX_FMT_BGRA: case AV_PIX_FMT_BGR0: return PackedPicture::Bgr32;
        case AV_PIX_FMT_RGBA: case AV_PIX_FMT_RGB0: return PackedPicture::Rgb32;
        default: throw Error(std::string("packed_to_cuda: frames must be uyvy422, bgra, bgr0, rgba or rgb0, not ") +
                             av_get_pix_fmt_name(format));
        }
    }
    // The declared matrix and range, and whether frames of `format` have a fourth byte that is not alpha.
    PackedConversion conversion(AVPixelFormat format) const {
        PackedConversion conversion;
        if (const AVLumaCoefficients *luma = color_.space >= 0 ? av_csp_luma_coeffs_from_avcsp(static_cast<AVColorSpace>(color_.space)) : nullptr) {
            conversion.kr = float(av_q2d(luma->cr));
            conversion.kb = float(av_q2d(luma->cb));
        }
        conversion.full_range = color_.range == AVCOL_RANGE_JPEG;
        conversion.opaque = format == AV_PIX_FMT_BGR0 || format == AV_PIX_FMT_RGB0;
        return conversion;
    }

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
        const auto format = static_cast<AVPixelFormat>(src->format);
        const bool interlaced = src->flags & AV_FRAME_FLAG_INTERLACED;
        if (!uploader_ || format != format_ || src->width != width_ || src->height != height_ || interlaced != interlaced_) {
            const PackedPicture packed = packing(format);
            uploader_.reset();
            uploader_ = std::make_unique<CudaPinnedUploader>(hwaccel_, storage_, src->width, src->height, false, 1, 0,
                                                             packed, interlaced, conversion(format));
            format_ = format;
            width_ = src->width;
            height_ = src->height;
            interlaced_ = interlaced;
        }
        const auto started = Clock::now();
        av_image_copy_plane(uploader_->block(), uploader_->layout().row_bytes[0], src->data[0], src->linesize[0],
                            av_image_get_linesize(format, width_, 0), height_);
        const auto copied = Clock::now();
        av::VideoFrame out = uploader_->upload();
        const auto uploaded = Clock::now();
        if (const int r = av_frame_copy_props(out.raw(), src); r < 0)
            throw Error("packed_to_cuda: av_frame_copy_props: " + av::error2string(r));
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
    av::PixelFormat realPixelFormat() override { return storage_; }

    Parameters getObject(const std::string key) override {
        if (key != "stats") throw Error("packed_to_cuda: unknown object " + key);
        return {{"frames", frames_.load()}, {"copy_us", copy_us_.load()}, {"upload_us", upload_us_.load()},
                {"pixel_format", format_ == AV_PIX_FMT_NONE ? "" : av_get_pix_fmt_name(format_)}};
    }

    static std::shared_ptr<PackedToCuda> create(NodeCreationInfo &nci) {
        const Parameters &params = nci.params;
        if (!params.count("hwaccel")) throw Error("packed_to_cuda requires hwaccel");
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<PackedToCuda>(nci.edges, params);
        if (params.count("frame_color")) r->color_ = FrameColorTags::parse(params["frame_color"]);
        const std::string storage = params.value("storage", "nv12");
        r->storage_ = av_get_pix_fmt(storage.c_str());
        if (r->storage_ != AV_PIX_FMT_NV12 && r->storage_ != AV_PIX_FMT_BGRA)
            throw Error("packed_to_cuda: storage must be nv12 or bgra, not " + storage);
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_ || r->hwaccel_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error("packed_to_cuda: hwaccel must name a CUDA device");
        return r;
    }

private:
    std::shared_ptr<IVideoFormatSource> upstreamFormat() {
        auto format = this->findNodeUp<IVideoFormatSource>();
        if (!format) throw Error("packed_to_cuda: nothing upstream tells the picture size");
        return format;
    }
};

DECLNODE(packed_to_cuda, PackedToCuda);
