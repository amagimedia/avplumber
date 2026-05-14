#include "node_common.hpp"
#include "../video_parameters.hpp"
#include <fstream>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

class MetadataDrivenCudaCrop: public NodeSISO<av::VideoFrame, av::VideoFrame>,
                              public IVideoFormatSource,
                              public IFrameRateSource,
                              public ITimeBaseSource,
                              public ReportsFinishByFlag {
private:
    std::string metadata_key_ = "reframer_bbox";
    std::string offset_log_path_ = "reframer.log";
    int dst_width_ = 0;
    int dst_height_ = 0;
    int fallback_dst_width_ = 0;
    int fallback_dst_height_ = 0;
    int debug_log_every_n_ = 0;
    std::ofstream offset_log_;

    VideoParameters input_params_{};
    av::Rational frame_rate_{0, 0};
    av::Rational timebase_{0, 0};

    AVFilterGraph *filter_graph_ = nullptr;
    AVFilterContext *buffersrc_ctx_ = nullptr;
    AVFilterContext *crop_ctx_ = nullptr;
    AVFilterContext *buffersink_ctx_ = nullptr;
    AVBufferRef *initial_hw_frames_ctx_ = nullptr;

    int last_crop_x_ = 0;
    int last_crop_y_ = 0;
    bool have_last_crop_ = false;
    int last_applied_x_ = INT_MIN;
    int last_applied_y_ = INT_MIN;
    uint64_t frame_counter_ = 0;

    void freeFilterGraph() {
        if (filter_graph_) {
            avfilter_graph_free(&filter_graph_);
            filter_graph_ = nullptr;
        }
        buffersrc_ctx_ = nullptr;
        crop_ctx_ = nullptr;
        buffersink_ctx_ = nullptr;
        last_applied_x_ = INT_MIN;
        last_applied_y_ = INT_MIN;
    }

    static bool hwFramesCtxSemanticallyEqual(const AVBufferRef *prev_ref, const AVBufferRef *cur_ref) {
        if (!(prev_ref && prev_ref->data && cur_ref && cur_ref->data)) return false;

        const AVHWFramesContext *prev = (const AVHWFramesContext *)prev_ref->data;
        const AVHWFramesContext *cur = (const AVHWFramesContext *)cur_ref->data;

        return cur->format == prev->format
            && cur->sw_format == prev->sw_format
            && cur->width == prev->width
            && cur->height == prev->height
            && cur->device_ref
            && prev->device_ref
            && cur->device_ref->data == prev->device_ref->data;
    }

    bool refreshHWFramesCtxFromFrame(const av::VideoFrame &frm) {
        const AVFrame *raw = frm.raw();
        if (!(raw && raw->hw_frames_ctx && raw->hw_frames_ctx->data)) return false;

        if (!initial_hw_frames_ctx_) {
            initial_hw_frames_ctx_ = av_buffer_ref(raw->hw_frames_ctx);
            if (!initial_hw_frames_ctx_) {
                throw Error("crop_metadata_cuda: failed to reference input hw_frames_ctx");
            }
            return false;
        }

        if (hwFramesCtxSemanticallyEqual(initial_hw_frames_ctx_, raw->hw_frames_ctx)) {
            return false;
        }

        const AVHWFramesContext *prev = (const AVHWFramesContext *)initial_hw_frames_ctx_->data;
        const AVHWFramesContext *cur = (const AVHWFramesContext *)raw->hw_frames_ctx->data;
        logstream << "crop_metadata_cuda: hw_frames_ctx changed (semantic mismatch:"
                  << " fmt " << prev->format << "->" << cur->format
                  << " sw_fmt " << prev->sw_format << "->" << cur->sw_format
                  << " size " << prev->width << "x" << prev->height
                  << "->" << cur->width << "x" << cur->height << ")";

        av_buffer_unref(&initial_hw_frames_ctx_);
        initial_hw_frames_ctx_ = av_buffer_ref(raw->hw_frames_ctx);
        if (!initial_hw_frames_ctx_) {
            throw Error("crop_metadata_cuda: failed to reference changed input hw_frames_ctx");
        }
        return true;
    }

    std::string buildSourceArgsString() const {
        if (timebase_.getNumerator() == 0 || timebase_.getDenominator() == 0) {
            throw Error("crop_metadata_cuda: unknown input timebase");
        }

        av::Rational frame_rate = frame_rate_;
        if (frame_rate.getNumerator() == 0 || frame_rate.getDenominator() == 0) {
            frame_rate = av::Rational(timebase_.getDenominator(), timebase_.getNumerator());
        }
        if (frame_rate.getNumerator() == 0 || frame_rate.getDenominator() == 0) {
            frame_rate = av::Rational(30, 1);
        }

        std::stringstream ss;
        ss << "video_size=" << input_params_.width << "x" << input_params_.height
           << ":pix_fmt=" << static_cast<int>(input_params_.pixel_format.get())
           << ":pixel_aspect=1/1"
           << ":time_base=" << timebase_.getNumerator() << "/" << timebase_.getDenominator()
           << ":frame_rate=" << frame_rate;
        return ss.str();
    }

    void initFilterGraph() {
        freeFilterGraph();

        if (!initial_hw_frames_ctx_ || !initial_hw_frames_ctx_->data) {
            throw Error("crop_metadata_cuda: missing input hw_frames_ctx");
        }
        if (input_params_.pixel_format != AV_PIX_FMT_CUDA) {
            throw Error("crop_metadata_cuda: input must be AV_PIX_FMT_CUDA");
        }
        if (dst_width_ <= 0 || dst_height_ <= 0) {
            throw Error("crop_metadata_cuda: dst_width and dst_height must be positive");
        }
        if (dst_width_ > input_params_.width || dst_height_ > input_params_.height) {
            throw Error("crop_metadata_cuda: crop size exceeds input frame size");
        }

        filter_graph_ = avfilter_graph_alloc();
        if (!filter_graph_) {
            throw Error("crop_metadata_cuda: avfilter_graph_alloc failed");
        }

        const AVFilter *buffersrc = avfilter_get_by_name("buffer");
        const AVFilter *crop = avfilter_get_by_name("crop_cuda");
        const AVFilter *buffersink = avfilter_get_by_name("buffersink");
        if (!buffersrc || !crop || !buffersink) {
            throw Error("crop_metadata_cuda: required FFmpeg filters not available");
        }

        const std::string in_args = buildSourceArgsString();
        int ret = avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in", in_args.c_str(), nullptr, filter_graph_);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: couldn't create buffer source");
        }

        AVBufferSrcParameters *src_params = av_buffersrc_parameters_alloc();
        if (!src_params) {
            throw Error("crop_metadata_cuda: av_buffersrc_parameters_alloc failed");
        }
        src_params->hw_frames_ctx = av_buffer_ref(initial_hw_frames_ctx_);
        ret = av_buffersrc_parameters_set(buffersrc_ctx_, src_params);
        av_buffer_unref(&src_params->hw_frames_ctx);
        av_freep(&src_params);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: av_buffersrc_parameters_set failed");
        }

        std::stringstream crop_args;
        crop_args << "w=" << dst_width_ << ":h=" << dst_height_
                  << ":x=" << last_crop_x_ << ":y=" << last_crop_y_;
        ret = avfilter_graph_create_filter(&crop_ctx_, crop, "cropper", crop_args.str().c_str(), nullptr, filter_graph_);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: couldn't create crop_cuda filter");
        }

        ret = avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out", nullptr, nullptr, filter_graph_);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: couldn't create buffer sink");
        }

        ret = avfilter_link(buffersrc_ctx_, 0, crop_ctx_, 0);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: couldn't link buffer source to crop_cuda");
        }
        ret = avfilter_link(crop_ctx_, 0, buffersink_ctx_, 0);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: couldn't link crop_cuda to buffer sink");
        }

        ret = avfilter_graph_config(filter_graph_, nullptr);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: avfilter_graph_config failed");
        }

        last_applied_x_ = last_crop_x_;
        last_applied_y_ = last_crop_y_;
    }

    bool loadCropMetadata(const av::VideoFrame &frm, Parameters &md_out) const {
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry *entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;
        try {
            md_out = Parameters::parse(entry->value);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    void validateAndSetDstSize(int w, int h, int frame_w, int frame_h) {
        if (w <= 0 || h <= 0) {
            throw Error("crop_metadata_cuda: crop width and height must be positive");
        }
        if ((w & 1) || (h & 1)) {
            throw Error("crop_metadata_cuda: crop width and height must be even");
        }
        if (w > frame_w || h > frame_h) {
            throw Error("crop_metadata_cuda: crop size exceeds input frame size");
        }
        dst_width_ = w;
        dst_height_ = h;
    }

    void updateDstDimensionsFromMetadata(const Parameters &md, int frame_w, int frame_h) {
        if (md.contains("viewport_dst_width") && md.contains("viewport_dst_height")) {
            const int w = md["viewport_dst_width"].get<int>();
            const int h = md["viewport_dst_height"].get<int>();
            validateAndSetDstSize(w, h, frame_w, frame_h);
            return;
        }
        if (fallback_dst_width_ > 0 && fallback_dst_height_ > 0) {
            validateAndSetDstSize(fallback_dst_width_, fallback_dst_height_, frame_w, frame_h);
            return;
        }
        throw Error("crop_metadata_cuda: metadata missing viewport_dst_width/viewport_dst_height "
                    "(set dst_width/dst_height on the node if the producer does not emit them)");
    }

    int clampCropX(int x) const {
        const int max_x = std::max(0, input_params_.width - dst_width_);
        const int clamped = std::max(0, std::min(x, max_x));
        return alignCropCoord(clamped, chromaXAlign());
    }

    int clampCropY(int y) const {
        const int max_y = std::max(0, input_params_.height - dst_height_);
        const int clamped = std::max(0, std::min(y, max_y));
        return alignCropCoord(clamped, chromaYAlign());
    }

    int chromaXAlign() const {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(input_params_.realPixelFormat().get());
        if (!desc || desc->log2_chroma_w < 0) return 1;
        return 1 << desc->log2_chroma_w;
    }

    int chromaYAlign() const {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(input_params_.realPixelFormat().get());
        if (!desc || desc->log2_chroma_h < 0) return 1;
        return 1 << desc->log2_chroma_h;
    }

    static int alignCropCoord(int value, int align) {
        if (align <= 1) return value;
        return value & ~(align - 1);
    }

    std::pair<int, int> centerCrop() const {
        return {
            clampCropX((input_params_.width - dst_width_) / 2),
            clampCropY((input_params_.height - dst_height_) / 2)
        };
    }

    bool parseCropPositionFromMd(const Parameters &md, int &x_out, int &y_out) const {
        double center_x = NAN;
        double center_y = NAN;

        if (md.contains("viewport_bbox") && md["viewport_bbox"].is_array() && md["viewport_bbox"].size() >= 4) {
            const auto &bbox = md["viewport_bbox"];
            const double x1 = bbox[0].get<double>();
            const double y1 = bbox[1].get<double>();
            const double x2 = bbox[2].get<double>();
            const double y2 = bbox[3].get<double>();
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else if (md.contains("viewport_center_x")) {
            center_x = md["viewport_center_x"].get<double>();
            center_y = input_params_.height * 0.5;
        } else if (md.contains("bbox_norm") && md["bbox_norm"].is_array() && md["bbox_norm"].size() >= 4) {
            const auto &bbox = md["bbox_norm"];
            const double fw = md.value("full_frame_width", input_params_.width);
            const double fh = md.value("full_frame_height", input_params_.height);
            const double x1 = bbox[0].get<double>() * fw;
            const double y1 = bbox[1].get<double>() * fh;
            const double x2 = bbox[2].get<double>() * fw;
            const double y2 = bbox[3].get<double>() * fh;
            center_x = (x1 + x2) * 0.5;
            center_y = (y1 + y2) * 0.5;
        } else {
            return false;
        }

        if (!std::isfinite(center_x) || !std::isfinite(center_y)) return false;

        x_out = clampCropX((int)std::lround(center_x - (double)dst_width_ * 0.5));
        y_out = clampCropY((int)std::lround(center_y - (double)dst_height_ * 0.5));
        return true;
    }

    void updateCropPosition(const Parameters &md) {
        int next_x = 0;
        int next_y = 0;
        bool parsed = parseCropPositionFromMd(md, next_x, next_y);

        if (!parsed) {
            if (have_last_crop_) {
                next_x = last_crop_x_;
                next_y = last_crop_y_;
            } else {
                std::tie(next_x, next_y) = centerCrop();
            }
        }

        last_crop_x_ = next_x;
        last_crop_y_ = next_y;
        have_last_crop_ = true;
    }

    void applyCropCommandsIfNeeded() {
        if (!crop_ctx_) {
            throw Error("crop_metadata_cuda: crop filter not initialized");
        }

        auto sendCommand = [&](const char *cmd, int value) {
            char response[256] = {};
            const int ret = avfilter_process_command(crop_ctx_, cmd, std::to_string(value).c_str(), response, sizeof(response), 0);
            if (ret < 0) {
                throw Error(std::string("crop_metadata_cuda: failed updating crop_cuda ") + cmd);
            }
        };

        if (last_crop_x_ != last_applied_x_) {
            sendCommand("x", last_crop_x_);
            last_applied_x_ = last_crop_x_;
        }
        if (last_crop_y_ != last_applied_y_) {
            sendCommand("y", last_crop_y_);
            last_applied_y_ = last_crop_y_;
        }
    }

    void maybeLogFrame() const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "crop_metadata_cuda: frame=" << frame_counter_
                  << " x=" << last_crop_x_
                  << " y=" << last_crop_y_
                  << " w=" << dst_width_
                  << " h=" << dst_height_;
    }

    void writeOffsetLog() {
        if (!offset_log_.is_open()) return;
        offset_log_ << last_crop_x_ << "\n";
        offset_log_.flush();
    }

public:
    bool consumeEofIfPresent() override {
        return false;
    }

    MetadataDrivenCudaCrop(std::unique_ptr<Source<av::VideoFrame>> &&source,
                           std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                           std::string metadata_key,
                           std::string offset_log_path,
                           int fallback_dst_width,
                           int fallback_dst_height,
                           av::Rational frame_rate,
                           av::Rational timebase,
                           int debug_log_every_n)
        : NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          offset_log_path_(std::move(offset_log_path)),
          fallback_dst_width_(fallback_dst_width),
          fallback_dst_height_(fallback_dst_height),
          debug_log_every_n_(debug_log_every_n),
          frame_rate_(frame_rate),
          timebase_(timebase) {
        if ((fallback_dst_width_ > 0) != (fallback_dst_height_ > 0)) {
            throw Error("crop_metadata_cuda: dst_width and dst_height must both be set or both zero");
        }
        if (fallback_dst_width_ > 0 && ((fallback_dst_width_ & 1) || (fallback_dst_height_ & 1))) {
            throw Error("crop_metadata_cuda: dst_width and dst_height must be even");
        }
        offset_log_.open(offset_log_path_, std::ios::out | std::ios::trunc);
        if (!offset_log_.is_open()) {
            throw Error("crop_metadata_cuda: failed to open offset log file " + offset_log_path_);
        }
    }

    ~MetadataDrivenCudaCrop() override {
        freeFilterGraph();
        if (initial_hw_frames_ctx_) {
            av_buffer_unref(&initial_hw_frames_ctx_);
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            throw Error("crop_metadata_cuda: non-CUDA frame received");
        }

        Parameters md;
        if (!loadCropMetadata(frm, md)) {
            throw Error("crop_metadata_cuda: missing metadata key " + metadata_key_);
        }

        const VideoParameters vp(frm);
        const int prev_dw = dst_width_;
        const int prev_dh = dst_height_;
        updateDstDimensionsFromMetadata(md, vp.width, vp.height);
        const bool dst_changed = (dst_width_ != prev_dw || dst_height_ != prev_dh);

        const bool hw_frames_ctx_changed = refreshHWFramesCtxFromFrame(frm);
        const bool input_changed = input_params_ != vp || frm.timeBase() != timebase_;
        if (input_changed || hw_frames_ctx_changed || dst_changed || !filter_graph_) {
            input_params_ = vp;
            timebase_ = frm.timeBase();
            if (frame_rate_.getNumerator() == 0 || frame_rate_.getDenominator() == 0) {
                if (timebase_.getNumerator() > 0 && timebase_.getDenominator() > 0) {
                    frame_rate_ = av::Rational(timebase_.getDenominator(), timebase_.getNumerator());
                }
                if (frame_rate_.getNumerator() == 0 || frame_rate_.getDenominator() == 0) {
                    frame_rate_ = av::Rational(30, 1);
                }
            }
            if (dst_changed || !have_last_crop_) {
                std::tie(last_crop_x_, last_crop_y_) = centerCrop();
                have_last_crop_ = true;
            }
            initFilterGraph();
        }

        updateCropPosition(md);
        applyCropCommandsIfNeeded();
        writeOffsetLog();

        int ret = av_buffersrc_add_frame_flags(buffersrc_ctx_, frm.raw(), 0);
        if (ret < 0) {
            throw Error("crop_metadata_cuda: error feeding filter graph: " + av::error2string(ret));
        }

        av::VideoFrame out;
        ret = av_buffersink_get_frame(buffersink_ctx_, out.raw());
        if (ret < 0) {
            throw Error("crop_metadata_cuda: error receiving filtered frame: " + av::error2string(ret));
        }

        out.setTimeBase(buffersink_ctx_->inputs[0]->time_base);
        out.setComplete(true);
        maybeLogFrame();
        this->sink_->put(out);
    }

    int width() override {
        return dst_width_;
    }

    int height() override {
        return dst_height_;
    }

    av::PixelFormat pixelFormat() override {
        return input_params_.pixel_format == AV_PIX_FMT_NONE ? av::PixelFormat(AV_PIX_FMT_CUDA) : input_params_.pixel_format;
    }

    av::PixelFormat realPixelFormat() override {
        return input_params_.real_pixel_format == AV_PIX_FMT_NONE ? pixelFormat() : input_params_.real_pixel_format;
    }

    av::Rational frameRate() override {
        return frame_rate_;
    }

    av::Rational timeBase() override {
        return timebase_;
    }

    static std::shared_ptr<MetadataDrivenCudaCrop> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto frame_rate_src = src_edge->findNodeUp<IFrameRateSource>();
        auto timebase_src = src_edge->findNodeUp<ITimeBaseSource>();

        const std::string metadata_key = params.value("metadata_key", std::string("reframer_bbox"));
        const std::string offset_log_path = params.value("offset_log_path", std::string("reframer.log"));
        const int fallback_dw = params.value("dst_width", 0);
        const int fallback_dh = params.value("dst_height", 0);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const av::Rational frame_rate = frame_rate_src ? frame_rate_src->frameRate() : av::Rational{0, 0};
        const av::Rational timebase = timebase_src ? timebase_src->timeBase() : av::Rational{0, 0};

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDrivenCudaCrop>(
            edges, params, metadata_key, offset_log_path, fallback_dw, fallback_dh, frame_rate, timebase,
            debug_log_every_n);
    }
};

DECLNODE(crop_metadata_cuda, MetadataDrivenCudaCrop);
