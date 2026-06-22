#include "node_common.hpp"
extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/pixdesc.h>
}
#include "../ts_equalizer.hpp"
#include "../video_parameters.hpp"
#include "../audio_parameters.hpp"
#include "../hwaccel.hpp"
#include <avcpp/channellayout.h>

template<typename T> struct FilterMediaSpecific {
};

template<> struct FilterMediaSpecific<av::VideoFrame> {
    using Parameters = VideoParameters;
    using NodeInterface = IVideoFormatSource;
    static constexpr const char* source_filter_name = "buffer";
    static constexpr const char* sink_filter_name = "buffersink";
    static constexpr bool default_do_shift = false;
    VideoParameters par_;
    std::string getSourceArgsString(VideoParameters params, std::shared_ptr<Edge<av::VideoFrame>> edge) {
        std::shared_ptr<IFrameRateSource> vfr = edge->findNodeUp<IFrameRateSource>();
        if (vfr == nullptr) {
            throw Error("Unknown input video frame rate.");
        }
        par_ = params;
        
        std::stringstream ss;
        // FIXME: we're supporting only square pixels. Appearently avcpp doesn't have interface for changing pixel aspect ratio
        ss << "video_size=" << par_.width << "x" << par_.height << ":pix_fmt=" << static_cast<int>(par_.pixel_format.get()) << ":pixel_aspect=1/1:frame_rate=" << vfr->frameRate();
        return ss.str();
    }
    /* bool checkFrame(av::VideoFrame &frm) {
        return par_ == VideoParameters(frm);
    } */
    bool checkParameters(VideoParameters &p) {
        return par_ == p;
    }
    static Parameters parametersFromNodeInterface(NodeInterface &ni) {
        return ni.videoParameters();
    }
    void initHWAccel(AVHWFramesContext &frmctx, AVPixelFormat hw_format) {
        frmctx.sw_format = par_.real_pixel_format;
        frmctx.width = par_.width;
        frmctx.height = par_.height;
        frmctx.format = hw_format;
    }
    bool inputIsHWFormat() const {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(par_.pixel_format.get());
        return desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL);
    }
};
template<> struct FilterMediaSpecific<av::AudioSamples> {
    using Parameters = AudioParameters;
    using NodeInterface = IAudioMetadataSource;
    static constexpr const char* source_filter_name = "abuffer";
    static constexpr const char* sink_filter_name = "abuffersink";
    static constexpr bool default_do_shift = true;
    AudioParameters par_;
    std::string getSourceArgsString(AudioParameters params, std::shared_ptr<Edge<av::AudioSamples>> edge) {
        par_ = params;
        
        std::stringstream ss;
        ss << "sample_rate=" << par_.sample_rate << ":sample_fmt=" << par_.sample_format.name() << ":channel_layout=0x" << std::hex << par_.channel_layout;
        return ss.str();
    }
    /* bool checkFrame(av::AudioSamples &frm) {
        return par_ == AudioParameters(frm);
    } */
    bool checkParameters(AudioParameters &p) {
        return par_ == p;
    }
    static Parameters parametersFromNodeInterface(NodeInterface &ni) {
        return ni.audioParameters();
    }
    void initHWAccel(AVHWFramesContext&, AVPixelFormat) {
        throw Error("hwaccel specified for audio filter");
    }
    bool inputIsHWFormat() const {
        return false;
    }
};

template<typename Child, typename T, AVMediaType media_type> class FilterNode: public NodeMultiInput<T>, public NodeMultiOutput<T>, public ReportsFinishByFlag, public ITimeBaseSource {
    friend struct FilterMediaSpecific<T>;
protected:
    using MediaSpecific = FilterMediaSpecific<T>;
    static AVFilterInOut* appendFilterInOut(AVFilterInOut* prev, AVFilterContext* filter_ctx, const std::string name) {
        AVFilterInOut* inout = avfilter_inout_alloc();
        inout->name = av_strdup(name.c_str());
        inout->filter_ctx = filter_ctx;
        inout->pad_idx = 0;
        inout->next = nullptr;
        if (prev!=nullptr) prev->next = inout;
        return inout;
    }
    class Port {
        MediaSpecific ms_;
        av::Rational prev_tb_{0, 0};
        AVFilterContext* ctx_ = nullptr;
        std::string in_args_;
        AVBufferRef* initial_hw_frames_ctx_ = nullptr;
        bool eof_closed_ = false;
    public:
        Port() {};
        ~Port() {
            if (initial_hw_frames_ctx_) {
                av_buffer_unref(&initial_hw_frames_ctx_);
                initial_hw_frames_ctx_ = nullptr;
            }
        }
        bool checkParameters(typename MediaSpecific::Parameters params, AVFrame *raw, av::Rational timebase, std::shared_ptr<Edge<T>> edge) {
            // Detect hw_frames_ctx changes by comparing the semantic content of
            // the AVHWFramesContext (format, sw_format, device),
            // not the pointer value. NVENC and hwupload_cuda can rotate pool
            // objects (new AVBufferRef->data) while keeping the same format and
            // size, and pointer comparison would trigger unnecessary rebuilds.
            //
            // Do not include AVHWFramesContext width/height here. For CUDA frames
            // those fields can describe the aligned allocation size (for example
            // 1088x1920) rather than the visible frame size (1080x1920). The
            // visible frame parameters are checked separately by ms_.checkParameters
            // and are what the buffersrc args use.
            bool hw_frames_ctx_changed = false;
            if (raw && raw->hw_frames_ctx && raw->hw_frames_ctx->data) {
                if (!initial_hw_frames_ctx_) {
                    // First time we see a hw_frames_ctx: not a "change" per se
                    // (initial state was null), but we must update in_args_ and
                    // rebuild because the buffersrc was created without it.
                    hw_frames_ctx_changed = true;
                    logstream << "hw frames ctx appeared, was null";
                } else {
                    const AVHWFramesContext *prev = (const AVHWFramesContext *)initial_hw_frames_ctx_->data;
                    const AVHWFramesContext *cur  = (const AVHWFramesContext *)raw->hw_frames_ctx->data;
                    bool semantically_equal =
                        (cur->format    == prev->format) &&
                        (cur->sw_format == prev->sw_format) &&
                        (cur->device_ref && prev->device_ref &&
                         cur->device_ref->data == prev->device_ref->data);
                    if (!semantically_equal) {
                        hw_frames_ctx_changed = true;
                        logstream << "hw frames ctx changed (semantic mismatch:"
                                  << " fmt " << prev->format << "->" << cur->format
                                  << " sw_fmt " << prev->sw_format << "->" << cur->sw_format
                                  << " size " << prev->width << "x" << prev->height
                                  << "->" << cur->width << "x" << cur->height << ")";
                        av_buffer_unref(&initial_hw_frames_ctx_);
                        initial_hw_frames_ctx_ = nullptr;
                    }
                    // else: pool rotation with same semantics — transparent, no rebuild
                }
            }
            bool result = (timebase==prev_tb_) && (!hw_frames_ctx_changed) && ms_.checkParameters(params);
            if (!result) {
                std::stringstream args_stream;
                args_stream << "time_base=" << timebase.getNumerator() << "/" << timebase.getDenominator() << ":" << ms_.getSourceArgsString(params, edge);
                in_args_ = args_stream.str();
                logstream << "Filter input args: " << in_args_;
                prev_tb_ = timebase;
            }
            return result;
        }
        bool checkFrame(T &frm, std::shared_ptr<Edge<T>> edge) {
            return checkParameters(typename MediaSpecific::Parameters(frm), frm.raw(), frm.timeBase(), edge);
        }
        void captureInitialHWFramesCtxFromFrame(T &frm) {
            AVFrame *raw = frm.raw();
            if (!initial_hw_frames_ctx_ && raw && raw->hw_frames_ctx && raw->hw_frames_ctx->data) {
                initial_hw_frames_ctx_ = av_buffer_ref(raw->hw_frames_ctx);
            }
        }
        bool isSourceReadyToInit() {
            return !in_args_.empty();
        }
        AVFilterContext* context() {
            return ctx_;
        }
        void invalidateFilterContext() {
            ctx_ = nullptr;
            eof_closed_ = false;
        }
        void initSourceFilter(const int index, AVFilterGraph *filter_graph, std::shared_ptr<HWAccelDevice> hwaccel, AVFilterInOut *dst) {
            std::string name = "in" + std::to_string(index);
            eof_closed_ = false;
            
            if (in_args_.empty()) {
                logstream << "Unable to init source filter " << name << ": in_args_ not initialized";
            }
            
            // create buffersrc filter
            const AVFilter* buffersrc = avfilter_get_by_name(ms_.source_filter_name);
            int ret = avfilter_graph_create_filter(&ctx_, buffersrc, name.c_str(), in_args_.c_str(), nullptr, filter_graph);
            if (ret < 0) {
                throw Error("Couldn't create buffer source");
            }
            
            ret = avfilter_link(ctx_, 0, dst->filter_ctx, dst->pad_idx);
            if (ret != 0) {
                throw Error("Couldn't link " + name);
            }
            
            // Prefer copying hw_frames_ctx from the first frame (if captured),
            // otherwise fall back to allocating a new hwframes context from the device
            if (initial_hw_frames_ctx_) {
                soft_assert(ctx_ != nullptr, "source context null");
                AVBufferSrcParameters* params = av_buffersrc_parameters_alloc();
                params->hw_frames_ctx = av_buffer_ref(initial_hw_frames_ctx_);
                soft_assert(params->hw_frames_ctx && params->hw_frames_ctx->data, "initial hw_frames_ctx null");
                av_buffersrc_parameters_set(ctx_, params);
                // av_buffersrc_parameters_set has increased the refcount, we should unref our temp ref
                av_buffer_unref(&params->hw_frames_ctx);
                av_freep(&params);
                // Keep our captured copy to detect future hw_frames_ctx changes reliably
            } else if (hwaccel && ms_.inputIsHWFormat()) {
                // Only set a HW hw_frames_ctx on the buffersrc when the upstream is already
                // delivering HW frames (e.g. cuda decoder output). For software inputs
                // (e.g. rgba from a PNG decoder), leave the buffersrc as software —
                // filter_graph_->hw_device_ctx (set in maybeInitFilterGraph) provides the
                // CUDA device to hwupload_cuda / hwupload filters further downstream.
                soft_assert(ctx_ != nullptr, "source context null");
                AVBufferSrcParameters* params = av_buffersrc_parameters_alloc();
                params->hw_frames_ctx = av_hwframe_ctx_alloc(hwaccel->deviceContext());
                soft_assert(params->hw_frames_ctx && params->hw_frames_ctx->data, "hw_frames_ctx null");
                
                AVHWFramesContext *frmctx = (AVHWFramesContext *)(params->hw_frames_ctx->data);
                ms_.initHWAccel(*frmctx, hwaccel->hardwarePixelFormat());
                int r = av_hwframe_ctx_init(params->hw_frames_ctx);
                if (r < 0) {
                    throw Error("av_hwframe_ctx_init failed: " + av::error2string(r));
                }
                av_buffersrc_parameters_set(ctx_, params);

                // av_buffersrc_parameters_set has increased the refcount, we should unref
                av_buffer_unref(&params->hw_frames_ctx);
                av_freep(&params);
            }
        }
        void initSinkFilter(const int index, AVFilterGraph *filter_graph, AVFilterInOut *src) {
            std::string name = "out" + std::to_string(index);
            
            const AVFilter* buffersink = avfilter_get_by_name(ms_.sink_filter_name);
            //AVBufferSinkParams* buffersink_params = av_buffersink_params_alloc();
            
            int ret = avfilter_graph_create_filter(&ctx_, buffersink, name.c_str(), nullptr, /*buffersink_params*/ nullptr, filter_graph);
            if (ret < 0) {
                throw Error("Couldn't create buffer sink");
            }
            
            ret = avfilter_link(src->filter_ctx, src->pad_idx, ctx_, 0);
            if (ret != 0) {
                throw Error("Couldn't link " + name);
            }
        }
        const AVFilterContext* getFilterContext() {
            return ctx_;
        }
        const AVFilterLink* getSinkLink() {
            return ctx_->inputs[0];
        }
        AVFilterLink* getSourceLink() {
            return ctx_->outputs[0];
        }
        bool checkSinkFilterMediaType() {
            return getSinkLink()->type == media_type;
        }
        int putFrame(T &frm) {
            if (!ctx_)
                throw Error("graph input count does not match node sources");
            return av_buffersrc_add_frame_flags(ctx_, frm.raw(), 0 /*AV_BUFFERSRC_FLAG_KEEP_REF*/);
        }
        int closeAtEof() {
            if (!ctx_ || eof_closed_)
                return 0;
            int ret = av_buffersrc_close(ctx_, AV_NOPTS_VALUE, 0);
            if (ret >= 0 || ret == AVERROR_EOF)
                eof_closed_ = true;
            return ret;
        }
        int getFrame(T &frm) {
            frm.setTimeBase(getSinkLink()->time_base);
            return av_buffersink_get_frame(ctx_, frm.raw());
        }
    };
    
    std::vector<Port> sinks_;
    std::vector<Port> sources_;
    AVFilterGraph *filter_graph_ = nullptr;
    const AVFilterContext* out_ctx_ = nullptr;
    TSEqualizer eq_;
    std::string graph_desc_;
    bool do_shift_ = true;
    bool defer_preliminary_init_ = false;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    std::vector<bool> input_eof_;

    void freeFilterGraph() {
        if (filter_graph_ == nullptr) return;
        avfilter_graph_free(&filter_graph_);
        filter_graph_ = nullptr;
        out_ctx_ = nullptr;
        for (Port &p : sources_)
            p.invalidateFilterContext();
        for (Port &p : sinks_)
            p.invalidateFilterContext();
    }
    bool drainFilterOutputs() {
        if (filter_graph_ == nullptr)
            return false;
        int finished_sinks = 0;
        for (int sink_index=0; sink_index<sinks_.size(); sink_index++) {
            Port& sink_port = sinks_[sink_index];
            while(true) {
                T frmout;
                int ret = sink_port.getFrame(frmout);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    if (ret==AVERROR_EOF) {
                        finished_sinks++;
                    }
                    break;
                }
                if (ret < 0) {
                    throw Error("Filtering error: " + std::to_string(ret));
                }
                if (!(frmout.isNull() || frmout.pts().isNoPts())) {
                    frmout.setComplete(true);
                    if (do_shift_) {
                        eq_.out(frmout);
                    }
                    if (!this->sink_edges_[sink_index]->enqueue(frmout)) {
                        this->finished_ = true;
                        return true;
                    }
                } else {
                    logstream << "WARNING: Invalid frame received from filter graph";
                }
            }
        }
        if (finished_sinks==sinks_.size()) {
            forwardEofToSinks();
            this->finished_ = true;
            return true;
        }
        return false;
    }
    void initPorts() {
        sources_.resize(this->source_edges_.size());
        sinks_.resize(this->sink_edges_.size());
        input_eof_.resize(this->source_edges_.size(), false);
    }
    bool maybeInitFilterGraph() {
        if (filter_graph_ != nullptr) {
            freeFilterGraph();
        }
        
        for (Port &port: sources_) {
            if (!port.isSourceReadyToInit()) {
                // not ready yet
                return false;
            }
        }
        
        filter_graph_ = avfilter_graph_alloc();
        
        AVFilterInOut* inputs = nullptr;
        AVFilterInOut* outputs = nullptr;
        
        int ret;
        ret = avfilter_graph_parse2(filter_graph_, graph_desc_.c_str(), &inputs, &outputs);
        if (ret < 0) {
            throw Error("Couldn't parse filter graph");
        }
        
        #ifdef AVFILTER_FLAG_HWDEVICE
        // Provide the hardware device to any filter in the graph that requests it
        // (e.g. hwupload_cuda, hwupload, scale_cuda). This lets those filters
        // allocate output HW frames even when the buffersrc is a software format.
        // ffmpeg 6.1+ is required for this to work.
        if (hwaccel_) {
            for (unsigned j = 0; j < filter_graph_->nb_filters; j++) {
                AVFilterContext *fctx = filter_graph_->filters[j];
                if (fctx->filter->flags & AVFILTER_FLAG_HWDEVICE) {
                    fctx->hw_device_ctx = hwaccel_->refDeviceContext();
                }
            }
        }
        #endif
        
        auto forEachInOut = [](AVFilterInOut *inout, std::function<void(AVFilterInOut*)> cb) {
            for (; inout != nullptr; inout = inout->next) {
                cb(inout);
            }
        };
        
        int i = 0;
        forEachInOut(inputs, [this, &i](AVFilterInOut* in) {
            if (i >= sources_.size()) {
                throw Error("Too many inputs in filtergraph");
            }
            
            sources_[i].initSourceFilter(i, filter_graph_, hwaccel_, in);
            i++;
        });
        if (i != (int)sources_.size()) {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
            freeFilterGraph();
            throw Error("Filter graph exposes " + std::to_string(i) + " input pad(s) but this node has " +
                std::to_string(sources_.size()) +
                " source(s). For filters with a configurable input count (e.g. overlay_many_cuda), "
                "set inputs=N in the graph string, e.g. overlay_many_cuda=inputs=3");
        }
        i = 0;
        forEachInOut(outputs, [this, &i](AVFilterInOut* out) {
            if (i >= sinks_.size()) {
                throw Error("Too many outputs in filtergraph");
            }
            sinks_[i].initSinkFilter(i, filter_graph_, out);
            i++;
        });
        if (i != (int)sinks_.size()) {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
            freeFilterGraph();
            throw Error("Filter graph exposes " + std::to_string(i) + " output pad(s) but this node has " +
                std::to_string(sinks_.size()) + " sink(s)");
        }
        avfilter_inout_free(&inputs);
        avfilter_inout_free(&outputs);
        
        ret = avfilter_graph_config(filter_graph_, nullptr);
        if (ret < 0) {
            throw Error("avfilter_graph_config error");
        }
        for (Port &port: sinks_) {
            if (!port.checkSinkFilterMediaType()) {
                freeFilterGraph();
                throw Error("Filter outputs invalid media type");
            }
        }
        if (sinks_.size()==1) {
            out_ctx_ = sinks_[0].getFilterContext();
        } else {
            freeFilterGraph();
            throw Error("Exactly one destination is needed");
        }
        return true;
    }
    void preliminaryInit() {
        assert(sources_.size() == this->source_edges_.size());
        for (int i=0; i<sources_.size(); i++) {
            auto edge = this->source_edges_[i];
            std::shared_ptr<typename MediaSpecific::NodeInterface> mdsrc = edge->template findNodeUp<typename MediaSpecific::NodeInterface>();
            if (mdsrc==nullptr) {
                logstream << "preliminary init: ignoring input " << i << ": no metadata source";
                continue;
            }
            std::shared_ptr<ITimeBaseSource> tbsrc = edge->template findNodeUp<ITimeBaseSource>();
            if (tbsrc==nullptr) {
                logstream << "preliminary init: ignoring input " << i << ": no timebase source";
                continue;
            }
            typename MediaSpecific::Parameters params = MediaSpecific::parametersFromNodeInterface(*mdsrc);
            sources_[i].checkParameters(params, nullptr, tbsrc->timeBase(), edge);
        }
        maybeInitFilterGraph();
    }
    bool allInputsEof() const {
        for (bool e : input_eof_) {
            if (!e) return false;
        }
        return !input_eof_.empty();
    }
    void drainAndFinish() {
        if (filter_graph_ == nullptr) {
            forwardEofToSinks();
            this->finished_ = true;
            return;
        }
        for (int sink_index = 0; sink_index < (int)sinks_.size(); sink_index++) {
            Port& sink_port = sinks_[sink_index];
            while (true) {
                T frmout;
                int ret = sink_port.getFrame(frmout);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    throw Error("Filtering error while draining: " + std::to_string(ret));
                }
                if (!(frmout.isNull() || frmout.pts().isNoPts())) {
                    frmout.setComplete(true);
                    if (do_shift_) {
                        eq_.out(frmout);
                    }
                    if (!this->sink_edges_[sink_index]->enqueue(frmout)) {
                        this->finished_ = true;
                        return;
                    }
                }
            }
        }
        forwardEofToSinks();
        this->finished_ = true;
    }
    void forwardEofToSinks() {
        for (int i = 0; i < (int)this->sink_edges_.size(); i++) {
            this->sink_edges_[i]->enqueue(createEofMarker<T>());
        }
    }
    bool pullSinkFrames() {
        int finished_sinks = 0;
        for (int sink_index = 0; sink_index < (int)sinks_.size(); sink_index++) {
            Port& sink_port = sinks_[sink_index];
            while (true) {
                T frmout;
                int ret = sink_port.getFrame(frmout);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    if (ret == AVERROR_EOF) {
                        finished_sinks++;
                    }
                    break;
                }
                if (ret < 0) {
                    throw Error("Filtering error: " + std::to_string(ret));
                }
                if (!(frmout.isNull() || frmout.pts().isNoPts())) {
                    frmout.setComplete(true);
                    if (do_shift_) {
                        eq_.out(frmout);
                    }
                    if (!this->sink_edges_[sink_index]->enqueue(frmout)) {
                        this->finished_ = true;
                        return true;
                    }
                } else {
                    logstream << "WARNING: Invalid frame received from filter graph";
                }
            }
        }
        if (finished_sinks == (int)sinks_.size()) {
            forwardEofToSinks();
            this->finished_ = true;
            return true;
        }
        return false;
    }
public:
    FilterNode(const std::string &graph_desc, const bool do_shift):
        graph_desc_(graph_desc), do_shift_(do_shift) {
        this->auto_eof_ = false;
    }
    virtual ~FilterNode() {
        freeFilterGraph();
    }
    virtual void process() {
        // Check for EOF markers on all inputs before normal processing
        for (int i = 0; i < (int)this->source_edges_.size(); i++) {
            if (input_eof_[i]) continue;
            T* p = this->source_edges_[i]->peek();
            if (p && isEofMarker(*p)) {
                this->source_edges_[i]->pop();
                input_eof_[i] = true;
                logstream << "EOF on filter input " << i;
                if (filter_graph_ != nullptr) {
                    int ret = sources_[i].closeAtEof();
                    if (ret < 0 && ret != AVERROR_EOF) {
                        throw Error("Error closing filter graph source: " + av::error2string(ret));
                    }
                }
            }
        }
        if (allInputsEof()) {
            drainAndFinish();
            return;
        }

        T* frmin = nullptr;
        int source_index = this->findSourceWithData();
        if (source_index >= 0) {
            std::shared_ptr<Edge<T>> edge = this->source_edges_[source_index];
            frmin = edge->peek();
            if (frmin && (!frmin->isNull()) && frmin->isComplete() && frmin->timeBase().getNumerator() && frmin->timeBase().getDenominator()) {
                Port &source_port = sources_[source_index];
                if (!source_port.checkFrame(*frmin, edge)) {
                    if (filter_graph_!=nullptr) {
                        logstream << "Input parameters changed. Restarting filter.";
                    }
                    freeFilterGraph();
                }
                source_port.captureInitialHWFramesCtxFromFrame(*frmin);
                if (filter_graph_==nullptr) {
                    maybeInitFilterGraph();
                }
                if (filter_graph_!=nullptr) {
                    if (do_shift_) {
                        eq_.in(*frmin);
                    }
                    int ret = source_port.putFrame(*frmin);
                    if (ret < 0 && ret != AVERROR(EAGAIN)) {
                        throw Error("Error feeding filter graph: " + av::error2string(ret));
                    } else if (ret >= 0) {
                        edge->pop(); // no need to retry, pop this frame
                    } else {
                        // AVERROR(EAGAIN): this buffersrc is not accepting yet. findSourceWithData()
                        // always prefers the smallest PTS among inputs; when timebases are not comparable,
                        // only that pad is ever fed and multi-input filters may never advance.
                        for (int j = 0; j < (int)this->source_edges_.size(); j++) {
                            if (j == source_index) continue;
                            std::shared_ptr<Edge<T>> e2 = this->source_edges_[j];
                            T* f2 = e2->peek();
                            if (!f2 || f2->isNull() || !f2->isComplete() ||
                                !f2->timeBase().getNumerator() || !f2->timeBase().getDenominator())
                                continue;
                            Port& p2 = sources_[j];
                            if (!p2.checkFrame(*f2, e2)) {
                                if (filter_graph_!=nullptr) {
                                    logstream << "Input parameters changed. Restarting filter.";
                                }
                                freeFilterGraph();
                            }
                            p2.captureInitialHWFramesCtxFromFrame(*f2);
                            if (filter_graph_==nullptr) {
                                maybeInitFilterGraph();
                            }
                            if (filter_graph_==nullptr)
                                continue;
                            if (do_shift_) {
                                eq_.in(*f2);
                            }
                            int ret2 = p2.putFrame(*f2);
                            if (ret2 < 0 && ret2 != AVERROR(EAGAIN)) {
                                throw Error("Error feeding filter graph: " + av::error2string(ret2));
                            }
                            if (ret2 >= 0) {
                                e2->pop();
                                break;
                            }
                        }
                    }
                }
                if (filter_graph_!=nullptr) {
                    if (drainFilterOutputs()) {
                        return;
                    }
                } else { // filter_graph_==nullptr
                    // filter_graph_ couldn't be created because not all input
                    // parameters are known yet.
                    //
                    // Peek at every source that still lacks in_args_ and try to
                    // capture its parameters now.  Without this, a deadlock occurs
                    // when the input queues fill up before all parameters are known:
                    // waitForInput() polls eventfds that have already been drained,
                    // no new items can be added to the full queues, so poll() blocks
                    // forever while upstream nodes are stuck in enqueue().
                    for (int i = 0; i < (int)this->source_edges_.size(); i++) {
                        if (sources_[i].isSourceReadyToInit()) continue;
                        auto &ei = this->source_edges_[i];
                        T* fi = ei->peek();
                        if (fi && (!fi->isNull()) && fi->isComplete() &&
                                fi->timeBase().getNumerator() && fi->timeBase().getDenominator()) {
                            sources_[i].checkFrame(*fi, ei);
                            sources_[i].captureInitialHWFramesCtxFromFrame(*fi);
                        }
                    }
                    maybeInitFilterGraph();
                    if (filter_graph_ == nullptr) {
                        this->waitForInput();
                    }
                }
            } else {
                edge->pop();
                logstream << "filter got null / incomplete / invalid frame (timebase " << frmin->timeBase() << ")";
                logstream << "filter finishing because got null or incomplete frame";
                forwardEofToSinks();
                this->finished_ = true;
            }
        } else if (this->stopping_) {
            forwardEofToSinks();
            this->finished_ = true;
        }
    }
public:
    virtual void initDefaults(const Parameters &params) = 0;
    static std::shared_ptr<Child> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::string graph_desc = params.at("graph").get<std::string>();
        bool shift = MediaSpecific::default_do_shift;
        if (params.count("shift")==1) {
            shift = params["shift"].get<bool>();
        }
        std::shared_ptr<Child> result = std::make_shared<Child>(graph_desc, shift);
        if (params.count("defer_preliminary_init")==1) {
            result->defer_preliminary_init_ = params["defer_preliminary_init"].get<bool>();
        }
        if (params.count("hwaccel")) {
            result->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        }
        result->initDefaults(params);
        result->createSourcesFromParameters(edges, params);
        result->createSinksFromParameters(edges, params);
        result->initPorts();
        if (result->sinks_.size()!=1) {
            // TODO (would require huge change in architecture - support of in/out pads)
            throw Error("Currently only single-output filters are supported");
        }
        if (!result->defer_preliminary_init_) {
            try {
                result->preliminaryInit();
            } catch (std::exception &e) {
                logstream << "preliminary init failed, will retry when we get first frame: " << e.what();
                result->freeFilterGraph();
            }
        }
        return result;
    }
    virtual av::Rational timeBase() {
        ensureNotNull(out_ctx_, "timeBase(): out ctx none");
        return av_buffersink_get_time_base(out_ctx_);
    }
};

#define assign(to, from) if (params.count(from)==1) default_params_.to = params[from]

class VideoFilter: public FilterNode<VideoFilter, av::VideoFrame, AVMEDIA_TYPE_VIDEO>, public IVideoFormatSource, public IFrameRateSource {
protected:
    VideoParameters default_params_;
    av::Rational default_frame_rate_{0, 1};
public:
    using FilterNode::FilterNode;
    virtual void initDefaults(const Parameters& params) {
        assign(width, "dst_width");
        assign(height, "dst_height");
        if (params.count("dst_pixel_format")==1) default_params_.pixel_format = av::PixelFormat(params["dst_pixel_format"].get<std::string>());
        if (params.count("dst_frame_rate")==1) default_frame_rate_ = parseRatio(params["dst_frame_rate"]);
    }
    virtual int width() {
        if (out_ctx_) {
            return av_buffersink_get_w(out_ctx_);
        } else if (default_params_.width>0) {
            return default_params_.width;
        } else {
            throw Error("unknown filter output width");
        }
    }
    virtual int height() {
        if (out_ctx_) {
            return av_buffersink_get_h(out_ctx_);
        } else if (default_params_.height>0) {
            return default_params_.height;
        } else {
            throw Error("unknown filter output height");
        }
    }
    virtual av::PixelFormat pixelFormat() {
        if (out_ctx_) {
            return av::PixelFormat(static_cast<AVPixelFormat>(av_buffersink_get_format(out_ctx_)));
        } else if (default_params_.pixel_format.get()!=AV_PIX_FMT_NONE) {
            return default_params_.pixel_format;
        } else {
            throw Error("unknown filter output pixel format");
        }
    }
    virtual av::PixelFormat realPixelFormat() {
        if (out_ctx_) {
            AVBufferRef* ref = av_buffersink_get_hw_frames_ctx(out_ctx_);
            if (ref && ref->data) {
                AVHWFramesContext *frmctx = (AVHWFramesContext *)(ref->data);
                logstream << "have hw frames context in filter outlink, sw_format " << av::PixelFormat(frmctx->sw_format);
                if (frmctx->sw_format != AV_PIX_FMT_NONE) {
                    return frmctx->sw_format;
                } else {
                    logstream << "falling back to pixelFormat()";
                }
            }
        }
        return pixelFormat();
    }
    virtual av::Rational frameRate() {
        if (out_ctx_) {
            return av_buffersink_get_frame_rate(out_ctx_);
        } else if (default_frame_rate_.getNumerator()>0 && default_frame_rate_.getDenominator()>0) {
            return default_frame_rate_;
        } else {
            throw Error("unknown filter output frame rate");
        }
    }
};
class AudioFilter: public FilterNode<AudioFilter, av::AudioSamples, AVMEDIA_TYPE_AUDIO>, public IAudioMetadataSource {
protected:
    AudioParameters default_params_;
public:
    using FilterNode::FilterNode;
    virtual void initDefaults(const Parameters& params) {
        if (params.count("dst_channel_layout")==1) {
            std::string layout_s = params["dst_channel_layout"].get<std::string>();
            default_params_.channel_layout = stringToChannelLayout(layout_s);
        } else if (params.count("dst_channels")==1) {
            int64_t cnt = params["dst_channels"].get<int>();
#if API_NEW_CHANNEL_LAYOUT
            default_params_.channel_layout = av::ChannelLayout(cnt).layout();
#else
            default_params_.channel_layout = av_get_channel_layout_nb_channels(cnt);
#endif
        }
        
        assign(sample_rate, "dst_sample_rate");
        if (params.count("dst_sample_format")==1) default_params_.sample_format = av::SampleFormat(params["dst_sample_format"].get<std::string>());
    }
    virtual int sampleRate() {
        if (out_ctx_) {
            return av_buffersink_get_sample_rate(out_ctx_);
        } else if (default_params_.sample_rate>0) {
            return default_params_.sample_rate;
        } else {
            throw Error("unknown filter output sample rate");
        }
    }
    virtual av::SampleFormat sampleFormat() {
        if (out_ctx_) {
            return av::SampleFormat(static_cast<AVSampleFormat>(av_buffersink_get_format(out_ctx_)));
        } else if (default_params_.sample_format.get()!=AV_SAMPLE_FMT_NONE) {
            return default_params_.sample_format;
        } else {
            throw Error("unknown filter output sample format");
        }
    }
    virtual uint64_t channelLayout() {
        if (out_ctx_) {
#if API_NEW_CHANNEL_LAYOUT
            // TODO
            AVChannelLayout chl;
            if (av_buffersink_get_ch_layout(out_ctx_, &chl) < 0) {
                logstream << "av_buffersink_get_ch_layout failed";
                return 0;
            }
            return chl.order == AV_CHANNEL_ORDER_NATIVE ? chl.u.mask : 0;
#else
            return av_buffersink_get_channel_layout(out_ctx_);
#endif
        } else if (default_params_.channel_layout>0) {
            return default_params_.channel_layout;
        } else {
            throw Error("unknown filter output channel layout");
        }
    }
};

#undef assign

DECLNODE(filter_video, VideoFilter);
DECLNODE(filter_audio, AudioFilter);
