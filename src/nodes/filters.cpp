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
    public:
        Port() {};
        ~Port() {
            if (initial_hw_frames_ctx_) {
                av_buffer_unref(&initial_hw_frames_ctx_);
                initial_hw_frames_ctx_ = nullptr;
            }
        }
        bool checkParameters(typename MediaSpecific::Parameters params, AVFrame *raw, av::Rational timebase, std::shared_ptr<Edge<T>> edge) {
            // Detect hw_frames_ctx changes by comparing the SEMANTIC content of
            // the AVHWFramesContext (format, sw_format, width, height, device),
            // not the pointer value.  NVENC and hwupload_cuda can rotate pool
            // objects (new AVBufferRef→data) while keeping the same format and
            // size.  Pointer comparison causes spurious rebuilds mid-wipe which
            // reset framesync and trigger the EXT_NULL race.
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
                        (cur->width     == prev->width) &&
                        (cur->height    == prev->height) &&
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
        }
        void initSourceFilter(const int index, AVFilterGraph *filter_graph, std::shared_ptr<HWAccelDevice> hwaccel, AVFilterInOut *dst) {
            std::string name = "in" + std::to_string(index);
            
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
    std::shared_ptr<HWAccelDevice> hwaccel_;

    // --- Post-rebuild peer synchronisation state ---
    // After a rebuild, the triggering source's first frame has a smaller PTS
    // than the peer input that was empty at rebuild time.  If we feed it to
    // framesync, framesync advances the trigger before advancing the peer →
    // peer.frame is still NULL (STATE_BOF) → EXT_NULL output (no overlay).
    //
    // Fix: skip frames from the triggering source until the peer has been fed
    // at least once; then framesync can advance the peer first.
    //
    // -1 = no pending sync; >= 0 = index of the peer we are waiting for.
    int  rebuild_pending_peer_       = -1;
    int  rebuild_pending_src_        = -1;  // source that triggered the rebuild
    int  rebuild_pending_skip_count_ = 0;
    static constexpr int kMaxRebuildSkipFrames = 8;
    
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
    void initPorts() {
        sources_.resize(this->source_edges_.size());
        sinks_.resize(this->sink_edges_.size());
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
public:
    FilterNode(const std::string &graph_desc, const bool do_shift):
        graph_desc_(graph_desc), do_shift_(do_shift) {
    }
    virtual ~FilterNode() {
        freeFilterGraph();
    }
    virtual void process() {
        T* frmin = nullptr;
        int source_index = this->findSourceWithData();
        // verify that some data is waiting for us:
        if (source_index>=0) {
            std::shared_ptr<Edge<T>> edge = this->source_edges_[source_index];
            frmin = edge->peek();
            if (frmin && (!frmin->isNull()) && frmin->isComplete() && frmin->timeBase().getNumerator() && frmin->timeBase().getDenominator()) {

                // --- Post-rebuild peer sync gate ---
                // Discard frames from the triggering source until the waiting
                // peer input has been fed to framesync at least once, so that
                // framesync can advance the peer before the trigger and avoid
                // an EXT_NULL output frame.
                if (rebuild_pending_peer_ >= 0) {
                    if (source_index == rebuild_pending_peer_) {
                        // Peer has arrived.  Clear the gate and fall through to
                        // normal processing below.
                        logstream << "post-rebuild peer sync: peer in["
                                  << rebuild_pending_peer_ << "] arrived (pts "
                                  << frmin->pts()
                                  << "); resuming normal feed after "
                                  << rebuild_pending_skip_count_ << " skipped trigger frames";
                        rebuild_pending_peer_       = -1;
                        rebuild_pending_src_        = -1;
                        rebuild_pending_skip_count_ = 0;
                        // fall through – feed this peer frame normally
                    } else {
                        // Still the triggering source.  Skip it.
                        if (rebuild_pending_skip_count_ < kMaxRebuildSkipFrames) {
                            ++rebuild_pending_skip_count_;
                            logstream << "post-rebuild peer sync: skipping in["
                                      << source_index << "] pts=" << frmin->pts()
                                      << " (skip " << rebuild_pending_skip_count_
                                      << "/" << kMaxRebuildSkipFrames
                                      << ", waiting for in[" << rebuild_pending_peer_ << "])";
                            edge->pop();
                            return;
                        } else {
                            // Safety: give up waiting; proceed normally.
                            logstream << "post-rebuild peer sync: timeout waiting for in["
                                      << rebuild_pending_peer_ << "] after "
                                      << rebuild_pending_skip_count_
                                      << " skipped frames; giving up";
                            rebuild_pending_peer_       = -1;
                            rebuild_pending_src_        = -1;
                            rebuild_pending_skip_count_ = 0;
                        }
                    }
                }

                Port &source_port = sources_[source_index];;
                if (!source_port.checkFrame(*frmin, edge)) {
                    if (filter_graph_!=nullptr) {
                        logstream << "Input parameters changed. Restarting filter.";
                    }
                    freeFilterGraph();
                }
                source_port.captureInitialHWFramesCtxFromFrame(*frmin);

                // Before rebuilding, also pre-capture hw_frames_ctx for every
                // peer input that has a frame in its edge.  Without this,
                // maybeInitFilterGraph() initialises each peer's buffersrc with
                // whatever initial_hw_frames_ctx_ is currently stored (possibly
                // null, or stale from a previous wipe).  When the peer frame is
                // later fed via the normal path, checkFrame() detects the
                // mismatch and forces an immediate second rebuild.  Worse, it
                // causes clone-injection below to fail (checkFrame(clone)
                // rejects the clone because its ctx differs from the peer Port's
                // stored one), leaving us to fall back to skip_trigger which
                // has worse behaviour at wipe start.
                if (filter_graph_ == nullptr && this->source_edges_.size() > 1) {
                    for (int j = 0; j < (int)this->source_edges_.size(); j++) {
                        if (j == source_index) continue;
                        auto ej = this->source_edges_[j];
                        T* fj = ej->peek();
                        if (fj && !fj->isNull() && fj->isComplete() &&
                            fj->timeBase().getNumerator() && fj->timeBase().getDenominator()) {
                            sources_[j].checkFrame(*fj, ej);
                            sources_[j].captureInitialHWFramesCtxFromFrame(*fj);
                        }
                    }
                }

                bool was_rebuilt = false;
                if (filter_graph_==nullptr) {
                    // Any rebuild_pending_peer_ state refers to the previous
                    // filter graph's framesync; with a fresh graph it is
                    // meaningless.  Clearing it here prevents cross-wipe state
                    // leakage (e.g. the gate being armed at the end of wipe N
                    // and still blocking trigger frames at the start of wipe
                    // N+1).
                    if (rebuild_pending_peer_ >= 0) {
                        logstream << "clearing stale rebuild_pending_peer_=" << rebuild_pending_peer_
                                  << " (src=" << rebuild_pending_src_
                                  << ", skip=" << rebuild_pending_skip_count_ << ") before rebuild";
                    }
                    rebuild_pending_peer_       = -1;
                    rebuild_pending_src_        = -1;
                    rebuild_pending_skip_count_ = 0;
                    was_rebuilt = maybeInitFilterGraph();
                }
                // After a rebuild, framesync resets to STATE_BOF for all inputs.
                // Downstream filters that follow the overlay convention
                // (EXT_NULL on slave inputs — see ff_framesync_init_dualinput)
                // do NOT wait for slaves to leave STATE_BOF before emitting
                // output: as soon as the master is in STATE_RUN, framesync
                // blends with a NULL slave (framesync.c ~line 230-232).  After
                // a rebuild this manifests as one camera-only frame (missing
                // overlay) whenever the trigger reaches framesync before the
                // peer does.
                //
                // Primary defence: the semantic hw_frames_ctx check prevents
                // pool-rotation rebuilds mid-wipe, so this post-rebuild path
                // is normally entered only once (at wipe-start).
                //
                // Strategy, by peer state at rebuild time:
                //   earlier peer (j < source_index):
                //     drop stale frames so the first peer frame framesync sees
                //     is at/after the trigger PTS.
                //   later peer (j > source_index):
                //     AHEAD or EMPTY: skip the trigger and arm
                //       rebuild_pending_peer_ so that subsequent trigger frames
                //       are also dropped until the peer has been fed.  After
                //       that, framesync advances the peer first → no EXT_NULL.
                //     BEHIND trigger: feed peer frames to buffersrc[j] until
                //       it's close to trigger, then fall through to normal
                //       feed.

                // Set to true when a peer is already ahead: skip feeding the trigger.
                bool skip_trigger = false;
                // Set to the index of a peer that was empty at rebuild time, so we
                // can arm rebuild_pending_peer_ if we end up skipping the trigger.
                int  skip_trigger_pending_peer = -1;

                if (was_rebuilt && this->source_edges_.size() > 1) {
                    static const av::Rational kUsTimebase{1, 1000000};
                    static constexpr int64_t kOneFrame30fps_us = 33334; // ≈ 1/30 s in µs
                    int64_t trigger_us = frmin->pts().timestamp(kUsTimebase);

                    logstream << "Post-rebuild: source_index=" << source_index
                              << " trigger_pts=" << frmin->pts()
                              << " (us=" << trigger_us << ")";

                    for (int j = 0; j < (int)this->source_edges_.size(); j++) {
                        if (j == source_index) continue;
                        if (filter_graph_ == nullptr) break;
                        std::shared_ptr<Edge<T>> ej = this->source_edges_[j];
                        Port &pj = sources_[j];
                        T* fj = ej->peek();

                        // Log peer state regardless of branch taken.
                        if (fj && !fj->isNull() && fj->isComplete() &&
                            fj->timeBase().getNumerator() && fj->timeBase().getDenominator()) {
                            logstream << "  peer in[" << j << "] pts=" << fj->pts()
                                      << " (us=" << fj->pts().timestamp(kUsTimebase) << ")"
                                      << (fj->pts().timestamp(kUsTimebase) > trigger_us ? " AHEAD" :
                                         (fj->pts().timestamp(kUsTimebase) == trigger_us ? " EQUAL" : " BEHIND"));
                        } else {
                            logstream << "  peer in[" << j << "] EMPTY";
                        }

                        if (j < source_index) {
                            // j is an "earlier" input: drop stale frames so the
                            // first frame framesync sees is contemporaneous with
                            // the trigger (prevents premature in[j] advancement
                            // causing EXT_NULL for the trigger input).
                            int n_dropped = 0;
                            while (true) {
                                fj = ej->peek();
                                if (!fj || fj->isNull() || !fj->isComplete() ||
                                    !fj->timeBase().getNumerator() || !fj->timeBase().getDenominator())
                                    break;
                                if (fj->pts().timestamp(kUsTimebase) >= trigger_us) break;
                                ej->pop();
                                n_dropped++;
                            }
                            if (n_dropped > 0) {
                                logstream << "  dropped " << n_dropped
                                          << " stale in[" << j << "] frames";
                            }
                        } else {
                            // j is a "later" input (e.g. wipe animation in[1]).
                            if (fj && !fj->isNull() && fj->isComplete() &&
                                fj->timeBase().getNumerator() && fj->timeBase().getDenominator() &&
                                fj->pts().timestamp(kUsTimebase) > trigger_us) {
                                // Peer is AHEAD of the trigger.  Skip the trigger
                                // and arm the pending-peer gate so subsequent
                                // trigger frames are also dropped until the peer
                                // has been fed once.  At that point framesync can
                                // advance the peer first, leaving the master in
                                // STATE_BOF → no EXT_NULL frame.
                                //
                                // With force_fps normalisation on both inputs this
                                // case should be rare at wipe-start (both grids
                                // are 30fps/tb 1/30), and eliminated mid-wipe once
                                // the semantic hw_frames_ctx check prevents pool-
                                // rotation rebuilds.
                                skip_trigger = true;
                                skip_trigger_pending_peer = j;
                                logstream << "  -> peer in[" << j
                                          << "] AHEAD (us=" << fj->pts().timestamp(kUsTimebase)
                                          << " vs trigger=" << trigger_us
                                          << "); skip_trigger + pending peer";
                            } else if (!fj || fj->isNull() || !fj->isComplete() ||
                                       !fj->timeBase().getNumerator() || !fj->timeBase().getDenominator()) {
                                // Peer is EMPTY: feeding the trigger now would
                                // advance it before the peer → EXT_NULL once the
                                // peer arrives.  Skip trigger AND arm the pending
                                // peer gate so subsequent trigger frames are also
                                // dropped until the peer is processed.
                                skip_trigger = true;
                                skip_trigger_pending_peer = j;
                                logstream << "  -> skip_trigger + pending peer (peer empty)";
                            } else {
                                // Peer LAGS the trigger: prime it so framesync has
                                // it close to the trigger PTS.
                                int n_fed = 0;
                                while (filter_graph_ != nullptr) {
                                    fj = ej->peek();
                                    if (!fj || fj->isNull() || !fj->isComplete() ||
                                        !fj->timeBase().getNumerator() || !fj->timeBase().getDenominator())
                                        break;
                                    if (!pj.checkFrame(*fj, ej)) {
                                        logstream << "  input parameters changed during post-rebuild priming; restarting filter";
                                        freeFilterGraph();
                                        break;
                                    }
                                    pj.captureInitialHWFramesCtxFromFrame(*fj);
                                    if (filter_graph_ == nullptr) break;
                                    if (do_shift_) eq_.in(*fj);
                                    int rj = pj.putFrame(*fj);
                                    if (rj < 0) break;
                                    ej->pop();
                                    n_fed++;
                                    int64_t fj_us = fj->pts().timestamp(kUsTimebase);
                                    if (fj_us + kOneFrame30fps_us >= trigger_us) break;
                                }
                                logstream << "  -> primed " << n_fed << " in[" << j << "] frames";
                            }
                        }
                    }
                }
                if (filter_graph_!=nullptr) {
                    if (skip_trigger) {
                        // Drop the trigger frame without feeding it to framesync.
                        // frmin must not be used after this pop.
                        edge->pop();
                        if (skip_trigger_pending_peer >= 0) {
                            // Arm the gate to also skip subsequent trigger frames
                            // until the peer has been fed.
                            rebuild_pending_peer_       = skip_trigger_pending_peer;
                            rebuild_pending_src_        = source_index;
                            rebuild_pending_skip_count_ = 0;
                            logstream << "Post-rebuild peer sync armed: waiting for in["
                                      << rebuild_pending_peer_ << "]";
                        }
                    } else {
                    if (do_shift_) {
                        eq_.in(*frmin);
                    }
                    int ret = source_port.putFrame(*frmin);
                    if (ret < 0 && ret != AVERROR(EAGAIN)) {
                        throw Error("Error feeding filter graph: " + av::error2string(ret));
                    } else if (ret >= 0) {
                        edge->pop(); // no need to retry, pop this frame
                    } else {
                        // AVERROR(EAGAIN): this buffersrc is not accepting yet.  findSourceWithData()
                        // always prefers the smallest PTS among inputs; when timebases are not comparable
                        // (e.g. program 1/30 vs wallclock ms on a wipe), only that pad is ever fed and
                        // multi-input filters (overlay, framesync) never advance.
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
                    } // end else (skip_trigger)
                }
                if (filter_graph_!=nullptr) {
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
                                    return;
                                }
                            } else {
                                logstream << "WARNING: Invalid frame received from filter graph";
                            }
                        }
                    }
                    if (finished_sinks==sinks_.size()) {
                        this->finished_ = true;
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
                this->finished_ = true;
            }
        } else if (this->stopping_) {
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
        try {
            result->preliminaryInit();
        } catch (std::exception &e) {
            logstream << "preliminary init failed, will retry when we get first frame: " << e.what();
            result->freeFilterGraph();
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
