#include "node_common.hpp"
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include "../ts_equalizer.hpp"
#include "../video_parameters.hpp"
#include "../audio_parameters.hpp"
#include "../hwaccel.hpp"

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
    bool checkFrame(av::VideoFrame &frm) {
        return par_ == VideoParameters(frm);
    }
    bool checkParameters(VideoParameters &p) {
        return par_ == p;
    }
    static Parameters parametersFromNodeInterface(NodeInterface &ni) {
        return ni.videoParameters();
    }
    void initHWAccel(AVHWFramesContext &frmctx) {
        frmctx.sw_format = par_.real_pixel_format;
        frmctx.width = par_.width;
        frmctx.height = par_.height;
        frmctx.format = AV_PIX_FMT_CUDA; // TODO deduce from somewhere
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
    bool checkFrame(av::AudioSamples &frm) {
        return par_ == AudioParameters(frm);
    }
    bool checkParameters(AudioParameters &p) {
        return par_ == p;
    }
    static Parameters parametersFromNodeInterface(NodeInterface &ni) {
        return ni.audioParameters();
    }
    void initHWAccel(AVHWFramesContext&) {
        throw Error("hwaccel specified for audio filter");
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
    public:
        Port() {};
        bool checkParameters(typename MediaSpecific::Parameters params, av::Rational timebase, std::shared_ptr<Edge<T>> edge) {
            bool result = (timebase==prev_tb_) && ms_.checkParameters(params);
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
            return checkParameters(typename MediaSpecific::Parameters(frm), frm.timeBase(), edge);
        }
        bool isSourceReadyToInit() {
            return !in_args_.empty();
        }
        AVFilterContext* context() {
            return ctx_;
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
            
            if (hwaccel) {
                AVFilterLink* link = getSourceLink();
                soft_assert(link != nullptr, "source link null");
                link->hw_frames_ctx = av_hwframe_ctx_alloc(hwaccel->deviceContext());
                AVHWFramesContext *frmctx = (AVHWFramesContext *)(link->hw_frames_ctx->data);
                ms_.initHWAccel(*frmctx);
                av_hwframe_ctx_init(link->hw_frames_ctx);
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
    const AVFilterLink *outlink_ = nullptr;
    TSEqualizer eq_;
    std::string graph_desc_;
    bool do_shift_ = true;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    
    void freeFilterGraph() {
        if (filter_graph_ == nullptr) return;
        avfilter_graph_free(&filter_graph_);
        filter_graph_ = nullptr;
        outlink_ = nullptr;
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
        i = 0;
        forEachInOut(outputs, [this, &i](AVFilterInOut* out) {
            if (i >= sinks_.size()) {
                throw Error("Too many outputs in filtergraph");
            }
            sinks_[i].initSinkFilter(i, filter_graph_, out);
            i++;
        });
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
            outlink_ = sinks_[0].getSinkLink();
            logstream << outlink_->type;
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
            sources_[i].checkParameters(params, tbsrc->timeBase(), edge);
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
                Port &source_port = sources_[source_index];
                if (!source_port.checkFrame(*frmin, edge)) {
                    if (filter_graph_!=nullptr) {
                        logstream << "Input parameters changed. Restarting filter.";
                    }
                    freeFilterGraph();
                }
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
                    }
                    int finished_sinks = 0;
                    for (int sink_index=0; sink_index<sinks_.size(); sink_index++) {
                        Port& sink_port = sinks_[sink_index];
                        while(true) {
                            T frmout;
                            
                            ret = sink_port.getFrame(frmout);
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
                    // filter_graph_ couldn't be created
                    // because not all input parameters are known
                    // so wait for some more packetz
                    this->waitForInput();
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
        }
        return result;
    }
    virtual av::Rational timeBase() {
        ensureNotNull(outlink_, "timeBase(): outlink none");
        return outlink_->time_base;
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
        if (outlink_) {
            return outlink_->w;
        } else if (default_params_.width>0) {
            return default_params_.width;
        } else {
            throw Error("unknown filter output width");
        }
    }
    virtual int height() {
        if (outlink_) {
            return outlink_->h;
        } else if (default_params_.height>0) {
            return default_params_.height;
        } else {
            throw Error("unknown filter output height");
        }
    }
    virtual av::PixelFormat pixelFormat() {
        if (outlink_) {
            return av::PixelFormat(static_cast<AVPixelFormat>(outlink_->format));
        } else if (default_params_.pixel_format.get()!=AV_PIX_FMT_NONE) {
            return default_params_.pixel_format;
        } else {
            throw Error("unknown filter output pixel format");
        }
    }
    virtual av::PixelFormat realPixelFormat() {
        if (outlink_ && outlink_->hw_frames_ctx && outlink_->hw_frames_ctx->data) {
            AVHWFramesContext *frmctx = (AVHWFramesContext *)(outlink_->hw_frames_ctx->data);
            logstream << "have hw frames context in filter outlink, sw_format " << av::PixelFormat(frmctx->sw_format);
            if (frmctx->sw_format != AV_PIX_FMT_NONE) {
                return frmctx->sw_format;
            } else {
                logstream << "falling back to pixelFormat()";
            }
        }
        return pixelFormat();
    }
    virtual av::Rational frameRate() {
        if (outlink_) {
            return outlink_->frame_rate;
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
            default_params_.channel_layout = av_get_channel_layout(layout_s.c_str());
        } else if (params.count("dst_channels")==1) {
            int64_t cnt = params["dst_channels"].get<int>();
            default_params_.channel_layout = av_get_channel_layout_nb_channels(cnt);
        }
        
        assign(sample_rate, "dst_sample_rate");
        if (params.count("dst_sample_format")==1) default_params_.sample_format = av::SampleFormat(params["dst_sample_format"].get<std::string>());
    }
    virtual int sampleRate() {
        if (outlink_) {
            return outlink_->sample_rate;
        } else if (default_params_.sample_rate>0) {
            return default_params_.sample_rate>0;
        } else {
            throw Error("unknown filter output sample rate");
        }
    }
    virtual av::SampleFormat sampleFormat() {
        if (outlink_) {
            return av::SampleFormat(static_cast<AVSampleFormat>(outlink_->format));
        } else if (default_params_.sample_format.get()!=AV_SAMPLE_FMT_NONE) {
            return default_params_.sample_format;
        } else {
            throw Error("unknown filter output sample format");
        }
    }
    virtual uint64_t channelLayout() {
        if (outlink_) {
            return outlink_->channel_layout;
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
