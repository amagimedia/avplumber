#include "node_common.hpp"
#include <unordered_set>
#include <avcpp/codeccontext.h>
#include "../hwaccel.hpp"

template<typename Child, typename EncoderContext, typename InputFrame> class Encoder: public NodeSISO<InputFrame, av::Packet>, public IEncoder, public ReportsFinishByFlag, public IFlushable {
protected:
    AVCodecParameters* codecpar_ = nullptr;
    std::unordered_set<AVCodecParameters*> codecpars_;
    av::Codec codec_;
    EncoderContext enc_;
    std::recursive_mutex mutex_;
    av::Dictionary options_;
    int enc_flags_ = 0;
    bool timestamps_passthrough_ = false;
    av::Timestamp prev_ts_ = NOTS;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    virtual void initEncoderPreOpen(av::Stream&) {
    }
    virtual void initEncoderPostOpen() {
    }
    av::Rational getTimeBase() {
        std::shared_ptr<ITimeBaseSource> tbmd = this->template findNodeUp<ITimeBaseSource>();
        if (!tbmd) {
            throw Error("Unknown timebase");
        }
        av::Rational timebase = tbmd->timeBase();
        if (!(timebase.getNumerator()>0 && timebase.getDenominator()>0)) {
            logstream << "WARNING: Encoder timebase " << timebase;
        }
        logstream << "getTimeBase " << timebase;
        return timebase;
    }
public:
    Encoder(std::unique_ptr<Source<InputFrame>> &&source, std::unique_ptr<Sink<av::Packet>> &&sink, av::Codec codec, std::shared_ptr<HWAccelDevice> hwaccel):
        NodeSISO<InputFrame, av::Packet>(std::move(source), std::move(sink)),
        codec_(codec),
        hwaccel_(hwaccel) {
        //enc_.open(av::Codec());
    };
    av::Dictionary& options() {
        return options_;
    }
    virtual av::Codec& encodingCodec() {
        return codec_;
    }
    virtual AVCodecParameters* codecParameters() {
        return codecpar_;
    }
    virtual void openEncoder(av::Stream stream = av::Stream()) {
        if (!enc_.isOpened()) {
            logstream << "opening encoder with stream " << (stream.isNull() ? "null" : "not null");
            
            if (!enc_.isValid()) {
                av::Stream fake_stream = av::Stream();
                initEncoderPreOpen(fake_stream /*stream*/);
            }
            enc_.setFlags(enc_flags_);
            
            if (hwaccel_) {
                enc_.raw()->hw_device_ctx = hwaccel_->refDeviceContext();
                enc_.raw()->hw_frames_ctx = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
                AVHWFramesContext *frmctx = (AVHWFramesContext *)(enc_.raw()->hw_frames_ctx->data);
                frmctx->sw_format = enc_.raw()->pix_fmt;
                frmctx->width = enc_.raw()->width;
                frmctx->height = enc_.raw()->height;
                frmctx->format = AV_PIX_FMT_CUDA; // TODO deduce from somewhere
                enc_.raw()->pix_fmt = frmctx->format;
                int r = av_hwframe_ctx_init(enc_.raw()->hw_frames_ctx);
                if (r != 0) {
                    throw Error("Failed to setup hardware encoder: av_hwframe_ctx_init failed: " + av::error2string(r));
                }
            }
            
            enc_.setTimeBase(getTimeBase());
            
            // make copy of options because otherwise enc_.open will remove all consumed ones
            av::Dictionary options(options_);
            enc_.open(options, codec_);
            if (options.count()>0) {
                logstream << "Unknown options: " << options.toString('=', ',');
            }
            
            logstream << "encoder bitrate after open: " << enc_.bitRate();
            initEncoderPostOpen();
        }
        for (AVCodecParameters* cpar: codecpars_) {
            // FIXME does it really make sense? maybe better do it right after opening and on stream adding
            avcodec_parameters_from_context(cpar, enc_.raw());
        }
    }
    virtual void setOutput(av::Stream &stream, av::FormatContext &octx) {
        if ( (!codec_.isNull()) && (!octx.outputFormat().codecSupported(codec_)) ) {
            throw Error(std::string("Codec ") + codec_.name() + " not supported by container " + octx.outputFormat().name());
        }
        stream.setTimeBase(getTimeBase());
        
        std::shared_ptr<IFrameRateSource> frs = this->template findNodeUp<IFrameRateSource>();
        if (frs) {
            av::Rational fr = frs->frameRate();
            stream.setAverageFrameRate(fr);
            stream.setFrameRate(fr);
        }
        
        codecpars_.insert(stream.raw()->codecpar);
        if (codecpar_ == nullptr) {
            codecpar_ = stream.raw()->codecpar;
        }
        enc_flags_ = octx.outputFormat().isFlags(AVFMT_GLOBALHEADER) ? AV_CODEC_FLAG_GLOBAL_HEADER : 0;

        if (!enc_.isValid()) {
            av::Stream fake_stream = av::Stream();
            initEncoderPreOpen(fake_stream /*stream*/);
        }
    }
    virtual void start() {
        openEncoder();
    }
    virtual void flush() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        av::Packet pkt;
        do {
            try {
                pkt = enc_.encode();
                // TODO honor timestamps_passthrough_
                logstream << "enc flush out: PTS = " << pkt.pts();
                if (!(pkt.timeBase().getDenominator() && pkt.timeBase().getNumerator())) {
                    logstream << "enc flush out: invalid timebase, not outputting! " << pkt.timeBase();
                } else {
                    this->sink_->put(pkt);
                }
            } catch (std::exception &e) {
                logstream << "Warning: Exception " << e.what() << " when flushing encoder." << std::endl;
                break;
            }
        } while (pkt);
        this->finished_ = true;
    }
    virtual void process() {
        // wait for frame
        InputFrame frame = this->source_->get();
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            if (frame.isComplete()) {
                // not a flush frame
                //logstream << "enc in: PTS = " << frame.pts() << std::endl;
                if (prev_ts_.isValid() && frame.pts().isValid() && addTS(frame.pts(), negateTS(prev_ts_)).timestamp() < 0) {
                    logstream << "input PTS went backwards " << prev_ts_ << " -> " << frame.pts() << ", discarding frame";
                    return;
                }
                av::Packet pkt = enc_.encode(frame);
                if (pkt) {
                    if (timestamps_passthrough_) {
                        // HACK to support timestamps passthrough in pcm_* audio "encoders"
                        pkt.setPts(rescaleTS(frame.pts(), pkt.pts().timebase()));
                        pkt.setDts(rescaleTS(frame.pts(), pkt.dts().timebase()));
                    }
                    //logstream << "enc out: PTS = " << pkt.pts() << std::endl;
                    this->sink_->put(pkt);
                } else if (timestamps_passthrough_) {
                    logstream << "WARNING: encoder does buffer but we overwrite timestamps, this may cause desync!";
                }
                if (frame.pts().isValid()) {
                    prev_ts_ = frame.pts();
                }
            } else {
                // flush encoder
                flush();
            }
        }
    }
    static std::shared_ptr<Child> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::string codecname = params["codec"];
        std::shared_ptr<Edge<InputFrame>> src_edge = edges.find<InputFrame>(params["src"]);
        std::shared_ptr<Edge<av::Packet>> dst_edge = edges.find<av::Packet>(params["dst"]);
        std::shared_ptr<HWAccelDevice> hwaccel;
        if (params.count("hwaccel")) {
            hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        }
        auto r = std::make_shared<Child>(src_edge->makeSource(), dst_edge->makeSink(), av::findEncodingCodec(codecname), hwaccel);
        if (params.count("options") > 0) {
            r->options() = parametersToDict(params["options"]);
        }
        if (params.count("timestamps_passthrough") > 0) {
            r->timestamps_passthrough_ = params["timestamps_passthrough"];
        }
        return r;
    }
};


class VideoEncoder: public Encoder<VideoEncoder, av::VideoEncoderContext, av::VideoFrame> {
public:
    using Encoder::Encoder;
protected:
    virtual void initEncoderPreOpen(av::Stream &stream) {
        if (stream.isNull()) {
            this->enc_ = av::VideoEncoderContext(codec_);
        } else {
            this->enc_ = av::VideoEncoderContext(stream, codec_);
        }
        std::shared_ptr<IVideoFormatSource> metadata = this->findNodeUp<IVideoFormatSource>();
        if (metadata==nullptr) {
            throw Error("Couldn't initialize video encoder: no metadata source");
        }
        this->enc_.setWidth(metadata->width());
        this->enc_.setHeight(metadata->height());
#if USE_CODECPAR
        if (codecpar_) {
            codecpar_->width = metadata->width();
            codecpar_->height = metadata->height();
        }
#endif
        if (metadata->realPixelFormat() != AV_PIX_FMT_NONE) {
            this->enc_.setPixelFormat(metadata->realPixelFormat());
        }
    }
};
class AudioEncoder: public Encoder<AudioEncoder, av::AudioEncoderContext, av::AudioSamples> {
public:
    using Encoder::Encoder;
protected:
    virtual void initEncoderPreOpen(av::Stream &stream) {
        if (stream.isNull()) {
            this->enc_ = av::AudioEncoderContext(codec_);
        } else {
            this->enc_ = av::AudioEncoderContext(stream, codec_);
        }
        std::shared_ptr<IAudioMetadataSource> metadata = this->findNodeUp<IAudioMetadataSource>();
        if (metadata==nullptr) {
            throw Error("Couldn't initialize audio encoder: no metadata source");
        }
        this->enc_.setSampleRate(metadata->sampleRate());
        this->enc_.setSampleFormat(metadata->sampleFormat());
        this->enc_.setChannelLayout(metadata->channelLayout());
#if USE_CODECPAR
        if (codecpar_) {
            codecpar_->sample_rate = metadata->sampleRate();
            codecpar_->format = metadata->sampleFormat().get();
            codecpar_->channel_layout = metadata->channelLayout();
        }
#endif
    }
    virtual void initEncoderPostOpen() {
        std::shared_ptr<INeedsOutputFrameSize> ofs = this->findNodeUp<INeedsOutputFrameSize>();
        if (ofs!=nullptr) {
            ofs->setOutputFrameSize(this->enc_.frameSize());
        }
    }
};

DECLNODE(enc_video, VideoEncoder);
DECLNODE(enc_audio, AudioEncoder);
