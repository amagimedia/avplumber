#include "node_common.hpp"
#include <algorithm>
#include "../avutils.hpp"
#include <avcpp/codeccontext.h>
#include "../hwaccel.hpp"

template<typename Child, typename DecoderContext, typename OutputFrame> class Decoder:
    public NodeSISO<av::Packet, OutputFrame>, public ReportsFinishByFlag, public IFlushable, public IDecoder, public ITimeBaseSource {
protected:
    av::Codec codec_;
    DecoderContext dec_;
    std::recursive_mutex mutex_;
    std::shared_ptr<IStreamsInput> input_hold_;
    size_t dec_errors_ = 0;
    av::Timestamp last_pts_ = NOTS;
    av::PixelFormat pixel_format_ = AV_PIX_FMT_NONE;
    bool pixel_format_optional_ = false;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    //AVBufferRef *out_frames_ref_ = nullptr;
    /* input_hold_ is a workaround to prevent StreamInput from being destroyed
     * when the shared_ptr is set to null in NodeWrapper
     * There are other ways to handle this:
     * X- use waitSinksEmpty() as it is and redesign the EdgeSource to have separate
     *    peek() and pop() functions (just like ReaderWriterQueue has)
     *    - IT DOESN'T WORK!
     *  - redesign NodeWrapper to not destroy wrapped Node until the group is destroyed
     */
    static av::Codec codecFromName(const std::string codec_name) {
        if (codec_name.length()>0) {
            av::Codec r = av::findDecodingCodec(codec_name);
            if (r.isNull()) {
                throw Error("unknown codec: " + codec_name);
            }
            return r;
        } else {
            return {};
        }
    }
public:
    template<typename ...Ts> Decoder(std::unique_ptr<Source<av::Packet>> &&source, std::unique_ptr<Sink<OutputFrame>> &&sink, av::Stream &stream, const std::string codec_name, av::Dictionary options, std::string pixel_format, std::shared_ptr<HWAccelDevice> hwaccel):
        NodeSISO<av::Packet, OutputFrame>(std::move(source), std::move(sink)),
        codec_(codecFromName(codec_name)),
        dec_(stream, codec_),
        hwaccel_(hwaccel)
    {
        if (!pixel_format.empty()) {
            pixel_format_optional_ = pixel_format[0]=='?';
            pixel_format_ = av::PixelFormat(pixel_format_optional_ ? pixel_format.substr(1) : pixel_format);
        }

        input_hold_ = this->template findNodeUp<IStreamsInput>();
        dec_.setRefCountedFrames(true);

        bool good_tb = dec_.timeBase().getNumerator() && dec_.timeBase().getDenominator();
        if (!good_tb) {
            logstream << "Decoder detected invalid timebase";
            #if LIBAVCODEC_VERSION_MAJOR < 59 // FFmpeg 5.0
            FF_DISABLE_DEPRECATION_WARNINGS
            if (stream.raw()->codec != nullptr) {
                AVRational &tb = stream.raw()->codec->time_base;
                logstream << "stream->codec not null, tb " << tb.num << "/" << tb.den;
                if (tb.num && tb.den) {
                    dec_.setTimeBase(av::Rational(tb));
                    logstream << "Set timebase from stream->codec";
                    good_tb = true;
                }
            }
            FF_ENABLE_DEPRECATION_WARNINGS
            #else
            logstream << "FIXME: Unable to fix timebase, ffmpeg too new";
            #endif
        }
        if (!good_tb) {
            if (stream.timeBase().getNumerator() && stream.timeBase().getDenominator()) {
                dec_.setTimeBase(stream.timeBase());
                logstream << "Set timebase from stream: " << stream.timeBase();
                good_tb = true;
            }
        }
        
        if (hwaccel) {
            dec_.raw()->hw_device_ctx = hwaccel->refDeviceContext();
        }
        if (pixel_format_ != AV_PIX_FMT_NONE) {
            dec_.raw()->opaque = this;
            dec_.raw()->get_format = [](AVCodecContext *cc, const enum AVPixelFormat *pix_fmts) -> AVPixelFormat {
                Decoder &self = *(Decoder*)(cc->opaque);
                const enum AVPixelFormat *p;
                for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                    if (*p == self.pixel_format_) {
                        #if 0 // not really needed
                        if (self.hwaccel_) {
                            int r = avcodec_get_hw_frames_parameters(cc, self.hwaccel_->refDeviceContext(), *p, &cc->hw_frames_ctx);
                            if (r == AVERROR(ENOENT)) {
                                logstream << "avcodec_get_hw_frames_parameters not supported by this decoder, returning pixel format anyway";
                            } else {
                                if (r != 0) {
                                    logstream << "Failed to setup hardware decoder: avcodec_get_hw_frames_parameters failed: " << av::error2string(r);
                                    return AV_PIX_FMT_NONE;
                                }
                                r = av_hwframe_ctx_init(cc->hw_frames_ctx);
                                if (r != 0) {
                                    logstream << "Failed to setup hardware decoder: av_hwframe_ctx_init failed: " << av::error2string(r);
                                    return AV_PIX_FMT_NONE;
                                }
                            }
                        }
                        #endif
                        return *p;
                    }
                }
                logstream << "Decoder doesn't support specified pixel_format " << self.pixel_format_;
                if (self.pixel_format_optional_) {
                    logstream << "Using best decoder's pixel format: " << av::PixelFormat(pix_fmts[0]);
                    return pix_fmts[0];
                }
                return AV_PIX_FMT_NONE;
            };
        }

        dec_.open(options, codec_);
        logstream << "Opened decoder " << dec_.codec().name() << std::endl;
    };
    virtual std::string codecName() const {
        if (!dec_.codec().isNull()) {
            return dec_.codec().name();
        } else {
            return "???";
        }
    }
    virtual std::string codecMediaTypeString() const {
        if (!dec_.codec().isNull()) {
            AVMediaType mt = dec_.codec().type();
            return mediaTypeToString(mt);
        }
        return "?";
    }
    virtual std::string fieldOrderString() const {
        return fieldOrderToString(dec_.raw()->field_order);
    }
    virtual av::Rational timeBase() final {
        //return av::Rational(this->dec_.stream().raw()->codec->time_base);
        return av::Rational(this->dec_.raw()->time_base);
    }
    virtual void flush() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        OutputFrame frm;
        do {
            try {
                frm = dec_.decode(av::Packet());
                //this->sink_->put(frm); // don't do anything with the frame! otherwise, blinking happens when decoder is flushed after generating "no signal" card
            } catch (std::exception &e) {
                logstream << "Warning: Exception " << e.what() << " when flushing decoder." << std::endl;
                break; // flush error is not considered error
            }
        } while (frm);
        //dec_.close();
        //input_hold_ = nullptr;
        this->finished_ = true;
    };
    virtual void process() {
        // wait for packet
        //av::Packet pkt = this->source_->get();
        av::Packet *pktp = this->source_->peek();
        if (pktp==nullptr) {
            flush();
            return;
        }
        av::Packet &pkt = *pktp;
        // lock us, to prevent race condition with flushing!
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            if ( (!pkt.isNull()) && pkt.isComplete() ) {
                // not a flush packet
                try {
                    OutputFrame frm = dec_.decode(pkt);
                    if (frm) {
                        if ( last_pts_.isValid() && (last_pts_ > frm.pts()) ) {
                            logstream << "Warning: Got out of order frame from decoder: " << last_pts_ << " -> " << frm.pts();
                        }
                        last_pts_ = frm.pts();
                        this->sink_->put(frm);
                    }
                    dec_errors_ = 0;
                } catch (std::exception &e) {
                    dec_errors_++;
                    if (dec_errors_>200) {
                        throw;
                    }
                    logstream << "Decode error: " << e.what();
                }
                //if (!frm) this->finished_ = true;

                this->source_->pop();
            } else {
                // this is flush packet
                // so flush decoder
                flush();
                this->source_->pop();
            }
        }
    }
    virtual ~Decoder() {
        try {
            dec_.close();
        } catch (std::exception &e) {
        }
    }
    static std::shared_ptr<Child> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> src_edge = edges.find<av::Packet>(params["src"]);
        std::shared_ptr<Edge<OutputFrame>> dst_edge = edges.find<OutputFrame>(params["dst"]);
        std::shared_ptr<IStreamsInput> input = src_edge->findNodeUp<IStreamsInput>();
        std::shared_ptr<InputStreamMetadata> md = nullptr;
        if (input != nullptr) md = src_edge->findMetadataUp<InputStreamMetadata>();
        if (md == nullptr) {
            throw Error("Couldn't initialize decoder: no input stream.");
        }
        std::string codec_name;
        AVCodecParameters &cpar = *md->source_stream.raw()->codecpar;
        std::string in_codec = avcodec_get_name(cpar.codec_id);
        if (params.count("codec")==1) {
            codec_name = params["codec"];
        } else if (params.count("codec_map")==1) {
            const Parameters &map = params["codec_map"];
            if (map.count(in_codec)==1) {
                codec_name = map[in_codec];
                logstream << "Detected codec " << in_codec << ", using implementation: " << codec_name;
            } else {
                logstream << "Detected codec " << in_codec << " not in codec_map, using libavcodec default";
            }
        }
        std::string pixel_format;
        if (params.count("pixel_format")) {
            pixel_format = params["pixel_format"];
        }
        std::shared_ptr<HWAccelDevice> hwaccel;
        if (params.count("hwaccel")) {
            bool use_hw = true;
            if (params.count("hwaccel_only_for_codecs")) {
                auto allowed_codecs = jsonToStringList(params["hwaccel_only_for_codecs"]);
                use_hw = std::find(allowed_codecs.begin(), allowed_codecs.end(), in_codec) != allowed_codecs.end();
            }
            if (use_hw) {
                hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
            }
        }
        av::Dictionary options;
        if (params.count("options")) {
            options = parametersToDict(params["options"]);
        }
        std::shared_ptr<Child> r = std::make_shared<Child>(src_edge->makeSource(), dst_edge->makeSink(), md->source_stream, codec_name, options, pixel_format, hwaccel);
        return r;
    }
};


class VideoDecoder: public Decoder<VideoDecoder, av::VideoDecoderContext, av::VideoFrame>, public IVideoFormatSource, public IFrameRateSource {
public:
    using Decoder::Decoder;
    virtual int width() {
        return this->dec_.width();
    }
    virtual int height() {
        return this->dec_.height();
    }
    virtual av::PixelFormat pixelFormat() {
        av::PixelFormat pf = this->dec_.pixelFormat();
        if (pf==AV_PIX_FMT_NONE) {
            pf = realPixelFormat();
            logstream << "decoder pixel format none, using stream pixel format " << pf;
        }
        return pf;
    }
    virtual av::PixelFormat realPixelFormat() {
        AVCodecContext* cc = this->dec_.raw();
        if (cc->sw_pix_fmt != AV_PIX_FMT_NONE) {
            return cc->sw_pix_fmt;
        } else {
            logstream << "realPixelFormat(): sw_pix_fmt none, falling back to stream pixel format";
        }
        #if 0
        if (cc && cc->hw_frames_ctx && cc->hw_frames_ctx->data) {
            AVHWFramesContext *frmctx = (AVHWFramesContext *)(cc->hw_frames_ctx->data);
            logstream << "have hw frames context in decoder, sw_format " << av::PixelFormat(frmctx->sw_format);
            if (frmctx->sw_format != AV_PIX_FMT_NONE) {
                return frmctx->sw_format;
            } else {
                logstream << "falling back to stream pixel format";
            }
        }
        #endif
        return av::PixelFormat(static_cast<AVPixelFormat>(this->dec_.stream().raw()->codecpar->format));
    }
    virtual av::Rational frameRate() {
        return this->dec_.stream().frameRate();
    }
};
class AudioDecoder: public Decoder<AudioDecoder, av::AudioDecoderContext, av::AudioSamples>, public IAudioMetadataSource {
public:
    using Decoder::Decoder;
    virtual int sampleRate() {
        return this->dec_.sampleRate();
    }
    virtual av::SampleFormat sampleFormat() {
        return this->dec_.sampleFormat();
    }
    virtual uint64_t channelLayout() {
        return this->dec_.channelLayout();
    }
};

DECLNODE(dec_video, VideoDecoder);
DECLNODE(dec_audio, AudioDecoder);
