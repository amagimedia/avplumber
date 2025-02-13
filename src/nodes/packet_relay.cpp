#include "node_common.hpp"

#include <avcpp/codeccontext.h>

class PacketRelay: public TransparentNode<av::Packet>,
                   public IDecoder /* not really */, public IEncoder /* not really */,
                   public ITimeBaseSource, public IVideoFormatSource, public IFrameRateSource {
protected:
    av::Stream source_stream_;
    av::GenericCodecContext in_decoder_;
    av::VideoDecoderContext vdec_;
    av::Codec codec_;
public:
    virtual void setOutput(av::Stream &stream, av::FormatContext &octx) override {
        if (codec_.isNull()) {
            throw Error("codec is null when trying to init packet relay");
        }
        if (!octx.outputFormat().codecSupported(codec_)) {
            throw Error(std::string("Codec ") + codec_.name() + " not supported by container " + octx.outputFormat().name());
        }
        stream.setTimeBase(source_stream_.timeBase());
        avcodec_parameters_copy(stream.raw()->codecpar, source_stream_.raw()->codecpar);
        //out_coder_.addFlags(octx.outputFormat().isFlags(AVFMT_GLOBALHEADER) ? AV_CODEC_FLAG_GLOBAL_HEADER : 0);
    }
    virtual av::Codec& encodingCodec() override {
        return codec_;
    }
    virtual AVCodecParameters* codecParameters() override {
        return source_stream_.raw()->codecpar;
    }
    virtual av::Rational timeBase() override {
        return source_stream_.timeBase();
    }
    virtual std::string codecMediaTypeString() const override {
        return mediaTypeToString(source_stream_.mediaType());
    }
    virtual std::string codecName() const override {
        if (codec_.isNull()) {
            return "";
        }
        return codec_.name();
    }
    void ensureVideo() const {
        if (!vdec_.isValid()) {
            throw Error("video-related function called for non-video packet relay");
        }
    }
    virtual std::string fieldOrderString() const override {
        ensureVideo();
        return fieldOrderToString(vdec_.raw()->field_order);
    }
    virtual av::Rational frameRate() override {
        ensureVideo();
        return vdec_.raw()->framerate;
    };
    virtual int width() override {
        ensureVideo();
        return vdec_.width();
    }
    virtual int height() override {
        ensureVideo();
        return vdec_.height();
    }
    virtual av::PixelFormat pixelFormat() override {
        ensureVideo();
        return vdec_.pixelFormat();
    }
    virtual void discardUntil(av::Timestamp pts) override {
        logstream << "packet_relay ignoring discardUntil";
    }
    PacketRelay(std::unique_ptr<Source<av::Packet>> &&source, std::unique_ptr<Sink<av::Packet>> &&sink, av::Stream source_stream): TransparentNode<av::Packet>(std::move(source), std::move(sink)), source_stream_(source_stream), in_decoder_(source_stream_), codec_(in_decoder_.codec()) {
        if (source_stream_.mediaType() == AVMEDIA_TYPE_VIDEO) {
            vdec_ = av::VideoDecoderContext(source_stream);
        }
    }
    static std::shared_ptr<PacketRelay> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> src_edge = edges.find<av::Packet>(params["src"]);
        std::shared_ptr<IStreamsInput> input = src_edge->findNodeUp<IStreamsInput>();
        std::shared_ptr<InputStreamMetadata> md = nullptr;
        if (input != nullptr) md = src_edge->findMetadataUp<InputStreamMetadata>();
        if (md == nullptr) {
            throw Error("Couldn't initialize packet relay: no input stream.");
        }
        
        return NodeSISO<av::Packet, av::Packet>::template createCommon<PacketRelay>(edges, params, md->source_stream);
    }
};

DECLNODE(packet_relay, PacketRelay);
