#include "../node_common.hpp"

#include "fabric_protocol.hpp"

#include <avcpp/codeccontext.h>

class FabricUnpack: public NodeSISO<FabricPacket, av::Packet>,
                    public IDecoder /* not really */, public IEncoder /* not really */,
                    public ITimeBaseSource, public IVideoFormatSource, public IFrameRateSource {
protected:
    av::Stream source_stream_;
    av::VideoDecoderContext vdec_;
    av::Codec codec_;

public:
    FabricUnpack(std::unique_ptr<Source<FabricPacket>> &&source,
                 std::unique_ptr<Sink<av::Packet>> &&sink,
                 av::Stream source_stream):
        NodeSISO<FabricPacket, av::Packet>(std::move(source), std::move(sink)),
        source_stream_(source_stream) {
        codec_ = av::findEncodingCodec(source_stream_.raw()->codecpar->codec_id);
        if (source_stream_.direction() == av::Direction::Decoding && source_stream_.mediaType() == AVMEDIA_TYPE_VIDEO) {
            vdec_ = av::VideoDecoderContext(source_stream_);
        }
    }

    void process() override {
        FabricPacket *ptr = this->source_->peek();
        if (ptr == nullptr) return;
        FabricPacket fp = *ptr;
        this->source_->pop();
        if (!fp.complete || fp.message_type != avp_fabric::MSG_MEDIA) return;

        av::Packet pkt = fp.packet;
        av::Rational tb(fp.time_base_num, fp.time_base_den);
        pkt.setTimeBase(tb);
        pkt.setPts(av::Timestamp(fp.raw_pts, tb));
        if (fp.raw_dts != AV_NOPTS_VALUE) {
            pkt.setDts(av::Timestamp(fp.raw_dts, tb));
        } else {
            pkt.setDts(av::Timestamp(fp.raw_pts, tb));
        }
        pkt.setDuration(static_cast<int>(fp.duration));
        pkt.setStreamIndex(0);
        pkt.setKeyPacket((fp.packet_flags & avp_fabric::FLAG_KEYFRAME) != 0);
        pkt.setComplete(true);

        const AVCodecID codec_id = avp_fabric::codecIdFromWire(fp.codec);
        if (codec_id != AV_CODEC_ID_NONE && source_stream_.raw()->codecpar->codec_id != codec_id) {
            source_stream_.raw()->codecpar->codec_id = codec_id;
            codec_ = av::findEncodingCodec(codec_id);
        }
        this->sink_->put(pkt);
    }

    void setOutput(av::Stream &stream, av::FormatContext &octx) override {
        if (codec_.isNull()) {
            throw Error("codec is null when trying to init fabric_unpack");
        }
        if (!octx.outputFormat().codecSupported(codec_)) {
            throw Error(std::string("Codec ") + codec_.name() + " not supported by container " + octx.outputFormat().name());
        }
        stream.setTimeBase(source_stream_.timeBase());
        avcodec_parameters_copy(stream.raw()->codecpar, source_stream_.raw()->codecpar);
    }

    av::Codec& encodingCodec() override {
        return codec_;
    }

    AVCodecParameters* codecParameters() override {
        return source_stream_.raw()->codecpar;
    }

    av::Rational timeBase() override {
        return source_stream_.timeBase();
    }

    std::string codecMediaTypeString() const override {
        return mediaTypeToString(source_stream_.mediaType());
    }

    std::string codecName() const override {
        if (codec_.isNull()) return "";
        return codec_.name();
    }

    void ensureVideo() const {
        if (source_stream_.mediaType() != AVMEDIA_TYPE_VIDEO) {
            throw Error("video-related function called for non-video fabric_unpack");
        }
    }

    std::string fieldOrderString() const override {
        ensureVideo();
        if (vdec_.isValid()) return fieldOrderToString(vdec_.raw()->field_order);
        return "";
    }

    av::Rational frameRate() override {
        ensureVideo();
        if (vdec_.isValid()) return vdec_.raw()->framerate;
        return source_stream_.frameRate();
    };

    int width() override {
        ensureVideo();
        if (vdec_.isValid()) return vdec_.width();
        return source_stream_.raw()->codecpar->width;
    }

    int height() override {
        ensureVideo();
        if (vdec_.isValid()) return vdec_.height();
        return source_stream_.raw()->codecpar->height;
    }

    av::PixelFormat pixelFormat() override {
        ensureVideo();
        if (vdec_.isValid()) return vdec_.pixelFormat();
        return av::PixelFormat(static_cast<AVPixelFormat>(source_stream_.raw()->codecpar->format));
    }

    void discardUntil(av::Timestamp pts) override {
        logstream << "fabric_unpack ignoring discardUntil";
    }

    static std::shared_ptr<FabricUnpack> create(NodeCreationInfo &nci) {
        auto src_edge = nci.edges.find<FabricPacket>(nci.params["src"]);
        std::shared_ptr<IStreamsInput> input = src_edge->findNodeUp<IStreamsInput>();
        std::shared_ptr<InputStreamMetadata> md = nullptr;
        if (input != nullptr) md = src_edge->findMetadataUp<InputStreamMetadata>();
        if (md == nullptr) {
            throw Error("Couldn't initialize fabric_unpack: no input stream.");
        }
        return NodeSISO<FabricPacket, av::Packet>::template createCommon<FabricUnpack>(
            nci.edges, nci.params, md->source_stream);
    }
};

DECLNODE(fabric_unpack, FabricUnpack);
