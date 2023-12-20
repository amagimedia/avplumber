#include "node_common.hpp"
extern "C" {
#include <libavcodec/bsf.h>
}

class BitStreamFilterNode: public NodeSISO<av::Packet, av::Packet>, public IEncoder /* not really */, public IFlushable, public ReportsFinishByFlag {
protected:
    AVBSFContext* ctx_ = nullptr;
    AVCodecParameters* out_codecpar_ = nullptr;
    bool extradata_updated_ = false;
public:
    BitStreamFilterNode(std::unique_ptr<Source<av::Packet>> &&source, std::unique_ptr<Sink<av::Packet>> &&sink, const std::string filter_string): NodeSISO<av::Packet, av::Packet>(std::move(source), std::move(sink)) {
        int ret;
        ret = av_bsf_list_parse_str(filter_string.c_str(), &ctx_);
        if (ret < 0) {
            throw Error("Couldn't create BSF context: " + av::error2string(ret));
        }
        
        std::shared_ptr<ITimeBaseSource> tbsrc = findNodeUp<ITimeBaseSource>();
        if (!tbsrc) {
            throw Error("No timebase source above in chain");
        }
        ctx_->time_base_in = tbsrc->timeBase();
        
        std::shared_ptr<IEncoder> enc = findNodeUp<IEncoder>();
        if (!enc) {
            throw Error("No packets source above in chain");
        }
        
        ret = avcodec_parameters_copy(ctx_->par_in, enc->codecParameters());
        if (ret < 0) {
            throw Error("Couldn't copy input parameters to BSF");
        }
        
        ret = av_bsf_init(ctx_);
        if (ret < 0) {
            throw Error("Couldn't initialize BSF context: " + std::to_string(ret));
        }
    }
    virtual av::Codec& encodingCodec() {
        std::shared_ptr<IEncoder> enc = findNodeUp<IEncoder>();
        if (!enc) {
            throw Error("Couldn't forward encodingCodec() call: No packets source above in chain");
        }
        return enc->encodingCodec();
    }
    virtual AVCodecParameters* codecParameters() {
        ensureNotNull(out_codecpar_, "out codecpar null");
        return out_codecpar_;
    }
    virtual void setOutput(av::Stream &stream, av::FormatContext &octx) {
        std::shared_ptr<IEncoder> enc = findNodeUp<IEncoder>();
        if (!enc) {
            throw Error("Couldn't forward setOutput call: No packets source above in chain");
        }
        enc->setOutput(stream, octx);
        stream.setTimeBase(ctx_->time_base_out);
    }
    virtual void setOutputPostOpen(av::Stream &stream, av::FormatContext &octx) {
        out_codecpar_ = stream.raw()->codecpar;
        avcodec_parameters_copy(out_codecpar_, ctx_->par_out);
        //out_codecpar_->codec_tag = 0;
    }
    virtual void process() {
        av::Packet* pktp = this->source_->peek();
        if (pktp!=nullptr) {
            //logstream << "in: Rescaling PTS " << pktp->pts() << " to tb " << av::Rational(ctx_->time_base_in);
            pktp->setTimeBase(ctx_->time_base_in);
            int ret = av_bsf_send_packet(ctx_, pktp->raw());
            this->source_->pop();
            if (ret < 0) {
                throw Error("Couldn't send packet to BSF: " + std::to_string(ret));
            }
            outputPackets();
        }
    }
    virtual void flush() {
        av_bsf_send_packet(ctx_, nullptr);
        outputPackets();
        this->finished_ = true;
    }
protected:
    void outputPackets() {
        while (true) {
            av::Packet tmppkt;
            int ret = av_bsf_receive_packet(ctx_, tmppkt.raw());
            if (ret >= 0) {
                tmppkt.setComplete(true);
                av::Packet &outpkt = tmppkt; //.clone();
                outpkt.setTimeBase(ctx_->time_base_out);
                if ((!extradata_updated_) && out_codecpar_) {
                    avcodec_parameters_copy(out_codecpar_, ctx_->par_out);
                    extradata_updated_ = true;
                }
                this->sink_->put(outpkt);
            } else if ( (ret == AVERROR(EAGAIN)) || (ret == AVERROR_EOF) ) {
                break;
            } else if (ret < 0) {
                throw Error("Couldn't receive packet from BSF: " + std::to_string(ret));
            }
        }
    }
public:
    static std::shared_ptr<BitStreamFilterNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::string filter_string = params["bsf"];
        return NodeSISO<av::Packet, av::Packet>::template createCommon<BitStreamFilterNode>(edges, params, filter_string);
    }
};

DECLNODE(bsf, BitStreamFilterNode);
