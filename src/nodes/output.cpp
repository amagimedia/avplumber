#include "node_common.hpp"

class StreamOutput: public NodeSingleInput<av::Packet>, public IFlushable, public ReportsFinishByFlag {
protected:
    av::FormatContext octx_;
    bool should_close_ = false;
    int errors_ = 0;
public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    av::FormatContext& ctx() {
        return octx_;
    }
    virtual void process() {
        av::Packet pkt = this->source_->get();
        if (pkt) {
            try {
                octx_.writePacket(pkt);
                errors_ = 0;
            } catch (std::exception &e) {
                logstream << "writePacket failed: " << e.what();
                errors_++;
                if (errors_>20) {
                    throw;
                }
            }
        }
    }
    virtual void flush() {
        octx_.writeTrailer();
        octx_.close();
        this->finished_ = true;
    }
    static std::shared_ptr<StreamOutput> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["src"]);
        auto r = std::make_shared<StreamOutput>(make_unique<EdgeSource<av::Packet>>(edge));
        av::FormatContext &octx = r->ctx();
        std::string format;
        if (params.count("format") > 0) {
            format = params["format"];
        }
        av::Dictionary opts;
        if (params.count("options") > 0) {
            opts = parametersToDict(params["options"]);
        }
        
        std::string url = params["url"];
        
        logstream << "output url: " << url;
        
        av::OutputFormat ofmt(format, url);
        octx.setFormat(ofmt);
        
        std::shared_ptr<IMuxer> muxer = edge->findNodeUp<IMuxer>();
        if (muxer==nullptr) {
            throw Error("Muxer is mandatory before output!");
        }
        
        muxer->initFromFormatContext(octx);
        
        octx.raw()->url = av_strdup(url.c_str()); // workaround for avcpp not using avformat_alloc_output_context2
        octx.openOutput(url, opts);
        
        muxer->initFromFormatContextPostOpenPreWriteHeader(octx);

        octx.writeHeader(opts);
        edge->setConsumer(r);
        
        muxer->initFromFormatContextPostOpen(octx);
        
        return r;
    }
};

DECLNODE(output, StreamOutput);
