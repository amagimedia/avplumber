#include "node_common.hpp"
#include <fstream>

#pragma pack(push)
#pragma pack(1)
struct SeekTableEntry {
    int64_t timestamp_ms;
    uint64_t bytes;
};
#pragma pack(pop)

class StreamOutput: public NodeSingleInput<av::Packet>, public IFlushable, public ReportsFinishByFlag {
protected:
    av::FormatContext octx_;
    bool write_seek_table_ = false;
    std::ofstream seek_table_text_;
    std::ofstream seek_table_bin_;
    bool should_close_ = false;
    int last_flush_ = 0;
    int errors_ = 0;
public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    av::FormatContext& ctx() {
        return octx_;
    }
    virtual void process() {
        av::Packet pkt = this->source_->get();
        if (pkt) {
            if (write_seek_table_ && octx_.raw() && octx_.raw()->pb && (pkt.streamIndex() == 0)) {
                int64_t cur_pos = avio_tell(octx_.raw()->pb);
                int64_t ts_ms = pkt.dts().timestamp({1, 1000});
                if (seek_table_text_.is_open()) {
                    seek_table_text_ << ts_ms << " " << cur_pos << "\n";
                    if (!last_flush_) {
                        seek_table_text_.flush();
                    }
                }
                if (seek_table_bin_ .is_open()) {
                    SeekTableEntry entry { ts_ms, uint64_t(cur_pos) };
                    seek_table_bin_.write(reinterpret_cast<char*>(&entry), sizeof(entry));
                    if (!last_flush_) {
                        seek_table_bin_.flush();
                    }
                }
                if (last_flush_++ > 10) {
                    last_flush_ = 0;
                }
            }
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

        if (params.count("seek_table")) {
            r->seek_table_bin_.open(params["seek_table"], std::ios::binary);
            r->write_seek_table_ = true;
        }
        if (params.count("seek_table_text")) {
            r->seek_table_text_.open(params["seek_table_text"]);
            r->write_seek_table_ = true;
        }
        
        return r;
    }
};

DECLNODE(output, StreamOutput);
