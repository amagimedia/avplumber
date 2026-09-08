#include "node_common.hpp"
#include <fstream>

#pragma pack(push)
#pragma pack(1)
struct SeekTableEntry {
    int64_t timestamp_ms;
    uint64_t bytes;
};
#pragma pack(pop)

#define SEEK_TABLE_FILES_COUNT 4
#define MAX_SEEK_TABLE_ENTRIES 60

class StreamOutput: public NodeSingleInput<av::Packet>, public IFlushable, public ReportsFinishByFlag {
protected:
    av::FormatContext octx_;
    bool write_seek_table_ = false;
    std::ofstream seek_table_text_[SEEK_TABLE_FILES_COUNT];
    std::ofstream seek_table_bin_[SEEK_TABLE_FILES_COUNT];
    std::vector<SeekTableEntry> seek_table_entries_;
    std::stringstream seek_table_text_stream_;
    std::string seek_table_url_;
    std::string seek_table_text_url_;
    int current_seek_table_index_ = 0;
    bool should_close_ = false;
    int errors_ = 0;
    void commitPendingSeekEntries() {
        if (seek_table_entries_.empty()) {
            return;
        }
        if (seek_table_text_[current_seek_table_index_].is_open()) {
            seek_table_text_[current_seek_table_index_] << seek_table_text_stream_.rdbuf();
        }
        if (seek_table_bin_[current_seek_table_index_].is_open()) {
            seek_table_bin_[current_seek_table_index_].write(
                reinterpret_cast<char*>(seek_table_entries_.data()),
                seek_table_entries_.size() * sizeof(SeekTableEntry));
        }
        seek_table_entries_.clear();
        seek_table_text_stream_.str("");
        seek_table_text_stream_.clear();
    }
    void publishCurrentSeekTables() {
        if (seek_table_text_[current_seek_table_index_].is_open()) {
            seek_table_text_[current_seek_table_index_].flush();
        }
        if (seek_table_bin_[current_seek_table_index_].is_open()) {
            seek_table_bin_[current_seek_table_index_].flush();
        }
        auto publish = [this](const std::string& url) {
            if (url.empty()) {
                return;
            }
            unlink(url.c_str());
            symlink((url + "." + std::to_string(current_seek_table_index_)).c_str(), url.c_str());
        };
        publish(seek_table_url_);
        publish(seek_table_text_url_);
    }
public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    av::FormatContext& ctx() {
        return octx_;
    }
    virtual void process() {
        av::Packet pkt = this->source_->get();
        if (pkt && isEofMarker(pkt)) {
            flush();
            return;
        }
        if (pkt) {
            try {
                if (write_seek_table_ && octx_.raw() && octx_.raw()->pb && (pkt.streamIndex() == 0)) {
                    int64_t cur_pos = avio_tell(octx_.raw()->pb);
                    int64_t ts_ms = pkt.dts().timestamp({1, 1000});

                    seek_table_entries_.push_back({ ts_ms, uint64_t(cur_pos) });
                    seek_table_text_stream_ << ts_ms << " " << cur_pos << "\n";

                    for (int i = 0; i < SEEK_TABLE_FILES_COUNT; i++) {
                        if (i == current_seek_table_index_) {
                            continue;
                        }

                        if (seek_table_text_[i].is_open()) {
                            seek_table_text_[i] << ts_ms << " " << cur_pos << "\n";
                        }
                        if (seek_table_bin_[i].is_open()) {
                            SeekTableEntry entry { ts_ms, uint64_t(cur_pos) };
                            seek_table_bin_[i].write(reinterpret_cast<char*>(&entry), sizeof(entry));
                        }
                    }

                    if (seek_table_entries_.size() >= MAX_SEEK_TABLE_ENTRIES) {
                        // rotate seek table files
                        commitPendingSeekEntries();
                        current_seek_table_index_ = (current_seek_table_index_ + 1) % SEEK_TABLE_FILES_COUNT;
                        publishCurrentSeekTables();
                    }
                }
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
        if (should_close_) {
            return;
        }
        should_close_ = true;
        if (write_seek_table_) {
            commitPendingSeekEntries();
            publishCurrentSeekTables();
        }
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
            r->seek_table_url_ = params["seek_table"];
            for (int i = 0; i < SEEK_TABLE_FILES_COUNT; i++) {
                r->seek_table_bin_[i].open(r->seek_table_url_ + "." + std::to_string(i), std::ios::binary);
            }
            r->write_seek_table_ = true;
        }
        if (params.count("seek_table_text")) {
            r->seek_table_text_url_ = params["seek_table_text"];
            for (int i = 0; i < SEEK_TABLE_FILES_COUNT; i++) {
                r->seek_table_text_[i].open(r->seek_table_text_url_ + "." + std::to_string(i));
            }
            r->write_seek_table_ = true;
        }
        
        return r;
    }
};

DECLNODE(output, StreamOutput);
