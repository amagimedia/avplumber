#include <chrono>
#include "node_common.hpp"
#include "../OutputProgress.hpp"
#include "../OutputLogRedaction.hpp"
#include <atomic>
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

class StreamOutput: public NodeSingleInput<av::Packet>, public IFlushable, public ReportsFinishByFlag,
                    public IInputsObjects {
protected:
    av::FormatContext octx_;
    bool redact_url_ = false;
    OutputProgress progress_;
    bool fifo_ = false;
    bool fifo_failed_ = false;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int64_t> finalize_deadline_ms_{0};
    static int64_t steadyMilliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static int interruptFifo(void* opaque) {
        auto* self = static_cast<StreamOutput*>(opaque);
        const auto deadline = self->finalize_deadline_ms_.load();
        return self->stop_requested_.load() || (deadline && steadyMilliseconds() >= deadline);
    }
    // `hold`: packets are taken and not written, so the chain before the output runs (an encoder
    // opens on its first frame) while nothing leaves. Released, writing starts at a key packet.
    std::atomic<bool> held_{false};
    bool wait_for_key_ = false;
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
    std::chrono::steady_clock::time_point first_error_{};
    bool udp_ = false;   // the url's protocol is udp: its paced send queue can fill
    uint64_t queue_full_drops_ = 0;   // since the last line about them
    std::chrono::steady_clock::time_point queue_full_logged_{};
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
    ~StreamOutput() override {
        if (fifo_ && octx_.isOpened()) {
            stop_requested_.store(true);
            try { flush(); } catch (...) { }
        }
    }
    av::FormatContext& ctx() {
        return octx_;
    }
    void stop() override {
        stop_requested_.store(true);
        NodeSingleInput<av::Packet>::stop();
    }
    virtual void process() {
        av::Packet pkt = this->source_->get();
        if (pkt && isEofMarker(pkt)) {
            flush();
            return;
        }
        if (fifo_failed_) return; // Drain only this optional destination until real EOF.
        if (pkt && held_.load(std::memory_order_acquire)) {
            wait_for_key_ = true;
            return;
        }
        if (pkt && wait_for_key_) {
            if (!pkt.isKeyPacket()) return;
            wait_for_key_ = false;
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
                const auto payload_bytes = pkt.size();
                octx_.writePacket(pkt);
                progress_.wrote(payload_bytes, std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
                errors_ = 0;
            } catch (std::exception &e) {
                if (fifo_) {
                    // FIFO handles reconnects internally. A terminal worker failure
                    // disables only this transport; retain its consumer through EOF.
                    fifo_failed_ = true;
                    logstream << "live transport disabled after terminal FIFO error (destination redacted)";
                    return;
                }
                // FFmpeg's paced UDP sender (bitrate=, fifo_size=) reports its full send queue as
                // ENOMEM. An encoder catching up after a stall overfills it for a moment; that
                // packet is lost like any on the network (receivers ask for a keyframe), and the
                // output goes on. 2026-10-15: these drops reached the panic policy and restarted
                // the whole show four times. Any other protocol's ENOMEM is an error like the rest.
                const auto *code = dynamic_cast<const std::system_error *>(&e);
                if (udp_ && code && code->code().value() == AVERROR(ENOMEM)) {
                    ++queue_full_drops_;
                    const auto now = std::chrono::steady_clock::now();
                    if (now - queue_full_logged_ >= std::chrono::seconds(1)) {
                        logstream << "output: send queue full, " << queue_full_drops_ << " packet(s) dropped";
                        queue_full_drops_ = 0;
                        queue_full_logged_ = now;
                    }
                    return;
                }
                logstream << "writePacket failed: " << (redact_url_ ? "destination redacted" : e.what());
                // A burst of packets after a stall can fail for a few hundred milliseconds at once
                // (2026-10-15: 21 consecutive UDP writes failed within 100 ms on both program outputs
                // and the panic policy stopped the whole show). Give up only when the failures have
                // persisted for a while, not after a packet count that one burst exceeds.
                const auto now = std::chrono::steady_clock::now();
                if (errors_ == 0) first_error_ = now;
                errors_++;
                if (errors_ > 20 && now - first_error_ > std::chrono::seconds(3)) {
                    if (redact_url_) throw Error("output write failed (destination redacted)");
                    throw;
                }
            }
        }
    }
    void setObject(const std::string key, const Parameters &value) override {
        if (key != "hold") throw Error("output: unknown object key: " + key);
        held_.store(value.get<bool>(), std::memory_order_release);
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
        if (fifo_) finalize_deadline_ms_.store(steadyMilliseconds() + 3000);
        try {
            octx_.writeTrailer();
            octx_.close();
        } catch (...) {
            if (fifo_) {
                logstream << "live transport finalization ended with unavailable destination";
                octx_.close();
                this->finished_ = true;
                return;
            }
            if (redact_url_) throw Error("output finalization failed (destination redacted)");
            throw;
        }
        progress_.flush();
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
        
        r->redact_url_ = params.value("redact_url", false);
        r->fifo_ = params.count("fifo_options") != 0;
        if (r->redact_url_) OutputLogRedaction::add(url);
        r->progress_.setPath(params.value("progress_file", std::string{}));
        r->progress_.setSubmitted(r->fifo_);
        logstream << "output url: " << (r->redact_url_ ? "[redacted]" : url);
        const char *protocol = avio_find_protocol_name(url.c_str());
        r->udp_ = protocol && std::string(protocol) == "udp";
        
        av::OutputFormat ofmt(format, url);
        octx.setFormat(ofmt);
        
        std::shared_ptr<IMuxer> muxer = edge->findNodeUp<IMuxer>();
        if (muxer==nullptr) {
            throw Error("Muxer is mandatory before output!");
        }
        
        muxer->initFromFormatContext(octx);
        if (r->fifo_) {
            // Encoders see the real muxer's codec/global-header requirements once.
            // FIFO owns network open/recovery and clones these codec parameters;
            // neither recovery nor a missing receiver reinitializes an encoder.
            Parameters fifo_params = params["fifo_options"];
            fifo_params["fifo_format"] = format;
            fifo_params["format_opts"] = opts.toString('=', ':');
            opts = parametersToDict(fifo_params);
            octx.setFormat(av::OutputFormat("fifo", url));
            // Bypass AVCPP's non-atomic access timer: FIFO invokes this callback
            // on its own worker. Protocol rw_timeout bounds individual I/O calls.
            octx.raw()->interrupt_callback = {interruptFifo, r.get()};
        }
        
        octx.raw()->url = av_strdup(url.c_str()); // workaround for avcpp not using avformat_alloc_output_context2
        try {
            octx.openOutput(url, opts);
            muxer->initFromFormatContextPostOpenPreWriteHeader(octx);
            octx.writeHeader(opts);
        } catch (...) {
            if (r->redact_url_) throw Error("output initialization failed (destination redacted)");
            throw;
        }
        edge->setConsumer(r);
        
        muxer->initFromFormatContextPostOpen(octx);

        if (params.count("hold")) r->held_.store(params["hold"].get<bool>());
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
