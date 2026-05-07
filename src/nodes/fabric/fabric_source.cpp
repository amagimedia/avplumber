#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_protocol.hpp"
#include "fabric_rdm_receiver.hpp"
#include "fabric_source_timeline.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <arpa/inet.h>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

class FabricSourceNode: public NodeSingleOutput<av::Packet>,
                        public ReportsFinishByFlag,
                        public IStoppable,
                        public IInterruptible,
                        public IStreamsInput {
    std::string provider_;
    std::string bind_host_;
    std::string bind_port_;
    std::vector<std::string> bind_ports_;
    std::string addr_file_;
    size_t max_payload_bytes_ = 4 * 1024 * 1024;
    size_t queue_depth_ = 64;
    std::string redundancy_mode_ = "single";
    uint32_t active_replica_id_ = 1;
    uint32_t standby_replica_id_ = 2;
    uint64_t playout_delay_frames_ = 0;
    uint64_t promote_after_misses_ = 1;
    uint64_t active_timeout_ms_ = 50;
    std::string standby_control_host_ = "127.0.0.1";
    uint16_t standby_control_port_ = 0;
    std::unordered_map<uint32_t, uint16_t> control_ports_by_replica_;
    int control_fd_ = -1;
    av::FormatContext format_ctx_;
    av::Stream stream_;
    std::vector<std::unique_ptr<avp_fabric::RdmReceiver>> receivers_;
    size_t next_receiver_ = 0;
    avp_fabric::SourceTimeline timeline_;
    std::atomic_bool stopping_{false};
    uint64_t frames_received_ = 0;
    uint64_t bytes_received_ = 0;
    uint64_t report_start_ns_ = 0;
    uint64_t report_last_ns_ = 0;
    uint64_t report_last_bytes_ = 0;
    uint64_t report_last_frames_ = 0;
    uint64_t frames_emitted_ = 0;
    uint64_t bytes_emitted_ = 0;
    uint64_t emit_report_last_ns_ = 0;
    uint64_t emit_report_last_bytes_ = 0;
    uint64_t emit_report_last_frames_ = 0;

public:
    FabricSourceNode(std::unique_ptr<Sink<av::Packet>> &&sink, const Parameters &params):
        NodeSingleOutput<av::Packet>(std::move(sink)) {
        provider_ = params.value("provider", "shm");
        bind_host_ = params.value("bind_host", std::string());
        if (params.count("bind_ports")) {
            for (const auto &p: params.at("bind_ports")) {
                if (p.is_number_integer()) {
                    bind_ports_.push_back(std::to_string(p.get<int>()));
                } else {
                    bind_ports_.push_back(p.get<std::string>());
                }
            }
        } else {
            if (params.at("bind_port").is_number_integer()) {
                bind_port_ = std::to_string(params.at("bind_port").get<int>());
            } else {
                bind_port_ = params.at("bind_port").get<std::string>();
            }
            bind_ports_.push_back(bind_port_);
        }
        addr_file_ = params.value("addr_file", std::string());
        max_payload_bytes_ = params.value("max_payload_bytes", max_payload_bytes_);
        queue_depth_ = params.value("queue_depth", queue_depth_);
        redundancy_mode_ = params.value("redundancy_mode", redundancy_mode_);
        active_replica_id_ = params.value("active_replica_id", active_replica_id_);
        standby_replica_id_ = params.value("standby_replica_id", standby_replica_id_);
        playout_delay_frames_ = params.value("playout_delay_frames", playout_delay_frames_);
        promote_after_misses_ = params.value("promote_after_misses", promote_after_misses_);
        active_timeout_ms_ = params.value("active_timeout_ms", active_timeout_ms_);
        standby_control_host_ = params.value("standby_control_host", standby_control_host_);
        standby_control_port_ = params.value("standby_control_port", 0);
        if (params.count("standby_control_ports")) {
            for (auto it = params["standby_control_ports"].begin(); it != params["standby_control_ports"].end(); ++it) {
                control_ports_by_replica_[static_cast<uint32_t>(std::stoul(it.key()))] = it.value().get<uint16_t>();
            }
        }
        if (standby_control_port_) {
            control_ports_by_replica_[standby_replica_id_] = standby_control_port_;
        }
        if (redundancy_mode_ != "single" &&
            redundancy_mode_ != "hot_hot_first_complete" &&
            redundancy_mode_ != "active_standby_repair") {
            throw Error("fabric_source unsupported redundancy_mode: " + redundancy_mode_);
        }
        timeline_ = avp_fabric::SourceTimeline(
            redundancy_mode_, active_replica_id_, playout_delay_frames_, promote_after_misses_, active_timeout_ms_);

        if (params.count("reference_url")) {
            av::Dictionary opts;
            if (params.count("reference_options")) {
                opts = parametersToDict(params["reference_options"]);
            }
            format_ctx_.openInput(params["reference_url"].get<std::string>(), opts);
            format_ctx_.findStreamInfo();
            for (size_t i = 0; i < format_ctx_.streamsCount(); ++i) {
                av::Stream st = format_ctx_.stream(i);
                if (st.isVideo()) {
                    stream_ = st;
                    break;
                }
            }
            if (!stream_.isValid()) throw Error("fabric_source reference_url has no video stream");
            AVCodecParameters *cp = stream_.raw()->codecpar;
            if (cp->extradata) {
                av_freep(&cp->extradata);
                cp->extradata_size = 0;
            }
            cp->codec_tag = 0;
        } else {
            const int width = params.value("width", 1920);
            const int height = params.value("height", 1080);
            const av::Rational time_base = params.count("time_base") ?
                parseRatio(params["time_base"].get<std::string>()) : av::Rational(1, 1000);
            const av::Rational frame_rate = params.count("frame_rate") ?
                parseRatio(params["frame_rate"].get<std::string>()) : av::Rational(30000, 1001);
            const av::PixelFormat real_pixel_format(params.value("real_pixel_format", std::string("yuv420p")));

            stream_ = format_ctx_.addStream();
            stream_.setTimeBase(time_base);
            stream_.setFrameRate(frame_rate);
            stream_.setAverageFrameRate(frame_rate);
            AVCodecParameters *cp = stream_.raw()->codecpar;
            cp->codec_type = AVMEDIA_TYPE_VIDEO;
            cp->codec_id = avp_fabric::codecIdFromName(params.value("codec", std::string("h264_intra")));
            cp->width = width;
            cp->height = height;
            cp->format = real_pixel_format.get();
        }
        if (!control_ports_by_replica_.empty()) {
            control_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
            if (control_fd_ < 0) throw Error("fabric_source control socket failed");
        }
    }

    virtual ~FabricSourceNode() {
        if (control_fd_ >= 0) close(control_fd_);
    }

    virtual void stop() override {
        stopping_.store(true, std::memory_order_release);
        finished_ = true;
        for (auto &receiver: receivers_) {
            receiver->stop();
        }
    }

    virtual void interrupt() override {
        stop();
    }

    virtual void init(EdgeManager &edges, const Parameters &params) override {
        NodeSingleOutput<av::Packet>::init(edges, params);
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        std::shared_ptr<InputStreamMetadata> md = edge->metadata<InputStreamMetadata>(true);
        md->source_stream = stream_;
        const size_t max_msg = sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader) + max_payload_bytes_;
        for (size_t i = 0; i < bind_ports_.size(); ++i) {
            const std::string addr_file = (i == 0) ? addr_file_ : std::string();
            receivers_.push_back(std::make_unique<avp_fabric::RdmReceiver>(
                provider_, bind_host_, bind_ports_[i], addr_file, max_msg, queue_depth_, "fabric_source"));
        }
    }

    virtual void process() override {
        if (stopping_.load(std::memory_order_acquire)) return;
        if (redundancy_mode_ == "single") {
            avp_fabric::SourceCandidate c;
            do {
                if (!receiveOneRecord(c)) return;
            } while (c.payload.empty() && !stopping_.load(std::memory_order_acquire));
            if (c.payload.empty()) return;
            emitCandidate(c);
            return;
        }

        while (!stopping_.load(std::memory_order_acquire)) {
            avp_fabric::SourceCandidate selected;
            if (timeline_.trySelect(selected, controlSender())) {
                emitCandidate(selected);
                return;
            }
            avp_fabric::SourceCandidate ignored;
            if (!receiveOneRecord(ignored)) return;
        }
    }

    virtual size_t streamsCount() override { return 1; }
    virtual av::Stream stream(size_t i) override {
        if (i != 0) throw Error("fabric_source stream index out of range");
        return stream_;
    }
    virtual void discardAllStreams() override {}
    virtual void enableStream(size_t) override {}
    virtual av::FormatContext& formatContext() override { return format_ctx_; }

    static std::shared_ptr<FabricSourceNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        return std::make_shared<FabricSourceNode>(make_unique<EdgeSink<av::Packet>>(edge), params);
    }

private:
    avp_fabric::SourceTimeline::SendControl controlSender() {
        return [this](uint32_t replica_id, const std::string &cmd, int64_t normalized_pts) {
            sendControl(replica_id, cmd, normalized_pts);
        };
    }

    bool receiveOneRecord(avp_fabric::SourceCandidate &out) {
        if (receivers_.empty()) throw Error("fabric_source has no receivers");
        size_t receiver_idx = 0;
        size_t len = 0;
        if (receivers_.size() == 1) {
            receiver_idx = 0;
            if (!receivers_[0]->recvOne(len)) return false;
        } else {
            while (!stopping_.load(std::memory_order_acquire)) {
                for (size_t i = 0; i < receivers_.size(); ++i) {
                    const size_t idx = (next_receiver_ + i) % receivers_.size();
                    if (receivers_[idx]->tryRecvOne(len)) {
                        receiver_idx = idx;
                        next_receiver_ = (idx + 1) % receivers_.size();
                        goto got_record;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        }
got_record:
        const avp_fabric::RdmReceiver *receiver = receivers_[receiver_idx].get();
        if (len < sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader)) {
            throw Error("fabric_source received message shorter than AVP headers");
        }

        const auto *wire = reinterpret_cast<const avp_fabric::WireHeader *>(receiver->data());
        const auto *media = reinterpret_cast<const avp_fabric::MediaHeader *>(receiver->data() + sizeof(avp_fabric::WireHeader));
        avp_fabric::validateWireRecord(*wire, len, max_payload_bytes_, "fabric_source");

        const uint64_t received_ns = avp_fabric::monotonicNs();
        const int64_t normalized_pts =
            (wire->message_type == avp_fabric::MSG_FRAME_STATUS) ?
            timeline_.ingestStatus(*media, received_ns, controlSender()) :
            timeline_.ingestMedia(*media, receiver->data() + wire->header_bytes, wire->payload_bytes, received_ns);
        if (wire->message_type == avp_fabric::MSG_FRAME_STATUS) {
            out.payload.clear();
        } else {
            out.media = *media;
            out.payload.assign(receiver->data() + wire->header_bytes, receiver->data() + wire->header_bytes + wire->payload_bytes);
            out.normalized_frame_id = normalized_pts;
        }

        if (frames_received_ < 5) {
            logstream << "fabric_source received"
                      << " type=" << (wire->message_type == avp_fabric::MSG_MEDIA ? "media" : "status")
                      << " replica_id=" << media->replica_id
                      << " generation=" << media->generation
                      << " pts=" << media->pts
                      << " normalized_pts=" << normalized_pts
                      << " message_bytes=" << len
                      << " payload_bytes=" << wire->payload_bytes
                      << " payload_prefix=" << avp_fabric::hexPrefix(receiver->data() + wire->header_bytes, wire->payload_bytes);
        }
        frames_received_++;
        bytes_received_ += wire->payload_bytes;
        reportThroughput();
        return true;
    }

    void emitCandidate(const avp_fabric::SourceCandidate &c) {
        av::Packet pkt(c.payload);
        av::Rational tb(c.media.time_base_num, c.media.time_base_den);
        const int64_t pts_delta = c.normalized_frame_id - c.media.pts;
        pkt.setTimeBase(tb);
        pkt.setPts(av::Timestamp(static_cast<int64_t>(c.media.pts) + pts_delta, tb));
        if (c.media.dts != AV_NOPTS_VALUE) {
            pkt.setDts(av::Timestamp(static_cast<int64_t>(c.media.dts) + pts_delta, tb));
        } else {
            pkt.setDts(av::Timestamp(static_cast<int64_t>(c.media.pts) + pts_delta, tb));
        }
        pkt.setDuration(static_cast<int>(c.media.duration));
        pkt.setStreamIndex(0);
        pkt.setKeyPacket((c.media.packet_flags & avp_fabric::FLAG_KEYFRAME) != 0);
        if (avp_fabric::codecIdFromWire(c.media.codec) != AV_CODEC_ID_NONE) {
            AVCodecParameters *cp = stream_.raw()->codecpar;
            if (cp->codec_id == AV_CODEC_ID_NONE || cp->codec_id == AV_CODEC_ID_H264) {
                cp->codec_id = avp_fabric::codecIdFromWire(c.media.codec);
            }
        }
        pkt.setComplete(true);
        this->sink_->put(pkt);
        frames_emitted_++;
        bytes_emitted_ += c.payload.size();
        reportEmitThroughput(c);
    }

    void sendControl(uint32_t replica_id, const std::string &cmd, int64_t normalized_pts) {
        if (control_fd_ < 0) return;
        auto port_it = control_ports_by_replica_.find(replica_id);
        if (port_it == control_ports_by_replica_.end() || !port_it->second) return;
        const int64_t raw_pts = timeline_.denormalizeFrameId(replica_id, normalized_pts);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_it->second);
        if (inet_pton(AF_INET, standby_control_host_.c_str(), &addr.sin_addr) != 1) {
            logstream << "fabric_source invalid standby_control_host=" << standby_control_host_;
            return;
        }
        const std::string msg = cmd + " " + std::to_string(raw_pts);
        sendto(control_fd_, msg.data(), msg.size(), 0, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
        logstream << "fabric_source control " << cmd
                  << " replica_id=" << replica_id
                  << " normalized_pts=" << normalized_pts
                  << " raw_pts=" << raw_pts;
    }

    void reportThroughput() {
        const uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (!report_start_ns_) {
            report_start_ns_ = now;
            report_last_ns_ = now;
            report_last_bytes_ = bytes_received_;
            report_last_frames_ = frames_received_;
            return;
        }
        const uint64_t elapsed_ns = now - report_last_ns_;
        if (elapsed_ns < 1000000000ull) return;

        const uint64_t delta_bytes = bytes_received_ - report_last_bytes_;
        const uint64_t delta_frames = frames_received_ - report_last_frames_;
        const double seconds = static_cast<double>(elapsed_ns) / 1000000000.0;
        const double mbps = static_cast<double>(delta_bytes) * 8.0 / seconds / 1000000.0;
        const double mbytes = static_cast<double>(delta_bytes) / seconds / 1000000.0;
        const double fps = static_cast<double>(delta_frames) / seconds;
        logstream << "fabric_source throughput"
                  << " mbps=" << mbps
                  << " MBps=" << mbytes
                  << " fps=" << fps
                  << " total_frames=" << frames_received_
                  << " total_bytes=" << bytes_received_;
        report_last_ns_ = now;
        report_last_bytes_ = bytes_received_;
        report_last_frames_ = frames_received_;
    }

    void reportEmitThroughput(const avp_fabric::SourceCandidate &c) {
        const uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (!emit_report_last_ns_) {
            emit_report_last_ns_ = now;
            emit_report_last_bytes_ = bytes_emitted_;
            emit_report_last_frames_ = frames_emitted_;
            logstream << "fabric_source emitted"
                      << " replica_id=" << c.media.replica_id
                      << " normalized_pts=" << c.normalized_frame_id
                      << " payload_bytes=" << c.payload.size()
                      << " total_emitted=" << frames_emitted_;
            return;
        }
        const uint64_t elapsed_ns = now - emit_report_last_ns_;
        if (elapsed_ns < 1000000000ull) return;

        const uint64_t delta_bytes = bytes_emitted_ - emit_report_last_bytes_;
        const uint64_t delta_frames = frames_emitted_ - emit_report_last_frames_;
        const double seconds = static_cast<double>(elapsed_ns) / 1000000000.0;
        logstream << "fabric_source emit throughput"
                  << " replica_id=" << c.media.replica_id
                  << " mbps=" << (static_cast<double>(delta_bytes) * 8.0 / seconds / 1000000.0)
                  << " MBps=" << (static_cast<double>(delta_bytes) / seconds / 1000000.0)
                  << " fps=" << (static_cast<double>(delta_frames) / seconds)
                  << " total_emitted=" << frames_emitted_
                  << " total_emit_bytes=" << bytes_emitted_;
        emit_report_last_ns_ = now;
        emit_report_last_bytes_ = bytes_emitted_;
        emit_report_last_frames_ = frames_emitted_;
    }
};

#else

class FabricSourceNode: public NodeSingleOutput<av::Packet>, public ReportsFinishByFlag {
public:
    using NodeSingleOutput<av::Packet>::NodeSingleOutput;
    virtual void process() override {
        throw Error("fabric_source requested, but avplumber was built with HAVE_LIBFABRIC=0");
    }
    static std::shared_ptr<FabricSourceNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        return std::make_shared<FabricSourceNode>(make_unique<EdgeSink<av::Packet>>(edge));
    }
};

#endif

DECLNODE(fabric_source, FabricSourceNode);
