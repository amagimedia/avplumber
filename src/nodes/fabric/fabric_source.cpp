#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_protocol.hpp"
#include "fabric_rdm_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <poll.h>
#include <string>
#include <thread>
#include <vector>

class FabricSourceNode: public NodeSingleOutput<FabricPacket>,
                        public ReportsFinishByFlag,
                        public IStoppable,
                        public IInterruptible,
                        public IStreamsInput {
    std::string provider_;
    std::string bind_host_;
    std::vector<std::string> bind_ports_;
    std::string addr_file_;
    size_t max_payload_bytes_ = 4 * 1024 * 1024;
    size_t queue_depth_ = 64;
    size_t rx_buffers_ = 16;
    // 0 keeps waiting for a peer forever (a receiver that survives producer restarts).
    // > 0 ends the session after that many ms without a record, as if the producer sent EOF.
    // Record silence is the only liveness signal the tcp provider gives a receive-only endpoint:
    // it does not report a clean peer close.
    int idle_timeout_ms_ = 0;
    uint64_t last_message_ns_ = 0;
    bool session_ended_ = false;
    av::FormatContext format_ctx_;
    av::Stream stream_;
    // Owns the alpha output edge's stream metadata: a fill stream describes the picture, not its
    // key, so the alpha edge gets a stream of its own with the alpha codec and pixel format.
    av::FormatContext alpha_format_ctx_;
    av::Stream alpha_stream_;
    bool alpha_enabled_ = false;
    std::unique_ptr<Sink<FabricPacket>> alpha_sink_;
    std::vector<std::unique_ptr<avp_fabric::RdmReceiver>> receivers_;
    size_t next_receiver_ = 0;
    std::atomic_bool stopping_{false};
    uint64_t frames_received_ = 0;
    uint64_t bytes_received_ = 0;
    uint64_t report_start_ns_ = 0;
    uint64_t report_last_ns_ = 0;
    uint64_t report_last_bytes_ = 0;
    uint64_t report_last_frames_ = 0;

public:
    FabricSourceNode(std::unique_ptr<Sink<FabricPacket>> &&sink, const Parameters &params):
        NodeSingleOutput<FabricPacket>(std::move(sink)) {
        provider_ = params.value("provider", "shm");
        bind_host_ = params.value("bind_host", std::string());
        if (params.count("bind_ports")) {
            for (const auto &p: params.at("bind_ports")) {
                bind_ports_.push_back(p.is_number_integer() ? std::to_string(p.get<int>()) : p.get<std::string>());
            }
        } else {
            bind_ports_.push_back(params.at("bind_port").is_number_integer()
                                      ? std::to_string(params.at("bind_port").get<int>())
                                      : params.at("bind_port").get<std::string>());
        }
        addr_file_ = params.value("addr_file", std::string());
        max_payload_bytes_ = params.value("max_payload_bytes", max_payload_bytes_);
        queue_depth_ = params.value("queue_depth", queue_depth_);
        // Receive buffers posted per port. Packets downstream hold theirs until freed, so this must
        // exceed the output queue plus what the selector and decoder hold.
        rx_buffers_ = params.value("rx_buffers", rx_buffers_);
        idle_timeout_ms_ = params.value("idle_timeout_ms", idle_timeout_ms_);
        if (idle_timeout_ms_ < 0) {
            throw Error("fabric_source idle_timeout_ms must be >= 0 (0 = wait forever)");
        }

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
            const av::Rational time_base = params.count("time_base")
                ? parseRatio(params["time_base"].get<std::string>())
                : av::Rational(1, 1000);
            const av::Rational frame_rate = params.count("frame_rate")
                ? parseRatio(params["frame_rate"].get<std::string>())
                : av::Rational(25, 1);
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
        if (params.count("alpha_dst")) {
            // Geometry and timing are the fill's: a key plane covers the picture it keys.
            alpha_enabled_ = true;
            const AVCodecID alpha_codec = params.count("alpha_codec")
                ? avp_fabric::codecIdFromName(params.at("alpha_codec").get<std::string>())
                : stream_.raw()->codecpar->codec_id;
            const std::string alpha_pixel_format_name = params.value("alpha_pixel_format", std::string("gray"));
            const av::PixelFormat alpha_pixel_format(alpha_pixel_format_name);
            if (alpha_pixel_format == AV_PIX_FMT_NONE) {
                throw Error("fabric_source unknown alpha_pixel_format: " + alpha_pixel_format_name);
            }
            const auto alpha_layout = av::PixelFormat(params.value("alpha_real_pixel_format",
                alpha_pixel_format == AV_PIX_FMT_CUDA ? std::string("gray") : alpha_pixel_format_name));
            if (alpha_layout == AV_PIX_FMT_NONE || (av_pix_fmt_desc_get(alpha_layout.get())->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                throw Error("fabric_source alpha_real_pixel_format must be a software pixel format");
            }
            alpha_stream_ = alpha_format_ctx_.addStream();
            alpha_stream_.setTimeBase(stream_.timeBase());
            alpha_stream_.setFrameRate(stream_.frameRate());
            alpha_stream_.setAverageFrameRate(stream_.frameRate());
            AVCodecParameters *cp = alpha_stream_.raw()->codecpar;
            cp->codec_type = AVMEDIA_TYPE_VIDEO;
            cp->codec_id = alpha_codec;
            cp->width = stream_.raw()->codecpar->width;
            cp->height = stream_.raw()->codecpar->height;
            cp->format = alpha_layout.get();
        }
    }

    void init(EdgeManager &edges, const Parameters &params) override {
        NodeSingleOutput<FabricPacket>::init(edges, params);
        auto edge = edges.find<FabricPacket>(params["dst"]);
        auto md = edge->metadata<InputStreamMetadata>(true);
        md->source_stream = stream_;

        if (alpha_enabled_) {
            auto alpha_edge = edges.find<FabricPacket>(params["alpha_dst"]);
            if (params["alpha_dst"].get<std::string>() == params["dst"].get<std::string>()) {
                throw Error("fabric_source alpha_dst must be a different edge from dst");
            }
            alpha_edge->metadata<InputStreamMetadata>(true)->source_stream = alpha_stream_;
            alpha_edge->setProducer(shared_from_this());
            alpha_sink_ = alpha_edge->makeSink();
        }

        const size_t max_msg = sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader) + max_payload_bytes_;
        for (size_t i = 0; i < bind_ports_.size(); ++i) {
            const std::string addr_file = i == 0 ? addr_file_ : std::string();
            receivers_.push_back(std::make_unique<avp_fabric::RdmReceiver>(
                provider_, bind_host_, bind_ports_[i], addr_file, max_msg, rx_buffers_, "fabric_source"));
        }
    }

    // The alpha output is an output like any other: EOF, stop and drain reach it through here.
    void forEachOutput(std::function<void(Sink<FabricPacket> *)> cb) override {
        NodeSingleOutput<FabricPacket>::forEachOutput(cb);
        if (alpha_sink_) cb(alpha_sink_.get());
    }

    void process() override {
        if (stopping_.load(std::memory_order_acquire)) return;
        FabricPacket pkt;
        if (!receiveOne(pkt)) return;
        Sink<FabricPacket> *out = pkt.plane == avp_fabric::PLANE_ALPHA ? alpha_sink_.get() : &this->sink();
        if (out == nullptr) {
            throw Error(std::string("fabric_source received a plane=alpha record but has no alpha_dst output: "
                                    "configure alpha_dst, or stop the producer sending an alpha plane"));
        }
        if (pkt.complete) {
            out->put(pkt);
        }
    }

    size_t streamsCount() override { return 1; }
    av::Stream stream(size_t i) override {
        if (i != 0) throw Error("fabric_source stream index out of range");
        return stream_;
    }
    void discardAllStreams() override {}
    void enableStream(size_t) override {}
    av::FormatContext& formatContext() override { return format_ctx_; }

    void stop() override {
        stopping_.store(true, std::memory_order_release);
        finished_ = true;
        for (auto &receiver: receivers_) {
            receiver->stop();
        }
    }

    void interrupt() override {
        stop();
    }

    static std::shared_ptr<FabricSourceNode> create(NodeCreationInfo &nci) {
        auto edge = nci.edges.find<FabricPacket>(nci.params["dst"]);
        return std::make_shared<FabricSourceNode>(make_unique<EdgeSink<FabricPacket>>(edge), nci.params);
    }

private:
    // Waits on every port at once: its completion queue (where the provider has a descriptor)
    // and its wake eventfd (returned buffers, stop).
    bool nextMessage(size_t &receiver_idx, avp_fabric::RdmReceiver::Message &m) {
        if (receivers_.empty()) throw Error("fabric_source has no receivers");
        while (!stopping_.load(std::memory_order_acquire)) {
            for (size_t i = 0; i < receivers_.size(); ++i) {
                const size_t idx = (next_receiver_ + i) % receivers_.size();
                if (receivers_[idx]->tryNext(m)) {
                    receiver_idx = idx;
                    next_receiver_ = (idx + 1) % receivers_.size();
                    last_message_ns_ = avp_fabric::monotonicNs();
                    return true;
                }
            }
            if (idleExpired()) {
                endSession();
                return false;
            }
            std::vector<pollfd> fds;
            bool can_block = true;
            bool all_fd = true;
            for (auto &r: receivers_) {
                can_block = r->addWaitFds(fds) && can_block;
                all_fd = all_fd && r->hasWaitFd();
            }
            if (!can_block) continue;
            ::poll(fds.data(), fds.size(), all_fd ? pollTimeoutMs() : 1);
            for (auto &r: receivers_) r->consumeWake();
        }
        return false;
    }

    // The idle timer starts at the first record: a source that has never received waits forever,
    // so a consumer may be started before its producer.
    bool idleExpired() const {
        if (idle_timeout_ms_ <= 0 || last_message_ns_ == 0) return false;
        const uint64_t now = avp_fabric::monotonicNs();
        return now - last_message_ns_ >= static_cast<uint64_t>(idle_timeout_ms_) * 1000000ull;
    }

    // Poll timeout bounded by the idle timeout, so the deadline is not overshot by a full poll.
    int pollTimeoutMs() const {
        if (idle_timeout_ms_ <= 0 || last_message_ns_ == 0) return 100;
        const uint64_t now = avp_fabric::monotonicNs();
        const uint64_t deadline = last_message_ns_ + static_cast<uint64_t>(idle_timeout_ms_) * 1000000ull;
        if (now >= deadline) return 0;
        const uint64_t remaining_ms = (deadline - now + 999999ull) / 1000000ull;
        return static_cast<int>(std::min<uint64_t>(remaining_ms, 100));
    }

    // The peer went quiet for idle_timeout_ms: end the session as if the producer had sent EOF,
    // so downstream nodes finish instead of waiting for records that never come. A record that
    // arrives after this cannot be delivered (the node is finished), so a receiver that must
    // survive producer restarts leaves idle_timeout_ms at 0.
    void endSession() {
        if (session_ended_) return;
        session_ended_ = true;
        stopping_.store(true, std::memory_order_release);
        uint64_t peer_closes = 0;
        for (auto &r: receivers_) peer_closes += r->peerCloses();
        logstream << "fabric_source session ended: no record for " << idle_timeout_ms_ << " ms"
                  << " total_frames=" << frames_received_
                  << " total_bytes=" << bytes_received_
                  << " peer_closes=" << peer_closes;
        emitEof();
        finished_ = true;
        for (auto &r: receivers_) r->stop();
    }

    bool receiveOne(FabricPacket &out) {
        size_t receiver_idx = 0;
        avp_fabric::RdmReceiver::Message msg;
        if (!nextMessage(receiver_idx, msg)) return false;
        avp_fabric::RdmReceiver *receiver = receivers_[receiver_idx].get();
        const size_t len = msg.len;
        if (len < sizeof(avp_fabric::WireHeader)) {
            receiver->release(msg);
            throw Error("fabric_source received message shorter than AVP headers");
        }

        const auto *wire = reinterpret_cast<const avp_fabric::WireHeader *>(msg.data);
        avp_fabric::MediaHeader media;
        try {
            avp_fabric::validateWireRecord(*wire, len, max_payload_bytes_, "fabric_source");
            media = avp_fabric::readMediaHeader(msg.data + sizeof(*wire), wire->version);
            avp_fabric::validatePlane(media.plane, "fabric_source");
        } catch (...) {
            receiver->release(msg);
            throw;
        }

        // Everything is read from the buffer before takePayload: under buffer pressure it copies
        // the payload and reposts the buffer, which the provider may then overwrite.
        const uint16_t message_type = wire->message_type;
        const uint16_t header_bytes = wire->header_bytes;
        const uint32_t payload_bytes = wire->payload_bytes;
        out = FabricPacket();
        out.message_type = wire->message_type;
        out.stream_id_hash = media.stream_id_hash;
        out.replica_id = media.replica_id;
        out.generation = media.generation;
        out.raw_pts = media.pts;
        out.raw_dts = media.dts;
        out.time_base_num = media.time_base_num;
        out.time_base_den = media.time_base_den;
        out.media_type = media.media_type;
        out.codec = media.codec;
        out.plane = media.plane;
        out.packet_format = media.packet_format;
        out.packet_flags = media.packet_flags;
        out.duration = media.duration;
        out.width = media.width;
        out.height = media.height;
        out.pixel_format = media.pixel_format;
        out.real_pixel_format = media.real_pixel_format;
        out.sender_wallclock_ns = media.sender_wallclock_ns;
        out.receiver_wallclock_ns = avp_fabric::monotonicNs();
        out.complete = true;

        if (message_type == avp_fabric::MSG_MEDIA) {
            // References the receive buffer: no copy unless receive buffers are running low.
            out.packet = receiver->takePayload(msg, header_bytes, payload_bytes);
            out.packet.setTimeBase(av::Rational(out.time_base_num, out.time_base_den));
            out.packet.setPts(out.pts());
            out.packet.setDts(out.dts());
            out.packet.setDuration(static_cast<int>(out.duration));
            out.packet.setStreamIndex(0);
            out.packet.setKeyPacket((out.packet_flags & avp_fabric::FLAG_KEYFRAME) != 0);
            out.packet.setComplete(true);
        }

        if (frames_received_ < 5) {
            logstream << "fabric_source received"
                      << " type=" << (message_type == avp_fabric::MSG_MEDIA ? "media" : "status")
                      << " plane=" << avp_fabric::planeName(out.plane)
                      << " replica_id=" << out.replica_id
                      << " generation=" << out.generation
                      << " pts=" << out.raw_pts
                      << " message_bytes=" << len
                      << " payload_bytes=" << payload_bytes
                      << " payload_prefix=" << avp_fabric::hexPrefix(out.packet.data(), out.packet.size());
        }
        if (message_type != avp_fabric::MSG_MEDIA) {
            receiver->release(msg);     // nothing downstream references a status message
        }
        frames_received_++;
        bytes_received_ += payload_bytes;
        reportThroughput();
        return true;
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
        logstream << "fabric_source throughput"
                  << " mbps=" << (static_cast<double>(delta_bytes) * 8.0 / seconds / 1000000.0)
                  << " MBps=" << (static_cast<double>(delta_bytes) / seconds / 1000000.0)
                  << " fps=" << (static_cast<double>(delta_frames) / seconds)
                  << " total_frames=" << frames_received_
                  << " total_bytes=" << bytes_received_;
        report_last_ns_ = now;
        report_last_bytes_ = bytes_received_;
        report_last_frames_ = frames_received_;
    }
};

#else

class FabricSourceNode: public NodeSingleOutput<FabricPacket>, public ReportsFinishByFlag {
public:
    using NodeSingleOutput<FabricPacket>::NodeSingleOutput;
    void process() override {
        throw Error("fabric_source requested, but avplumber was built with HAVE_LIBFABRIC=0");
    }
    static std::shared_ptr<FabricSourceNode> create(NodeCreationInfo &nci) {
        auto edge = nci.edges.find<FabricPacket>(nci.params["dst"]);
        return std::make_shared<FabricSourceNode>(make_unique<EdgeSink<FabricPacket>>(edge));
    }
};

#endif

DECLNODE(fabric_source, FabricSourceNode);
