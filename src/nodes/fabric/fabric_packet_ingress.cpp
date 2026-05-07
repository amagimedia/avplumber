#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_protocol.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

class FabricRdmPacketReceiver {
    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    std::vector<uint8_t> buffer_;
    bool recv_posted_ = false;

    static void check(int ret, const std::string &what) {
        if (ret < 0) {
            throw Error(what + " failed: " + fi_strerror(-ret));
        }
    }

public:
    FabricRdmPacketReceiver(const std::string &provider,
                            const std::string &bind_host,
                            const std::string &port,
                            const std::string &addr_file,
                            size_t max_msg_size,
                            size_t queue_depth):
        buffer_(max_msg_size) {
        fi_info *hints = fi_allocinfo();
        if (!hints) throw Error("fi_allocinfo failed");
        hints->caps = FI_MSG;
        hints->ep_attr->type = FI_EP_RDM;
        hints->fabric_attr->prov_name = strdup(provider.c_str());
        hints->rx_attr->size = queue_depth;

        const char *node = bind_host.empty() ? nullptr : bind_host.c_str();
        int ret = fi_getinfo(FI_VERSION(1, 11), node, port.c_str(), FI_SOURCE, hints, &info_);
        fi_freeinfo(hints);
        check(ret, "fi_getinfo");
        if (!info_) throw Error("fi_getinfo returned no provider info");

        check(fi_fabric(info_->fabric_attr, &fabric_, nullptr), "fi_fabric");
        check(fi_domain(fabric_, info_, &domain_, nullptr), "fi_domain");

        fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_MSG;
        cq_attr.size = queue_depth;
        check(fi_cq_open(domain_, &cq_attr, &cq_, nullptr), "fi_cq_open");

        fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        av_attr.count = 1;
        check(fi_av_open(domain_, &av_attr, &av_, nullptr), "fi_av_open");

        check(fi_endpoint(domain_, info_, &ep_, nullptr), "fi_endpoint");
        check(fi_ep_bind(ep_, &cq_->fid, FI_RECV | FI_SEND), "fi_ep_bind cq");
        check(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind av");
        check(fi_enable(ep_), "fi_enable");

        if (!addr_file.empty()) {
            writeLocalAddress(addr_file);
        }

        logstream << "fabric_packet_ingress opened provider=" << provider
                  << " bind_host=" << (bind_host.empty() ? "*" : bind_host)
                  << " port=" << port
                  << " addr_file=" << (addr_file.empty() ? "-" : addr_file)
                  << " max_msg_size=" << max_msg_size;
    }

    ~FabricRdmPacketReceiver() {
        if (ep_) fi_close(&ep_->fid);
        if (av_) fi_close(&av_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
    }

    size_t recvOne(const std::atomic_bool &stop_requested) {
        postRecv(stop_requested);
        if (stop_requested.load()) return 0;
        fi_cq_msg_entry entry = {};
        int ret;
        do {
            ret = fi_cq_read(cq_, &entry, 1);
            if (ret == -FI_EAGAIN) progress();
            if (stop_requested.load()) return 0;
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_cq_read");
        if (ret != 1) throw Error("fi_cq_read returned unexpected count");
        recv_posted_ = false;
        return entry.len;
    }

    const uint8_t *data() const {
        return buffer_.data();
    }

private:
    void postRecv(const std::atomic_bool &stop_requested) {
        if (recv_posted_) return;
        int ret;
        do {
            ret = fi_recv(ep_, buffer_.data(), buffer_.size(), nullptr, FI_ADDR_UNSPEC, nullptr);
            if (ret == -FI_EAGAIN) progress();
            if (stop_requested.load()) return;
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_recv");
        recv_posted_ = true;
    }

    void writeLocalAddress(const std::string &path) {
        size_t len = 0;
        int ret = fi_getname(&ep_->fid, nullptr, &len);
        if (ret != -FI_ETOOSMALL || len == 0) check(ret, "fi_getname size");
        std::vector<char> addr(len);
        check(fi_getname(&ep_->fid, addr.data(), &len), "fi_getname");
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw Error("fabric_packet_ingress cannot open addr_file for writing: " + path);
        f.write(addr.data(), static_cast<std::streamsize>(len));
        if (!f) throw Error("fabric_packet_ingress cannot write addr_file: " + path);
    }

    void progress() {
        fi_cq_read(cq_, nullptr, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

} // namespace

class FabricPacketIngress: public NodeSingleOutput<FabricPacket>,
                                 public ReportsFinishByFlag,
                                 public IStoppable,
                                 public IInterruptible,
                                 public IStreamsInput {
    std::string provider_;
    std::string bind_host_;
    std::string bind_port_;
    std::string addr_file_;
    size_t max_payload_bytes_ = 4 * 1024 * 1024;
    size_t queue_depth_ = 64;
    av::FormatContext format_ctx_;
    av::Stream stream_;
    std::unique_ptr<FabricRdmPacketReceiver> receiver_;
    uint64_t frames_received_ = 0;
    std::atomic_bool stop_requested_{false};

public:
    FabricPacketIngress(std::unique_ptr<Sink<FabricPacket>> &&sink, const Parameters &params):
        NodeSingleOutput<FabricPacket>(std::move(sink)) {
        provider_ = params.value("provider", "shm");
        bind_host_ = params.value("bind_host", std::string());
        if (params.at("bind_port").is_number_integer()) {
            bind_port_ = std::to_string(params.at("bind_port").get<int>());
        } else {
            bind_port_ = params.at("bind_port").get<std::string>();
        }
        addr_file_ = params.value("addr_file", std::string());
        max_payload_bytes_ = params.value("max_payload_bytes", max_payload_bytes_);
        queue_depth_ = params.value("queue_depth", queue_depth_);

        if (params.count("reference_url")) {
            av::Dictionary opts;
            format_ctx_.openInput(params["reference_url"].get<std::string>(), opts);
            format_ctx_.findStreamInfo();
            for (size_t i = 0; i < format_ctx_.streamsCount(); ++i) {
                av::Stream st = format_ctx_.stream(i);
                if (st.isVideo()) {
                    stream_ = st;
                    break;
                }
            }
            if (!stream_.isValid()) throw Error("fabric_packet_ingress reference_url has no video stream");
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
                parseRatio(params["frame_rate"].get<std::string>()) : av::Rational(25, 1);
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
    }

    void init(EdgeManager &edges, const Parameters &params) override {
        NodeSingleOutput<FabricPacket>::init(edges, params);
        auto edge = edges.find<FabricPacket>(params["dst"]);
        auto md = edge->metadata<InputStreamMetadata>(true);
        md->source_stream = stream_;
        const size_t max_msg = sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader) + max_payload_bytes_;
        receiver_ = std::make_unique<FabricRdmPacketReceiver>(
            provider_, bind_host_, bind_port_, addr_file_, max_msg, queue_depth_);
    }

    void process() override {
        for (;;) {
            FabricPacket pkt = receiveOne();
            if (stop_requested_.load()) return;
            if (pkt.complete) {
                this->sink_->put(pkt);
                return;
            }
        }
    }

    size_t streamsCount() override { return 1; }
    av::Stream stream(size_t i) override {
        if (i != 0) throw Error("fabric_packet_ingress stream index out of range");
        return stream_;
    }
    void discardAllStreams() override {}
    void enableStream(size_t) override {}
    av::FormatContext& formatContext() override { return format_ctx_; }

    void stop() override {
        stop_requested_ = true;
        this->finished_ = true;
    }

    void interrupt() override {
        stop();
    }

    static std::shared_ptr<FabricPacketIngress> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto edge = edges.find<FabricPacket>(params["dst"]);
        return std::make_shared<FabricPacketIngress>(make_unique<EdgeSink<FabricPacket>>(edge), params);
    }

private:
    FabricPacket receiveOne() {
        const size_t len = receiver_->recvOne(stop_requested_);
        if (stop_requested_.load() || len == 0) return FabricPacket();
        if (len < sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader)) {
            throw Error("fabric_packet_ingress received message shorter than AVP headers");
        }

        const auto *wire = reinterpret_cast<const avp_fabric::WireHeader *>(receiver_->data());
        const auto *media = reinterpret_cast<const avp_fabric::MediaHeader *>(receiver_->data() + sizeof(avp_fabric::WireHeader));
        avp_fabric::validateWireRecord(*wire, len, max_payload_bytes_, "fabric_packet_ingress");
        if (wire->message_type != avp_fabric::MSG_MEDIA) return FabricPacket();

        const uint8_t *payload = receiver_->data() + wire->header_bytes;
        FabricPacket out;
        out.packet = av::Packet(std::vector<uint8_t>(payload, payload + wire->payload_bytes));
        out.stream_id_hash = media->stream_id_hash;
        out.replica_id = media->replica_id;
        out.generation = media->generation;
        out.raw_pts = media->pts;
        out.raw_dts = media->dts;
        out.time_base_num = media->time_base_num;
        out.time_base_den = media->time_base_den;
        out.media_type = media->media_type;
        out.codec = media->codec;
        out.packet_format = media->packet_format;
        out.packet_flags = media->packet_flags;
        out.duration = media->duration;
        out.width = media->width;
        out.height = media->height;
        out.pixel_format = media->pixel_format;
        out.real_pixel_format = media->real_pixel_format;
        out.sender_wallclock_ns = media->sender_wallclock_ns;
        out.receiver_wallclock_ns = avp_fabric::monotonicNs();
        out.complete = true;
        out.packet.setTimeBase(av::Rational(media->time_base_num, media->time_base_den));
        out.packet.setPts(out.pts());
        out.packet.setDts(out.dts());
        out.packet.setDuration(static_cast<int>(media->duration));
        out.packet.setStreamIndex(0);
        out.packet.setKeyPacket((media->packet_flags & avp_fabric::FLAG_KEYFRAME) != 0);
        out.packet.setComplete(true);

        if (frames_received_ < 5) {
            logstream << "fabric_packet_ingress received"
                      << " replica_id=" << out.replica_id
                      << " pts=" << out.raw_pts
                      << " payload_bytes=" << wire->payload_bytes
                      << " payload_prefix=" << avp_fabric::hexPrefix(payload, wire->payload_bytes);
        }
        frames_received_++;
        return out;
    }
};

#else

class FabricPacketIngress: public NodeSingleOutput<FabricPacket>, public ReportsFinishByFlag {
public:
    using NodeSingleOutput<FabricPacket>::NodeSingleOutput;
    void process() override {
        throw Error("fabric_packet_ingress requested, but avplumber was built with HAVE_LIBFABRIC=0");
    }
    static std::shared_ptr<FabricPacketIngress> create(NodeCreationInfo &nci) {
        auto edge = nci.edges.find<FabricPacket>(nci.params["dst"]);
        return std::make_shared<FabricPacketIngress>(make_unique<EdgeSink<FabricPacket>>(edge));
    }
};

#endif

DECLNODE(fabric_packet_ingress, FabricPacketIngress);
