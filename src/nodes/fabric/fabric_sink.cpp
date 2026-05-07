#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_protocol.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

constexpr uint32_t AVPF_MAGIC = 0x46505641; // "AVPF", little-endian.
constexpr uint16_t AVPF_VERSION = 1;
constexpr uint16_t AVPF_MSG_MEDIA = 1;
constexpr uint16_t AVPF_MSG_FRAME_STATUS = 2;
constexpr uint32_t AVPF_MEDIA_VIDEO = 1;
constexpr uint32_t AVPF_CODEC_H264_INTRA = 1;
constexpr uint32_t AVPF_CODEC_JPEG = 2;
constexpr uint32_t AVPF_PACKET_FORMAT_PASSTHROUGH = 1;
constexpr uint32_t AVPF_FLAG_KEYFRAME = 1u << 0;
constexpr uint32_t AVPF_FLAG_EOF = 1u << 1;

uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c: s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

uint32_t crc32_ieee(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

uint64_t realtimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string hexPrefix(const uint8_t *data, size_t size, size_t max_bytes = 24) {
    std::ostringstream os;
    const size_t n = std::min(size, max_bytes);
    for (size_t i = 0; i < n; ++i) {
        if (i) os << ' ';
        os << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]);
    }
    return os.str();
}

uint32_t codecId(const std::string &codec) {
    if (codec == "h264_intra") return AVPF_CODEC_H264_INTRA;
    if (codec == "jpeg" || codec == "mjpeg" || codec == "nvjpeg") return AVPF_CODEC_JPEG;
    throw Error("fabric_sink unsupported codec: " + codec);
}

uint32_t mediaTypeId(const std::string &media_type) {
    if (media_type == "video") return AVPF_MEDIA_VIDEO;
    throw Error("fabric_sink unsupported media_type: " + media_type);
}

std::vector<char> readBinaryFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("fabric_sink cannot open remote_addr_file: " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size <= 0) throw Error("fabric_sink remote_addr_file is empty: " + path);
    f.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    f.read(bytes.data(), size);
    if (!f) throw Error("fabric_sink failed to read remote_addr_file: " + path);
    return bytes;
}

class FabricRdmSender {
    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    fi_addr_t peer_ = FI_ADDR_UNSPEC;
    size_t max_msg_size_ = 0;

    static void check(int ret, const std::string &what) {
        if (ret < 0) {
            throw Error(what + " failed: " + fi_strerror(-ret));
        }
    }

public:
    FabricRdmSender(const std::string &provider,
                    const std::string &host,
                    const std::string &port,
                    const std::string &remote_addr_file,
                    size_t required_msg_size,
                    size_t queue_depth) {
        fi_info *hints = fi_allocinfo();
        if (!hints) {
            throw Error("fi_allocinfo failed");
        }
        hints->caps = FI_MSG;
        hints->ep_attr->type = FI_EP_RDM;
        hints->fabric_attr->prov_name = strdup(provider.c_str());
        hints->tx_attr->size = queue_depth;
        hints->tx_attr->op_flags = FI_COMPLETION;

        int ret = fi_getinfo(FI_VERSION(1, 11), host.c_str(), port.c_str(), 0, hints, &info_);
        fi_freeinfo(hints);
        check(ret, "fi_getinfo");
        if (!info_) {
            throw Error("fi_getinfo returned no provider info");
        }

        max_msg_size_ = info_->ep_attr ? info_->ep_attr->max_msg_size : 0;
        if (max_msg_size_ && required_msg_size > max_msg_size_) {
            throw Error("fabric_sink max_payload/header size " + std::to_string(required_msg_size) +
                        " exceeds provider max_msg_size " + std::to_string(max_msg_size_));
        }

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
        check(fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV), "fi_ep_bind cq");
        check(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind av");
        check(fi_enable(ep_), "fi_enable");

        std::vector<char> remote_addr;
        const void *dest_addr = info_->dest_addr;
        size_t dest_addrlen = info_->dest_addrlen;
        if (!remote_addr_file.empty()) {
            remote_addr = readBinaryFile(remote_addr_file);
            dest_addr = remote_addr.data();
            dest_addrlen = remote_addr.size();
        }

        if (!dest_addr || dest_addrlen == 0) {
            throw Error("fi_getinfo returned no destination address");
        }
        int inserted = fi_av_insert(av_, dest_addr, 1, &peer_, 0, nullptr);
        if (inserted != 1) {
            if (inserted < 0) {
                throw Error(std::string("fi_av_insert failed: ") + fi_strerror(-inserted));
            }
            throw Error("fi_av_insert inserted " + std::to_string(inserted) + " addresses");
        }

        logstream << "fabric_sink opened provider=" << provider
                  << " host=" << host
                  << " port=" << port
                  << " remote_addr_file=" << (remote_addr_file.empty() ? "-" : remote_addr_file)
                  << " max_msg_size=" << max_msg_size_;
    }

    ~FabricRdmSender() {
        if (ep_) fi_close(&ep_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (av_) fi_close(&av_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
    }

    void send(const void *data, size_t size) {
        int ret;
        do {
            ret = fi_send(ep_, data, size, nullptr, peer_, nullptr);
            if (ret == -FI_EAGAIN) {
                progress();
            }
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_send");
    }

    bool reapOne(bool block) {
        fi_cq_msg_entry entry = {};
        int ret = fi_cq_read(cq_, &entry, 1);
        if (ret == 1) return true;
        if (ret == -FI_EAGAIN && !block) return false;
        while (ret == -FI_EAGAIN && block) {
            Wallclock::sleepms(1);
            ret = fi_cq_read(cq_, &entry, 1);
        }
        if (ret == 1) return true;
        if (ret < 0) {
            fi_cq_err_entry err = {};
            int err_ret = fi_cq_readerr(cq_, &err, 0);
            if (err_ret >= 0) {
                throw Error(std::string("fi_cq_read failed: ") + fi_strerror(err.err));
            }
            throw Error(std::string("fi_cq_read failed: ") + fi_strerror(-ret));
        }
        return false;
    }

    void progress() {
        fi_cq_read(cq_, nullptr, 0);
        Wallclock::sleepms(1);
    }

};

class FabricPacketSink: public NodeSingleInput<av::Packet>, public ReportsFinishByFlag {
    std::string provider_;
    std::string remote_host_;
    std::string remote_port_;
    std::string remote_addr_file_;
    std::string stream_id_;
    uint64_t stream_id_hash_ = 0;
    uint32_t replica_id_ = 0;
    uint64_t generation_ = 0;
    uint32_t media_type_ = AVPF_MEDIA_VIDEO;
    uint32_t codec_ = AVPF_CODEC_H264_INTRA;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    av::PixelFormat pixel_format_{AV_PIX_FMT_NONE};
    av::PixelFormat real_pixel_format_{AV_PIX_FMT_NONE};
    bool require_keyframes_ = true;
    bool header_crc_ = true;
    bool payload_crc_ = false;
    size_t max_payload_bytes_ = 4 * 1024 * 1024;
    size_t queue_depth_ = 64;
    size_t completion_batch_ = 16;
    std::string backpressure_ = "block";
    std::string redundancy_mode_ = "active";
    size_t repair_window_frames_ = 120;
    int control_fd_ = -1;
    uint16_t control_port_ = 0;
    int64_t promote_from_pts_ = INT64_MAX;
    uint64_t frames_sent_ = 0;
    uint64_t bytes_sent_ = 0;
    uint64_t report_start_ns_ = 0;
    uint64_t report_last_ns_ = 0;
    uint64_t report_last_bytes_ = 0;
    uint64_t report_last_frames_ = 0;
    std::unique_ptr<FabricRdmSender> sender_;

    struct SendBuffer {
        std::vector<uint8_t> bytes;
    };
    std::deque<std::unique_ptr<SendBuffer>> free_;
    std::deque<std::unique_ptr<SendBuffer>> inflight_;
    std::deque<int64_t> repair_order_;
    std::unordered_map<int64_t, std::vector<uint8_t>> repair_store_;

public:
    FabricPacketSink(std::unique_ptr<Source<av::Packet>> &&source,
                     const Parameters &params,
                     std::shared_ptr<IVideoFormatSource> video_md):
        NodeSingleInput<av::Packet>(std::move(source)) {
        provider_ = params.value("provider", "tcp");
        remote_host_ = params.at("remote_host").get<std::string>();
        if (params.at("remote_port").is_number_integer()) {
            remote_port_ = std::to_string(params.at("remote_port").get<int>());
        } else {
            remote_port_ = params.at("remote_port").get<std::string>();
        }
        remote_addr_file_ = params.value("remote_addr_file", std::string());
        stream_id_ = params.at("stream_id").get<std::string>();
        stream_id_hash_ = fnv1a64(stream_id_);
        replica_id_ = params.at("replica_id").get<uint32_t>();
        generation_ = params.value("generation", 0ull);
        if (generation_ == 0) {
            generation_ = realtimeNs() ^ (static_cast<uint64_t>(replica_id_) << 32);
            logstream << "fabric_sink generated session generation=" << generation_
                      << " replica_id=" << replica_id_;
        }
        media_type_ = mediaTypeId(params.value("media_type", "video"));
        codec_ = codecId(params.value("codec", "h264_intra"));
        require_keyframes_ = params.value("require_keyframes", true);
        header_crc_ = params.value("header_crc", true);
        payload_crc_ = params.value("payload_crc", false);
        max_payload_bytes_ = params.value("max_payload_bytes", max_payload_bytes_);
        queue_depth_ = params.value("queue_depth", queue_depth_);
        completion_batch_ = params.value("completion_batch", completion_batch_);
        backpressure_ = params.value("backpressure", backpressure_);
        redundancy_mode_ = params.value("redundancy_mode", params.value("mode", redundancy_mode_));
        repair_window_frames_ = params.value("repair_window_frames", repair_window_frames_);
        control_port_ = params.value("control_port", 0);
        if (backpressure_ != "block" && backpressure_ != "fail") {
            throw Error("fabric_sink unsupported backpressure policy for MVP: " + backpressure_);
        }
        if (redundancy_mode_ != "active" && redundancy_mode_ != "standby" && redundancy_mode_ != "hot_hot") {
            throw Error("fabric_sink unsupported redundancy_mode: " + redundancy_mode_);
        }
        if (video_md) {
            width_ = video_md->width();
            height_ = video_md->height();
            pixel_format_ = video_md->pixelFormat();
            real_pixel_format_ = video_md->realPixelFormat();
        }
        if (params.count("width")) width_ = params["width"];
        if (params.count("height")) height_ = params["height"];
        if (params.count("pixel_format")) pixel_format_ = av::PixelFormat(params["pixel_format"].get<std::string>());
        if (params.count("real_pixel_format")) real_pixel_format_ = av::PixelFormat(params["real_pixel_format"].get<std::string>());
        if (!width_ || !height_) {
            throw Error("fabric_sink requires width/height parameters or upstream IVideoFormatSource");
        }
        if (pixel_format_ == AV_PIX_FMT_NONE) {
            throw Error("fabric_sink requires pixel_format parameter or upstream IVideoFormatSource");
        }

        const size_t max_msg = sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader) + max_payload_bytes_;
        for (size_t i = 0; i < queue_depth_; ++i) {
            auto b = std::make_unique<SendBuffer>();
            b->bytes.reserve(max_msg);
            free_.push_back(std::move(b));
        }
        sender_ = std::make_unique<FabricRdmSender>(provider_, remote_host_, remote_port_, remote_addr_file_, max_msg, queue_depth_);
        if (control_port_) openControlSocket();
    }

    virtual void process() override {
        pollControl();
        for (size_t i = 0; i < completion_batch_ && !inflight_.empty(); ++i) {
            if (!sender_->reapOne(false)) break;
            free_.push_back(std::move(inflight_.front()));
            inflight_.pop_front();
        }

        av::Packet pkt = this->source_->get();
        if (isEofMarker(pkt)) {
            this->finished_ = true;
            return;
        }
        if (!pkt.isComplete()) return;
        if (!pkt.pts().isValid()) {
            throw Error("fabric_sink packet PTS is invalid");
        }
        if (pkt.size() > max_payload_bytes_) {
            throw Error("fabric_sink payload " + std::to_string(pkt.size()) +
                        " exceeds max_payload_bytes " + std::to_string(max_payload_bytes_));
        }
        if (require_keyframes_ && !pkt.isKeyPacket()) {
            throw Error("fabric_sink require_keyframes=true but packet is not keyframe");
        }

        while (free_.empty()) {
            if (backpressure_ == "fail") {
                throw Error("fabric_sink send buffer pool exhausted");
            }
            sender_->reapOne(true);
            free_.push_back(std::move(inflight_.front()));
            inflight_.pop_front();
        }

        auto buf = std::move(free_.front());
        free_.pop_front();
        const int64_t pts = fillBuffer(*buf, pkt, AVPF_MSG_MEDIA);
        storeRepairFrame(pts, buf->bytes);
        const bool send_media = redundancy_mode_ == "active" ||
                                redundancy_mode_ == "hot_hot" ||
                                pts >= promote_from_pts_;
        if (!send_media) {
            fillBuffer(*buf, pkt, AVPF_MSG_FRAME_STATUS);
        }
        if (frames_sent_ < 5) {
            logstream << "fabric_sink sending"
                      << " pts=" << pts
                      << " type=" << (send_media ? "media" : "status")
                      << " message_bytes=" << buf->bytes.size()
                      << " payload_bytes=" << (send_media ? pkt.size() : 0)
                      << " payload_prefix=" << hexPrefix(pkt.data(), pkt.size());
        }
        sender_->send(buf->bytes.data(), buf->bytes.size());
        if (frames_sent_ < 5) {
            logstream << "fabric_sink posted"
                      << " message_bytes=" << buf->bytes.size();
        }
        inflight_.push_back(std::move(buf));
        frames_sent_++;
        if (send_media) bytes_sent_ += pkt.size();
        reportThroughput();
    }

    virtual void onEofConsumed() override {
        this->finished_ = true;
    }

    virtual ~FabricPacketSink() {
        if (control_fd_ >= 0) close(control_fd_);
        try {
            while (!inflight_.empty()) {
                sender_->reapOne(true);
                inflight_.pop_front();
            }
            logstream << "fabric_sink summary"
                      << " stream_id=" << stream_id_
                      << " replica_id=" << replica_id_
                      << " generation=" << generation_
                      << " frames_sent=" << frames_sent_
                      << " bytes_sent=" << bytes_sent_;
        } catch (std::exception &e) {
            logstream << "fabric_sink shutdown warning: " << e.what();
        }
    }

    static std::shared_ptr<FabricPacketSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["src"]);
        std::shared_ptr<IVideoFormatSource> video_md = edge->findNodeUp<IVideoFormatSource>();
        return std::make_shared<FabricPacketSink>(make_unique<EdgeSource<av::Packet>>(edge), params, video_md);
    }

private:
    int64_t fillBuffer(SendBuffer &buf, const av::Packet &pkt, uint16_t message_type) {
        avp_fabric::WireHeader wire = {};
        avp_fabric::MediaHeader media = {};
        wire.magic = AVPF_MAGIC;
        wire.version = 1;
        wire.header_bytes = sizeof(wire) + sizeof(media);
        wire.message_type = message_type;
        wire.flags = 0;
        wire.payload_bytes = message_type == AVPF_MSG_MEDIA ? pkt.size() : 0;
        media.stream_id_hash = stream_id_hash_;
        media.replica_id = replica_id_;
        media.generation = generation_;
        media.pts = pkt.pts().timestamp();
        media.dts = pkt.dts().isValid() ? pkt.dts().timestamp() : AV_NOPTS_VALUE;
        media.time_base_num = pkt.pts().timebase().getNumerator();
        media.time_base_den = pkt.pts().timebase().getDenominator();
        media.media_type = media_type_;
        media.codec = codec_;
        media.packet_format = AVPF_PACKET_FORMAT_PASSTHROUGH;
        media.packet_flags = pkt.isKeyPacket() ? AVPF_FLAG_KEYFRAME : 0;
        media.duration = pkt.duration();
        media.width = width_;
        media.height = height_;
        media.pixel_format = pixel_format_.get();
        media.real_pixel_format = real_pixel_format_.get();
        media.sender_wallclock_ns = realtimeNs();
        if (payload_crc_ && wire.payload_bytes) {
            wire.payload_crc = crc32_ieee(pkt.data(), pkt.size());
        }

        buf.bytes.resize(sizeof(wire) + sizeof(media) + wire.payload_bytes);
        std::memcpy(buf.bytes.data(), &wire, sizeof(wire));
        std::memcpy(buf.bytes.data() + sizeof(wire), &media, sizeof(media));
        if (wire.payload_bytes) {
            std::memcpy(buf.bytes.data() + sizeof(wire) + sizeof(media), pkt.data(), pkt.size());
        }
        if (header_crc_) {
            auto *wirep = reinterpret_cast<avp_fabric::WireHeader *>(buf.bytes.data());
            wirep->header_crc = 0;
            wirep->header_crc = crc32_ieee(buf.bytes.data(), sizeof(wire) + sizeof(media));
        }
        return media.pts;
    }

    void storeRepairFrame(int64_t pts, const std::vector<uint8_t> &bytes) {
        if (!repair_window_frames_) return;
        if (!repair_store_.count(pts)) repair_order_.push_back(pts);
        repair_store_[pts] = bytes;
        while (repair_order_.size() > repair_window_frames_) {
            repair_store_.erase(repair_order_.front());
            repair_order_.pop_front();
        }
    }

    void openControlSocket() {
        control_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (control_fd_ < 0) throw Error("fabric_sink control socket failed");
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(control_port_);
        if (bind(control_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            throw Error("fabric_sink control bind failed on port " + std::to_string(control_port_));
        }
        int flags = fcntl(control_fd_, F_GETFL, 0);
        fcntl(control_fd_, F_SETFL, flags | O_NONBLOCK);
        logstream << "fabric_sink control listening udp_port=" << control_port_;
    }

    void sendStoredFrame(int64_t pts) {
        auto it = repair_store_.find(pts);
        if (it == repair_store_.end()) {
            logstream << "fabric_sink repair miss pts=" << pts
                      << " replica_id=" << replica_id_;
            return;
        }
        while (free_.empty()) {
            if (inflight_.empty()) return;
            sender_->reapOne(true);
            free_.push_back(std::move(inflight_.front()));
            inflight_.pop_front();
        }
        auto buf = std::move(free_.front());
        free_.pop_front();
        buf->bytes = it->second;
        sender_->send(buf->bytes.data(), buf->bytes.size());
        inflight_.push_back(std::move(buf));
        logstream << "fabric_sink repair sent pts=" << pts
                  << " replica_id=" << replica_id_;
    }

    void pollControl() {
        if (control_fd_ < 0) return;
        char msg[256];
        for (;;) {
            const ssize_t n = recv(control_fd_, msg, sizeof(msg) - 1, 0);
            if (n <= 0) break;
            msg[n] = 0;
            std::istringstream is(std::string(msg, static_cast<size_t>(n)));
            std::string cmd;
            int64_t pts = 0;
            is >> cmd >> pts;
            if (cmd == "REPAIR") {
                sendStoredFrame(pts);
            } else if (cmd == "PROMOTE") {
                promote_from_pts_ = std::min(promote_from_pts_, pts);
                logstream << "fabric_sink promoted from pts=" << promote_from_pts_
                          << " replica_id=" << replica_id_;
            }
        }
    }

    void reportThroughput() {
        const uint64_t now = realtimeNs();
        if (!report_start_ns_) {
            report_start_ns_ = now;
            report_last_ns_ = now;
            report_last_bytes_ = bytes_sent_;
            report_last_frames_ = frames_sent_;
            return;
        }
        const uint64_t elapsed_ns = now - report_last_ns_;
        if (elapsed_ns < 1000000000ull) return;

        const uint64_t delta_bytes = bytes_sent_ - report_last_bytes_;
        const uint64_t delta_frames = frames_sent_ - report_last_frames_;
        const double seconds = static_cast<double>(elapsed_ns) / 1000000000.0;
        const double mbps = static_cast<double>(delta_bytes) * 8.0 / seconds / 1000000.0;
        const double mbytes = static_cast<double>(delta_bytes) / seconds / 1000000.0;
        const double fps = static_cast<double>(delta_frames) / seconds;
        logstream << "fabric_sink throughput"
                  << " stream_id=" << stream_id_
                  << " replica_id=" << replica_id_
                  << " mbps=" << mbps
                  << " MBps=" << mbytes
                  << " fps=" << fps
                  << " total_frames=" << frames_sent_
                  << " total_bytes=" << bytes_sent_;
        report_last_ns_ = now;
        report_last_bytes_ = bytes_sent_;
        report_last_frames_ = frames_sent_;
    }
};

#else

class FabricPacketSink: public NodeSingleInput<av::Packet>, public ReportsFinishByFlag {
public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    virtual void process() override {
        throw Error("fabric_sink requested, but avplumber was built with HAVE_LIBFABRIC=0");
    }
    static std::shared_ptr<FabricPacketSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["src"]);
        return std::make_shared<FabricPacketSink>(make_unique<EdgeSource<av::Packet>>(edge));
    }
};
#endif

DECLNODE(fabric_sink, FabricPacketSink);
