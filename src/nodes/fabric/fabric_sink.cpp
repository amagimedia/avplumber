#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_protocol.hpp"
#include "fabric_wait.hpp"

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
#include <sys/uio.h>
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

namespace {

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

} // namespace

// One message in flight: its header, a reference to the payload's packet (held until the
// provider reports completion), and a contiguous copy for providers without scatter-gather.
struct TxOp {
    fi_context2 ctx;            // first member: the completion's op_context is this op
    uint8_t header[sizeof(avp_fabric::WireHeader) + sizeof(avp_fabric::MediaHeader)];
    av::Packet payload;
    std::vector<uint8_t> copy;
    size_t message_bytes = 0;

    size_t payloadBytes() const {
        return message_bytes - sizeof(header);
    }
    void clear() {
        payload = av::Packet();
        message_bytes = 0;
    }
};

class FabricRdmSender {
    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    avp_fabric::CqWaiter waiter_;
    fi_addr_t peer_ = FI_ADDR_UNSPEC;
    size_t max_msg_size_ = 0;
    size_t iov_limit_ = 1;

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
        hints->mode = FI_CONTEXT | FI_CONTEXT2;
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
        iov_limit_ = info_->tx_attr && info_->tx_attr->iov_limit ? info_->tx_attr->iov_limit : 1;

        check(fi_fabric(info_->fabric_attr, &fabric_, nullptr), "fi_fabric");
        check(fi_domain(fabric_, info_, &domain_, nullptr), "fi_domain");

        fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_MSG;
        cq_attr.size = queue_depth;
        bool has_fd = false;
        cq_ = avp_fabric::CqWaiter::open(domain_, cq_attr, has_fd);

        fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        av_attr.count = 1;
        check(fi_av_open(domain_, &av_attr, &av_, nullptr), "fi_av_open");

        check(fi_endpoint(domain_, info_, &ep_, nullptr), "fi_endpoint");
        check(fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV), "fi_ep_bind cq");
        check(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind av");
        check(fi_enable(ep_), "fi_enable");
        waiter_.attach(fabric_, cq_, has_fd);

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

        logstream << "fabric_sink opened provider=" << info_->fabric_attr->prov_name
                  << " fabric_version=" << FI_MAJOR(fi_version()) << "." << FI_MINOR(fi_version())
                  << " host=" << host
                  << " port=" << port
                  << " remote_addr_file=" << (remote_addr_file.empty() ? "-" : remote_addr_file)
                  << " max_msg_size=" << max_msg_size_
                  << " iov_limit=" << iov_limit_
                  << " wait=" << (has_fd ? "fd" : "poll");
    }

    ~FabricRdmSender() {
        if (ep_) fi_close(&ep_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (av_) fi_close(&av_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
    }

    bool scatterGather() const {
        return iov_limit_ >= 2;
    }

    // Posts one message. Returns false when the provider has no room yet (-FI_EAGAIN).
    bool tryPost(TxOp &op) {
        int ret;
        if (scatterGather() && op.payload.size()) {
            iovec iov[2] = {
                {op.header, sizeof(op.header)},
                {const_cast<uint8_t *>(op.payload.data()), op.payload.size()},
            };
            ret = fi_sendv(ep_, iov, nullptr, 2, peer_, &op.ctx);
        } else if (op.payload.size()) {
            op.copy.resize(op.message_bytes);
            std::memcpy(op.copy.data(), op.header, sizeof(op.header));
            std::memcpy(op.copy.data() + sizeof(op.header), op.payload.data(), op.payload.size());
            ret = fi_send(ep_, op.copy.data(), op.copy.size(), nullptr, peer_, &op.ctx);
        } else {
            ret = fi_send(ep_, op.header, sizeof(op.header), nullptr, peer_, &op.ctx);
        }
        if (ret == -FI_EAGAIN) return false;
        check(ret, "fi_send");
        return true;
    }

    // Reads completions and hands each finished op to done(). With timeout_ms > 0 and nothing
    // ready, waits for one at most that long. A failed send throws after its op is returned.
    template<typename Done> size_t reap(Done &&done, int timeout_ms) {
        size_t n = readCompletions(done);
        if (n || timeout_ms <= 0) return n;
        waiter_.wait(timeout_ms);
        return readCompletions(done);
    }

private:
    template<typename Done> size_t readCompletions(Done &done) {
        size_t total = 0;
        fi_cq_msg_entry entries[16];
        for (;;) {
            int ret = fi_cq_read(cq_, entries, 16);
            if (ret == -FI_EAGAIN) return total;
            if (ret == -FI_EAVAIL) {
                fi_cq_err_entry err = {};
                int err_ret = fi_cq_readerr(cq_, &err, 0);
                if (err_ret < 0) check(err_ret, "fi_cq_readerr");
                if (err.op_context) done(static_cast<TxOp *>(err.op_context));
                throw Error(std::string("fabric_sink send failed: ") + fi_strerror(err.err) + " (" +
                            fi_cq_strerror(cq_, err.prov_errno, err.err_data, nullptr, 0) + ")");
            }
            check(ret, "fi_cq_read");
            for (int i = 0; i < ret; ++i) done(static_cast<TxOp *>(entries[i].op_context));
            total += ret;
            if (ret < 16) return total;
        }
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
    std::string backpressure_ = "block";
    std::string redundancy_mode_ = "active";
    size_t repair_window_frames_ = 120;
    int send_timeout_ms_ = 2000;
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

    std::vector<std::unique_ptr<TxOp>> ops_;
    std::vector<TxOp *> free_;
    size_t inflight_ = 0;

    // A frame kept for REPAIR: its MEDIA header and a reference to its packet, not a copy.
    struct RepairFrame {
        uint8_t header[sizeof(TxOp::header)];
        av::Packet payload;
    };
    std::deque<int64_t> repair_order_;
    std::unordered_map<int64_t, RepairFrame> repair_store_;

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
        queue_depth_ = std::max<size_t>(params.value("queue_depth", queue_depth_), 1);
        backpressure_ = params.value("backpressure", backpressure_);
        redundancy_mode_ = params.value("redundancy_mode", params.value("mode", redundancy_mode_));
        repair_window_frames_ = params.value("repair_window_frames", repair_window_frames_);
        send_timeout_ms_ = params.value("send_timeout_ms", send_timeout_ms_);
        control_port_ = params.value("control_port", 0);
        if (backpressure_ != "block" && backpressure_ != "fail") {
            throw Error("fabric_sink unsupported backpressure policy for MVP: " + backpressure_);
        }
        if (redundancy_mode_ != "active" && redundancy_mode_ != "standby" && redundancy_mode_ != "hot_hot") {
            throw Error("fabric_sink unsupported redundancy_mode: " + redundancy_mode_);
        }
        if (send_timeout_ms_ <= 0) {
            throw Error("fabric_sink send_timeout_ms must be positive");
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

        const size_t max_msg = sizeof(TxOp::header) + max_payload_bytes_;
        for (size_t i = 0; i < queue_depth_; ++i) {
            ops_.push_back(std::make_unique<TxOp>());
            free_.push_back(ops_.back().get());
        }
        sender_ = std::make_unique<FabricRdmSender>(provider_, remote_host_, remote_port_, remote_addr_file_, max_msg, queue_depth_);
        if (control_port_) openControlSocket();
    }

    virtual void process() override {
        pollControl();
        reap(0);

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

        if (free_.empty() && backpressure_ == "fail") {
            reap(0);
            if (free_.empty()) throw Error("fabric_sink send buffer pool exhausted");
        }
        TxOp *op = acquireOp();
        const int64_t pts = fillHeader(op->header, pkt, AVPF_MSG_MEDIA);
        storeRepairFrame(pts, op->header, pkt);
        const bool send_media = redundancy_mode_ == "active" ||
                                redundancy_mode_ == "hot_hot" ||
                                pts >= promote_from_pts_;
        if (send_media) {
            op->payload = pkt;
            op->message_bytes = sizeof(op->header) + pkt.size();
        } else {
            fillHeader(op->header, pkt, AVPF_MSG_FRAME_STATUS);
            op->message_bytes = sizeof(op->header);
        }
        if (frames_sent_ < 5) {
            logstream << "fabric_sink sending"
                      << " pts=" << pts
                      << " type=" << (send_media ? "media" : "status")
                      << " message_bytes=" << op->message_bytes
                      << " payload_bytes=" << (send_media ? pkt.size() : 0)
                      << " payload_prefix=" << hexPrefix(pkt.data(), pkt.size());
        }
        post(op);
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
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(send_timeout_ms_);
            while (inflight_ && avp_fabric::remainingMs(deadline) > 0) {
                reap(static_cast<int>(std::min<int64_t>(avp_fabric::remainingMs(deadline), 100)));
            }
            logstream << "fabric_sink summary"
                      << " stream_id=" << stream_id_
                      << " replica_id=" << replica_id_
                      << " generation=" << generation_
                      << " frames_sent=" << frames_sent_
                      << " bytes_sent=" << bytes_sent_
                      << " unfinished_sends=" << inflight_;
        } catch (std::exception &e) {
            logstream << "fabric_sink shutdown warning: " << e.what();
        }
        // Closing the endpoint cancels what is still in flight before the ops are freed.
        sender_.reset();
    }

    static std::shared_ptr<FabricPacketSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["src"]);
        std::shared_ptr<IVideoFormatSource> video_md = edge->findNodeUp<IVideoFormatSource>();
        return std::make_shared<FabricPacketSink>(make_unique<EdgeSource<av::Packet>>(edge), params, video_md);
    }

private:
    void release(TxOp *op) {
        op->clear();
        free_.push_back(op);
        --inflight_;
    }

    void reap(int timeout_ms) {
        sender_->reap([this](TxOp *op) { release(op); }, timeout_ms);
    }

    // A free op; with none free, waits for completions up to send_timeout_ms.
    TxOp *acquireOp() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(send_timeout_ms_);
        while (free_.empty()) {
            const int64_t left = avp_fabric::remainingMs(deadline);
            if (left == 0) {
                throw Error("fabric_sink timed out after " + std::to_string(send_timeout_ms_) +
                            " ms waiting for send completions");
            }
            reap(static_cast<int>(std::min<int64_t>(left, 100)));
        }
        TxOp *op = free_.back();
        free_.pop_back();
        return op;
    }

    void post(TxOp *op) {
        ++inflight_;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(send_timeout_ms_);
        while (!sender_->tryPost(*op)) {
            const int64_t left = avp_fabric::remainingMs(deadline);
            if (left == 0) {
                release(op);
                throw Error("fabric_sink timed out after " + std::to_string(send_timeout_ms_) +
                            " ms posting a send");
            }
            reap(static_cast<int>(std::min<int64_t>(left, 10)));
        }
    }

    int64_t fillHeader(uint8_t *dst, const av::Packet &pkt, uint16_t message_type) {
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

        std::memcpy(dst, &wire, sizeof(wire));
        std::memcpy(dst + sizeof(wire), &media, sizeof(media));
        if (header_crc_) {
            auto *wirep = reinterpret_cast<avp_fabric::WireHeader *>(dst);
            wirep->header_crc = 0;
            wirep->header_crc = crc32_ieee(dst, sizeof(wire) + sizeof(media));
        }
        return media.pts;
    }

    void storeRepairFrame(int64_t pts, const uint8_t *media_header, const av::Packet &pkt) {
        if (!repair_window_frames_) return;
        if (!repair_store_.count(pts)) repair_order_.push_back(pts);
        RepairFrame &frame = repair_store_[pts];
        std::memcpy(frame.header, media_header, sizeof(frame.header));
        frame.payload = pkt;
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
        TxOp *op = acquireOp();
        std::memcpy(op->header, it->second.header, sizeof(op->header));
        op->payload = it->second.payload;
        op->message_bytes = sizeof(op->header) + op->payload.size();
        post(op);
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
