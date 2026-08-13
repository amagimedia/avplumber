#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t AVPF_MAGIC = 0x46505641;

struct __attribute__((packed)) AvpFabricWireHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint16_t message_type;
    uint16_t flags;
    uint32_t header_crc;
    uint32_t payload_crc;
    uint32_t payload_bytes;
};

struct __attribute__((packed)) AvpFabricMediaHeader {
    uint64_t stream_id_hash;
    uint32_t track_id;
    uint32_t replica_id;
    uint64_t generation;
    uint64_t sequence;
    uint64_t frame_group_id;
    int64_t pts;
    int64_t dts;
    int32_t time_base_num;
    int32_t time_base_den;
    uint32_t media_type;
    uint32_t codec;
    uint32_t packet_format;
    uint32_t packet_flags;
    int64_t duration;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t real_pixel_format;
    uint64_t sender_wallclock_ns;
};

[[noreturn]] void fail(const std::string &msg, int err = 0) {
    if (err < 0) {
        std::cerr << msg << ": " << fi_strerror(-err) << " (" << err << ")\n";
    } else {
        std::cerr << msg << "\n";
    }
    std::exit(1);
}

void check(int ret, const std::string &what) {
    if (ret < 0) fail(what, ret);
}

class RdmReceiver {
    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    std::vector<uint8_t> buffer_;

public:
    RdmReceiver(const std::string &provider, const std::string &bind_host, const std::string &port, size_t max_msg_size, const std::string &addr_file):
        buffer_(max_msg_size) {
        fi_info *hints = fi_allocinfo();
        if (!hints) fail("fi_allocinfo failed");
        hints->caps = FI_MSG;
        hints->ep_attr->type = FI_EP_RDM;
        hints->fabric_attr->prov_name = strdup(provider.c_str());

        const char *node = bind_host.empty() ? nullptr : bind_host.c_str();
        int ret = fi_getinfo(FI_VERSION(1, 11), node, port.c_str(), FI_SOURCE, hints, &info_);
        fi_freeinfo(hints);
        check(ret, "fi_getinfo");
        if (!info_) fail("fi_getinfo returned no provider info");

        check(fi_fabric(info_->fabric_attr, &fabric_, nullptr), "fi_fabric");
        check(fi_domain(fabric_, info_, &domain_, nullptr), "fi_domain");

        fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_MSG;
        cq_attr.size = 128;
        check(fi_cq_open(domain_, &cq_attr, &cq_, nullptr), "fi_cq_open");

        fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        av_attr.count = 1;
        check(fi_av_open(domain_, &av_attr, &av_, nullptr), "fi_av_open");

        check(fi_endpoint(domain_, info_, &ep_, nullptr), "fi_endpoint");
        check(fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV), "fi_ep_bind cq");
        check(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind av");
        check(fi_enable(ep_), "fi_enable");
        if (!addr_file.empty()) {
            writeLocalAddress(addr_file);
        }

        std::cout << "receiver_open provider=" << provider
                  << " bind_host=" << (bind_host.empty() ? "*" : bind_host)
                  << " port=" << port
                  << " addr_file=" << (addr_file.empty() ? "-" : addr_file)
                  << " max_msg_size=" << max_msg_size
                  << " fi_max_msg_size=" << (info_->ep_attr ? info_->ep_attr->max_msg_size : 0)
                  << "\n";
    }

    ~RdmReceiver() {
        if (ep_) fi_close(&ep_->fid);
        if (av_) fi_close(&av_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
    }

    size_t recvOne() {
        int ret;
        do {
            ret = fi_recv(ep_, buffer_.data(), buffer_.size(), nullptr, FI_ADDR_UNSPEC, nullptr);
            if (ret == -FI_EAGAIN) progress();
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_recv");

        fi_cq_msg_entry entry = {};
        do {
            ret = fi_cq_read(cq_, &entry, 1);
            if (ret == -FI_EAGAIN) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_cq_read");
        if (ret != 1) fail("fi_cq_read returned unexpected count");
        return entry.len;
    }

    void describe(size_t len, size_t index) {
        if (len < sizeof(AvpFabricWireHeader) + sizeof(AvpFabricMediaHeader)) {
            std::cout << "frame=" << index << " len=" << len << " too_short=1\n";
            return;
        }
        const auto *wire = reinterpret_cast<const AvpFabricWireHeader *>(buffer_.data());
        const auto *media = reinterpret_cast<const AvpFabricMediaHeader *>(buffer_.data() + sizeof(AvpFabricWireHeader));
        const bool magic_ok = wire->magic == AVPF_MAGIC;
        std::cout << "frame=" << index
                  << " len=" << len
                  << " magic_ok=" << magic_ok
                  << " payload_bytes=" << wire->payload_bytes
                  << " sequence=" << media->sequence
                  << " frame_group_id=" << media->frame_group_id
                  << " pts=" << media->pts
                  << " tb=" << media->time_base_num << "/" << media->time_base_den
                  << " replica_id=" << media->replica_id
                  << " generation=" << media->generation
                  << " size=" << media->width << "x" << media->height
                  << " pixfmt=" << media->pixel_format
                  << " real_pixfmt=" << media->real_pixel_format
                  << "\n";
    }

private:
    void writeLocalAddress(const std::string &path) {
        size_t len = 0;
        int ret = fi_getname(&ep_->fid, nullptr, &len);
        if (ret != -FI_ETOOSMALL || len == 0) check(ret, "fi_getname size");
        std::vector<char> addr(len);
        check(fi_getname(&ep_->fid, addr.data(), &len), "fi_getname");
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) fail("cannot open addr file for writing: " + path);
        f.write(addr.data(), static_cast<std::streamsize>(len));
        if (!f) fail("cannot write addr file: " + path);
    }

    void progress() {
        fi_cq_read(cq_, nullptr, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

}

int main(int argc, char **argv) {
    std::string provider = "tcp";
    std::string bind_host;
    std::string port = "5555";
    std::string addr_file;
    size_t max_frames = 60;
    size_t max_msg_size = 4 * 1024 * 1024 + 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--provider" && i + 1 < argc) provider = argv[++i];
        else if (arg == "--bind-host" && i + 1 < argc) bind_host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = argv[++i];
        else if (arg == "--addr-file" && i + 1 < argc) addr_file = argv[++i];
        else if (arg == "--frames" && i + 1 < argc) max_frames = std::stoull(argv[++i]);
        else if (arg == "--max-msg-size" && i + 1 < argc) max_msg_size = std::stoull(argv[++i]);
        else fail("usage: fabric_dump_receiver [--provider tcp|shm] [--bind-host HOST] [--port PORT] [--addr-file PATH] [--frames N] [--max-msg-size BYTES]");
    }

    RdmReceiver receiver(provider, bind_host, port, max_msg_size, addr_file);
    for (size_t i = 0; i < max_frames; ++i) {
        size_t len = receiver.recvOne();
        receiver.describe(len, i);
    }
    std::cout << "receiver_summary frames=" << max_frames << "\n";
    return 0;
}
