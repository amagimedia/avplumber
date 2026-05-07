#pragma once

#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace avp_fabric {

class RdmReceiver {
    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    std::vector<uint8_t> buffer_;
    bool recv_posted_ = false;
    std::atomic_bool stopping_{false};
    std::string log_prefix_;

    static void check(int ret, const std::string &what) {
        if (ret < 0) {
            throw Error(what + " failed: " + fi_strerror(-ret));
        }
    }

public:
    RdmReceiver(const std::string &provider,
                const std::string &bind_host,
                const std::string &port,
                const std::string &addr_file,
                size_t max_msg_size,
                size_t queue_depth,
                std::string log_prefix):
        buffer_(max_msg_size),
        log_prefix_(std::move(log_prefix)) {
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

        logstream << log_prefix_ << " opened provider=" << provider
                  << " bind_host=" << (bind_host.empty() ? "*" : bind_host)
                  << " port=" << port
                  << " addr_file=" << (addr_file.empty() ? "-" : addr_file)
                  << " max_msg_size=" << max_msg_size
                  << " fi_max_msg_size=" << (info_->ep_attr ? info_->ep_attr->max_msg_size : 0);
    }

    ~RdmReceiver() {
        if (ep_) fi_close(&ep_->fid);
        if (av_) fi_close(&av_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
    }

    void stop() {
        stopping_.store(true, std::memory_order_release);
    }

    bool recvOne(size_t &len) {
        if (!postRecv()) return false;
        fi_cq_msg_entry entry = {};
        int ret;
        do {
            if (stopping_.load(std::memory_order_acquire)) return false;
            ret = fi_cq_read(cq_, &entry, 1);
            if (ret == -FI_EAGAIN) {
                progress();
            }
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_cq_read");
        if (ret != 1) throw Error("fi_cq_read returned unexpected count");
        recv_posted_ = false;
        len = entry.len;
        return true;
    }

    bool tryRecvOne(size_t &len) {
        if (!postRecv()) return false;
        fi_cq_msg_entry entry = {};
        int ret = fi_cq_read(cq_, &entry, 1);
        if (ret == -FI_EAGAIN) return false;
        check(ret, "fi_cq_read");
        if (ret != 1) throw Error("fi_cq_read returned unexpected count");
        recv_posted_ = false;
        len = entry.len;
        return true;
    }

    const uint8_t *data() const {
        return buffer_.data();
    }

private:
    bool postRecv() {
        if (recv_posted_) return true;
        int ret;
        do {
            if (stopping_.load(std::memory_order_acquire)) return false;
            ret = fi_recv(ep_, buffer_.data(), buffer_.size(), nullptr, FI_ADDR_UNSPEC, nullptr);
            if (ret == -FI_EAGAIN) progress();
        } while (ret == -FI_EAGAIN);
        check(ret, "fi_recv");
        recv_posted_ = true;
        return true;
    }

    void writeLocalAddress(const std::string &path) {
        size_t len = 0;
        int ret = fi_getname(&ep_->fid, nullptr, &len);
        if (ret != -FI_ETOOSMALL || len == 0) check(ret, "fi_getname size");
        std::vector<char> addr(len);
        check(fi_getname(&ep_->fid, addr.data(), &len), "fi_getname");
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw Error(log_prefix_ + " cannot open addr_file for writing: " + path);
        f.write(addr.data(), static_cast<std::streamsize>(len));
        if (!f) throw Error(log_prefix_ + " cannot write addr_file: " + path);
    }

    void progress() {
        fi_cq_read(cq_, nullptr, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

} // namespace avp_fabric

#endif
