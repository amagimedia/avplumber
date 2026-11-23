#pragma once

#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include "fabric_wait.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/buffer.h>
}

#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace avp_fabric {

// Receive buffers live here, not in the receiver: a packet handed downstream keeps its buffer
// (and so this pool) alive after the node is gone. Returned buffers are queued from any thread
// and reposted by the receiving thread, the only thread that calls libfabric.
struct RxPool {
    uint8_t *slab = nullptr;
    size_t stride = 0;
    size_t count = 0;
    int wake_fd = -1;           // eventfd: a buffer came back, or the receiver is stopping
    std::mutex mutex;
    std::vector<uint32_t> returned;

    RxPool(size_t count_, size_t stride_): stride(stride_), count(count_) {
        if (posix_memalign(reinterpret_cast<void **>(&slab), 4096, stride * count) != 0) {
            throw Error("fabric receive buffer allocation failed");
        }
        wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd < 0) {
            free(slab);
            throw Error("fabric receive eventfd failed");
        }
    }
    ~RxPool() {
        free(slab);
        if (wake_fd >= 0) close(wake_fd);
    }
    uint8_t *data(uint32_t i) const { return slab + static_cast<size_t>(i) * stride; }
    void wake() {
        const uint64_t one = 1;
        (void)!write(wake_fd, &one, sizeof(one));
    }
    void drainWake() {
        uint64_t v;
        while (read(wake_fd, &v, sizeof(v)) > 0) {}
    }
    void giveBack(uint32_t i) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            returned.push_back(i);
        }
        wake();
    }
};

struct RxLease {
    std::shared_ptr<RxPool> pool;
    uint32_t index;
};

inline void rxRelease(void *opaque, uint8_t *) {
    auto *lease = static_cast<RxLease *>(opaque);
    lease->pool->giveBack(lease->index);
    delete lease;
}

class RdmReceiver {
public:
    struct Message {
        uint32_t slot = 0;
        const uint8_t *data = nullptr;
        size_t len = 0;
    };

private:
    struct Slot {
        fi_context2 ctx;        // first member: the completion's op_context is this slot
        uint32_t index = 0;
        bool posted = false;
    };

    fi_info *info_ = nullptr;
    fid_fabric *fabric_ = nullptr;
    fid_domain *domain_ = nullptr;
    fid_av *av_ = nullptr;
    fid_cq *cq_ = nullptr;
    fid_ep *ep_ = nullptr;
    CqWaiter waiter_;
    std::shared_ptr<RxPool> pool_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> deferred_;     // reposts the provider refused with -FI_EAGAIN
    size_t max_msg_ = 0;
    size_t posted_ = 0;
    size_t low_water_ = 0;
    std::atomic_bool stopping_{false};
    std::string log_prefix_;
    uint64_t copied_under_pressure_ = 0;
    uint64_t receive_errors_ = 0;
    // Set when a receive fails because the peer's connection went away. Best effort: the tcp
    // provider does not deliver FI_SHUTDOWN for an RDM endpoint (its per-peer connection endpoints
    // are created without the application's EQ), so only a posted receive that tears mid-record
    // reports here. A clean or idle close is silent; record cadence and idle_timeout_ms carry
    // liveness. The endpoint stays open, so the same peer (or a new one) can send again; a record
    // clears the flag.
    std::atomic_bool peer_closed_{false};
    uint64_t peer_closes_ = 0;

    static void check(int ret, const std::string &what) {
        if (ret < 0) {
            throw Error(what + " failed: " + fi_strerror(-ret));
        }
    }

    // Errors that mean "the peer's connection ended", not "this receiver is broken": the
    // provider's own error codes are the platform's errno values (see fi_errno.h).
    static bool isConnectionLoss(int err) {
        switch (err) {
            case FI_ECONNRESET:
            case FI_ECONNREFUSED:
            case FI_ENOTCONN:
            case FI_ESHUTDOWN:
            case FI_ETIMEDOUT:
            case FI_ECONNABORTED:
                return true;
            default:
                return false;
        }
    }

public:
    RdmReceiver(const std::string &provider,
                const std::string &bind_host,
                const std::string &port,
                const std::string &addr_file,
                size_t max_msg_size,
                size_t rx_buffers,
                std::string log_prefix):
        max_msg_(max_msg_size),
        log_prefix_(std::move(log_prefix)) {
        fi_info *hints = fi_allocinfo();
        if (!hints) throw Error("fi_allocinfo failed");
        hints->caps = FI_MSG;
        hints->mode = FI_CONTEXT | FI_CONTEXT2;
        hints->ep_attr->type = FI_EP_RDM;
        hints->fabric_attr->prov_name = strdup(provider.c_str());
        hints->rx_attr->size = rx_buffers;

        const char *node = bind_host.empty() ? nullptr : bind_host.c_str();
        int ret = fi_getinfo(FI_VERSION(1, 11), node, port.c_str(), FI_SOURCE, hints, &info_);
        fi_freeinfo(hints);
        check(ret, "fi_getinfo");
        if (!info_) throw Error("fi_getinfo returned no provider info");

        size_t count = std::max<size_t>(rx_buffers, 2);
        if (info_->rx_attr && info_->rx_attr->size && info_->rx_attr->size < count) {
            count = info_->rx_attr->size;
        }
        low_water_ = std::max<size_t>(count / 4, 1);

        check(fi_fabric(info_->fabric_attr, &fabric_, nullptr), "fi_fabric");
        check(fi_domain(fabric_, info_, &domain_, nullptr), "fi_domain");

        fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_MSG;
        cq_attr.size = count;
        bool has_fd = false;
        cq_ = CqWaiter::open(domain_, cq_attr, has_fd);

        fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        av_attr.count = 1;
        check(fi_av_open(domain_, &av_attr, &av_, nullptr), "fi_av_open");

        check(fi_endpoint(domain_, info_, &ep_, nullptr), "fi_endpoint");
        check(fi_ep_bind(ep_, &cq_->fid, FI_RECV | FI_SEND), "fi_ep_bind cq");
        check(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind av");
        check(fi_enable(ep_), "fi_enable");
        waiter_.attach(fabric_, cq_, has_fd);

        if (!addr_file.empty()) {
            writeLocalAddress(addr_file);
        }

        // Room for the decoder's input padding after the largest message.
        const size_t stride = (max_msg_ + AV_INPUT_BUFFER_PADDING_SIZE + 4095) / 4096 * 4096;
        pool_ = std::make_shared<RxPool>(count, stride);
        slots_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            slots_[i].index = i;
            post(i);
        }

        logstream << log_prefix_ << " opened provider=" << info_->fabric_attr->prov_name
                  << " fabric_version=" << FI_MAJOR(fi_version()) << "." << FI_MINOR(fi_version())
                  << " bind_host=" << (bind_host.empty() ? "*" : bind_host)
                  << " port=" << port
                  << " addr_file=" << (addr_file.empty() ? "-" : addr_file)
                  << " max_msg_size=" << max_msg_size
                  << " fi_max_msg_size=" << (info_->ep_attr ? info_->ep_attr->max_msg_size : 0)
                  << " rx_buffers=" << count
                  << " wait=" << (has_fd ? "fd" : "poll");
    }

    ~RdmReceiver() {
        // Closing the endpoint cancels the posted receives; buffers held by packets stay valid
        // in the pool until those packets are freed.
        if (ep_) fi_close(&ep_->fid);
        if (av_) fi_close(&av_->fid);
        if (cq_) fi_close(&cq_->fid);
        if (domain_) fi_close(&domain_->fid);
        if (fabric_) fi_close(&fabric_->fid);
        if (info_) fi_freeinfo(info_);
        if (copied_under_pressure_ || receive_errors_ || peer_closes_) {
            logstream << log_prefix_ << " summary copied_under_pressure=" << copied_under_pressure_
                      << " receive_errors=" << receive_errors_
                      << " peer_closes=" << peer_closes_;
        }
    }

    void stop() {
        stopping_.store(true, std::memory_order_release);
        pool_->wake();
    }

    bool stopping() const {
        return stopping_.load(std::memory_order_acquire);
    }

    // True while the provider has reported a connection loss and no record has arrived since.
    bool peerClosed() const {
        return peer_closed_.load(std::memory_order_acquire);
    }

    uint64_t peerCloses() const {
        return peer_closes_;
    }

    // One completed receive, without blocking. Reposts buffers that came back first.
    bool tryNext(Message &m) {
        repostReturned();
        fi_cq_msg_entry entry = {};
        int ret = fi_cq_read(cq_, &entry, 1);
        if (ret == -FI_EAGAIN) return false;
        if (ret == -FI_EAVAIL) {
            handleError();
            return false;
        }
        check(ret, "fi_cq_read");
        if (ret != 1) throw Error("fi_cq_read returned unexpected count");
        if (peer_closed_.load(std::memory_order_acquire)) {
            peer_closed_.store(false, std::memory_order_release);
            logstream << log_prefix_ << " peer back after " << peer_closes_ << " close(s)";
        }
        auto *slot = static_cast<Slot *>(entry.op_context);
        slot->posted = false;
        --posted_;
        m.slot = slot->index;
        m.data = pool_->data(slot->index);
        m.len = entry.len;
        return true;
    }

    // Adds this receiver's wait descriptors. Returns false when completions are already pending,
    // in which case the caller must not block.
    bool addWaitFds(std::vector<pollfd> &fds) {
        fds.push_back({pool_->wake_fd, POLLIN, 0});
        return waiter_.addFd(fds);
    }

    bool hasWaitFd() const {
        return waiter_.hasFd();
    }

    void consumeWake() {
        pool_->drainWake();
    }

    // Blocks until a message, a returned buffer or stop, at most timeout_ms.
    bool next(Message &m, int timeout_ms) {
        while (!stopping()) {
            if (tryNext(m)) return true;
            std::vector<pollfd> fds;
            const bool can_block = addWaitFds(fds);
            if (can_block) {
                ::poll(fds.data(), fds.size(), hasWaitFd() ? timeout_ms : std::min(timeout_ms, 1));
                consumeWake();
            }
            if (timeout_ms == 0) return false;
        }
        return false;
    }

    // The payload as a packet that references the receive buffer until the packet is freed.
    // When few receives are left posted, copies and reposts at once instead, so the provider
    // never runs out of buffers while the decoder holds packets.
    av::Packet takePayload(const Message &m, size_t offset, size_t size) {
        uint8_t *base = pool_->data(m.slot);
        if (posted_ + deferred_.size() <= low_water_) {
            av::Packet copy(base + offset, size, true);
            release(m);
            if (++copied_under_pressure_ == 1 || copied_under_pressure_ % 1000 == 0) {
                logstream << log_prefix_ << " receive buffers low, copied payloads="
                          << copied_under_pressure_;
            }
            return copy;
        }
        std::memset(base + m.len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        auto *lease = new RxLease{pool_, m.slot};
        AVBufferRef *buf = av_buffer_create(base, pool_->stride, rxRelease, lease, 0);
        if (!buf) {
            delete lease;
            throw Error(log_prefix_ + " av_buffer_create failed");
        }
        AVPacket *p = av_packet_alloc();
        if (!p) {
            av_buffer_unref(&buf);
            throw Error(log_prefix_ + " av_packet_alloc failed");
        }
        p->buf = buf;
        p->data = base + offset;
        p->size = static_cast<int>(size);
        av::Packet out(p);      // takes its own reference to buf
        av_packet_free(&p);
        return out;
    }

    // Returns a message's buffer without handing it downstream.
    void release(const Message &m) {
        post(m.slot);
    }

private:
    void post(uint32_t i) {
        Slot &s = slots_[i];
        int ret = fi_recv(ep_, pool_->data(i), max_msg_, nullptr, FI_ADDR_UNSPEC, &s.ctx);
        if (ret == -FI_EAGAIN) {
            deferred_.push_back(i);
            return;
        }
        check(ret, "fi_recv");
        s.posted = true;
        ++posted_;
    }

    void repostReturned() {
        std::vector<uint32_t> back;
        {
            std::lock_guard<std::mutex> lock(pool_->mutex);
            back.swap(pool_->returned);
        }
        if (!deferred_.empty()) {
            back.insert(back.end(), deferred_.begin(), deferred_.end());
            deferred_.clear();
        }
        for (uint32_t i: back) post(i);
    }

    void handleError() {
        fi_cq_err_entry err = {};
        int ret = fi_cq_readerr(cq_, &err, 0);
        if (ret < 0) check(ret, "fi_cq_readerr");
        auto *slot = static_cast<Slot *>(err.op_context);
        if (slot) {
            slot->posted = false;
            --posted_;
        }
        if (err.err == FI_ECANCELED && stopping()) return;
        if (isConnectionLoss(err.err)) {
            if (stopping()) return;
            // The peer left. Not a receiver failure: keep the endpoint and its posted receives so
            // the peer can return (or a new one can connect). On tcp/RDM only a torn in-flight
            // record reports here, so this is diagnostic; the caller decides what record silence
            // means (fabric_source ends the session after idle_timeout_ms).
            ++peer_closes_;
            peer_closed_.store(true, std::memory_order_release);
            logstream << log_prefix_ << " peer closed: " << fi_strerror(err.err)
                      << " (" << fi_cq_strerror(cq_, err.prov_errno, err.err_data, nullptr, 0) << ")"
                      << " closes=" << peer_closes_;
            if (slot) post(slot->index);
            return;
        }
        ++receive_errors_;
        logstream << log_prefix_ << " receive error: " << fi_strerror(err.err)
                  << " (" << fi_cq_strerror(cq_, err.prov_errno, err.err_data, nullptr, 0) << ")"
                  << " len=" << err.len << " errors=" << receive_errors_;
        if (slot) post(slot->index);
        if (err.err != FI_ETRUNC && err.err != FI_ECANCELED) {
            throw Error(log_prefix_ + " receive failed: " + fi_strerror(err.err));
        }
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
};

} // namespace avp_fabric

#endif
