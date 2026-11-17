#pragma once

#include "../node_common.hpp"

#if HAVE_LIBFABRIC

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#include <poll.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace avp_fabric {

// Blocking on a completion queue instead of sleeping 1 ms between reads. Providers with a file
// descriptor wait object (tcp, rxm) block in poll(); shm has none (only FI_WAIT_NONE/YIELD), so
// there the caller polls the queue at most every millisecond, as before.
class CqWaiter {
    fid_fabric *fabric_ = nullptr;
    fid_cq *cq_ = nullptr;
    int fd_ = -1;

public:
    static fid_cq *open(fid_domain *domain, fi_cq_attr attr, bool &has_fd) {
        fid_cq *cq = nullptr;
        attr.wait_obj = FI_WAIT_FD;
        int ret = fi_cq_open(domain, &attr, &cq, nullptr);
        if (ret == 0) {
            has_fd = true;
            return cq;
        }
        attr.wait_obj = FI_WAIT_NONE;
        ret = fi_cq_open(domain, &attr, &cq, nullptr);
        if (ret < 0) {
            throw Error(std::string("fi_cq_open failed: ") + fi_strerror(-ret));
        }
        has_fd = false;
        return cq;
    }

    void attach(fid_fabric *fabric, fid_cq *cq, bool has_fd) {
        fabric_ = fabric;
        cq_ = cq;
        fd_ = -1;
        if (has_fd) {
            int ret = fi_control(&cq->fid, FI_GETWAIT, &fd_);
            if (ret < 0) {
                logstream << "fabric: FI_GETWAIT failed (" << fi_strerror(-ret) << "), polling instead";
                fd_ = -1;
            }
        }
    }

    bool hasFd() const {
        return fd_ >= 0;
    }

    // Adds the queue's descriptor. Returns false when completions are already pending: then the
    // caller must read the queue again instead of blocking.
    bool addFd(std::vector<pollfd> &fds) {
        if (fd_ < 0) return true;
        fid *f = &cq_->fid;
        if (fi_trywait(fabric_, &f, 1) != FI_SUCCESS) return false;
        fds.push_back({fd_, POLLIN, 0});
        return true;
    }

    // Waits for a completion, or for one of extra_fds, at most timeout_ms.
    void wait(int timeout_ms, std::vector<pollfd> extra_fds = {}) {
        if (!addFd(extra_fds)) return;
        if (fd_ < 0) timeout_ms = std::min(timeout_ms, 1);
        if (extra_fds.empty()) {
            if (timeout_ms > 0) Wallclock::sleepms(timeout_ms);
            return;
        }
        ::poll(extra_fds.data(), extra_fds.size(), timeout_ms);
    }
};

inline int64_t remainingMs(std::chrono::steady_clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now()).count();
    return std::max<int64_t>(left, 0);
}

} // namespace avp_fabric

#endif
