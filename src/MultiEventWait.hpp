#pragma once
#include <cerrno>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <sys/epoll.h>
#include "Event.hpp"
#include "util.hpp"

// Waits for any of a fixed set of events. The set is registered with the kernel once, so a wait
// costs the events that fired and not the size of the set: a mixer compositor waits on one event
// per source and is woken a few thousand times a second.
class MultiEventWait {
private:
    int epoll_fd_ = -1;
    std::vector<int> fds_;   // by place in the list given at construction
    std::vector<struct epoll_event> ready_;
public:
    MultiEventWait(const std::vector<Event*> &events) {
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) throw Error("wait: epoll_create1 error");
        size_t count = 0;
        fds_.reserve(events.size());
        for (Event* event: events) {
            fds_.push_back(event->fd_);
            if (event->fd_ < 0) continue;
            struct epoll_event registration = {};
            registration.events = EPOLLIN;   // level-triggered, as poll() was
            registration.data.u32 = static_cast<uint32_t>(fds_.size() - 1);
            if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event->fd_, &registration) == 0) {
                count++;
            } else if (errno != EEXIST) {    // the same event listed twice is one registration
                close(epoll_fd_);
                throw Error("wait: epoll_ctl error");
            }
        }
        ready_.resize(count ? count : 1);
    }
    MultiEventWait(const MultiEventWait &copyfrom) = delete;
    MultiEventWait& operator=(const MultiEventWait &copyfrom) = delete;
    ~MultiEventWait() {
        if (epoll_fd_ >= 0) close(epoll_fd_);
    }
    // returns false for timeout, true if at least 1 event was signalled.
    // `signalled`, when given, receives the places (in the list given at construction) of the
    // events that were; an event listed more than once is reported at its first place.
    bool wait(int timeout_ms = -1, std::vector<uint32_t> *signalled = nullptr) {
        if (signalled) signalled->clear();
        // A process stopped and resumed (SIGSTOP, a debugger, a frozen container) or a signal handler
        // makes epoll_wait return EINTR, and it is never restarted: wait out the rest of the timeout.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        int ret;
        while ((ret = epoll_wait(epoll_fd_, ready_.data(), static_cast<int>(ready_.size()), timeout_ms)) < 0) {
            if (errno != EINTR) throw Error("wait: poll error");
            if (timeout_ms > 0) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
                timeout_ms = left > 0 ? static_cast<int>(left) : 0;
            }
        }
        int64_t blackhole;
        for (int i=0; i<ret; i++) {
            read(fds_[ready_[i].data.u32], &blackhole, sizeof blackhole);
            Event::consumed(fds_[ready_[i].data.u32]);
            if (signalled) signalled->push_back(ready_[i].data.u32);
        }
        return ret>0;
    }
};
