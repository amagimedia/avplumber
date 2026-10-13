#pragma once
#include <cerrno>
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
    std::vector<struct epoll_event> ready_;
public:
    MultiEventWait(const std::vector<Event*> &events) {
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) throw Error("wait: epoll_create1 error");
        size_t count = 0;
        for (Event* event: events) {
            if (event->fd_ < 0) continue;
            struct epoll_event registration = {};
            registration.events = EPOLLIN;   // level-triggered, as poll() was
            registration.data.fd = event->fd_;
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
    bool wait(int timeout_ms = -1) { // returns false for timeout, true if at least 1 event was signalled
        int ret = epoll_wait(epoll_fd_, ready_.data(), static_cast<int>(ready_.size()), timeout_ms);
        if (ret<0) {
            throw Error("wait: poll error");
        }
        int64_t blackhole;
        for (int i=0; i<ret; i++) {
            read(ready_[i].data.fd, &blackhole, sizeof blackhole);
        }
        return ret>0;
    }
};
