#pragma once
#include <vector>
#include <stdexcept>
#include "Event.hpp"
#include "util.hpp"


class MultiEventWait {
private:
    struct pollfd *pfds_;
    size_t count_;
public:
    MultiEventWait(const std::vector<Event*> &events) {
        count_ = events.size();
        pfds_ = new struct pollfd[count_];
        struct pollfd *pfd = pfds_;
        for (Event* event: events) {
            pfd->fd = event->fd_;
            pfd->events = POLLIN;
            pfd->revents = 0;
            pfd++;
        }
    }
    MultiEventWait(const MultiEventWait &copyfrom) {
        count_ = copyfrom.count_;
        pfds_ = new struct pollfd[count_];
        std::copy(copyfrom.pfds_, copyfrom.pfds_+copyfrom.count_, pfds_);
    }
    MultiEventWait(MultiEventWait &&movefrom) {
        count_ = movefrom.count_;
        pfds_ = movefrom.pfds_;
        movefrom.count_ = 0;
        movefrom.pfds_ = nullptr;
    }
    ~MultiEventWait() {
        if (pfds_ != nullptr) delete[] pfds_;
    }
    bool wait(int timeout_ms = -1) { // returns false for timeout, true if at least 1 event was signalled
        bool result = false;
        int ret = poll(pfds_, count_, timeout_ms);
        if (ret<0) {
            throw Error("wait: poll error");
        } else if (ret>0) {
            int64_t blackhole;
            for (size_t i=0; i<count_; i++) {
                if (pfds_[i].revents & POLLIN) {
                    result = true;
                    read(pfds_[i].fd, &blackhole, sizeof blackhole);
                }
                pfds_[i].revents = 0;
            }
        }
        return result;
    }
};
